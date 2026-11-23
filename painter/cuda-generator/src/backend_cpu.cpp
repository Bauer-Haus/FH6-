// The layer search on the CPU, step for step what the CUDA kernels do. It is
// slow, and exists so the GPU can be checked against it.
#include <algorithm>
#include <stdexcept>

#include "backend.hpp"
#include "tables.hpp"

namespace fpg {
namespace {

class CpuBackend : public Backend {
public:
    std::string describe() const override { return "CPU (reference)"; }

    void init(const PreparedImage& img, const RunConfig& cfg) override {
        w_ = img.width;
        h_ = img.height;
        cfg_ = cfg;
        target_ = img.target;
        current_ = img.current;
        mask_ = img.mask;
        st_ = build_static_table(target_.data(), mask_.data(), w_, h_);
        dyn_ = build_dynamic_table(target_.data(), current_.data(), mask_.data(), w_, h_);
        grid_.assign((size_t)cfg_.gridW * cfg_.gridH, 0.0f);
        cdf_.assign(grid_.size(), 0.0);
        refresh_error_map(nullptr);
    }

    // Scores one shape the way a GPU warp does, including the order in which
    // the lanes' partial sums are combined.
    float score(Candidate& c, bool roundColor) const {
        Geom g = make_geom(c, w_, h_);
        Stats p[kWarp];
        for (int l = 0; l < kWarp; ++l) {
            p[l].clear();
            accumulate_lane(dyn_.data(), st_.data(), w_, g, l, p[l]);
        }
        for (int o = kWarp / 2; o > 0; o >>= 1) {
            Stats q[kWarp];
            for (int l = 0; l < kWarp; ++l) {
                q[l] = p[l];
                q[l].add(p[l ^ o]);
            }
            for (int l = 0; l < kWarp; ++l) p[l] = q[l];
        }
        return eval_stats(p[0], c.a, roundColor, c);
    }

    LayerResult run_layer(const LayerParams& p) override {
        const int R = cfg_.randomCount, K = cfg_.starts, M = cfg_.perRound;
        LayerResult res = {};
        cands_.resize((size_t)R);
        scores_.resize((size_t)R);
        keys_.resize((size_t)R);
        for (int i = 0; i < R; ++i) {
            Candidate c = random_candidate(p, cdf_.data(), (uint32_t)i);
            float s = score(c, false);
            cands_[i] = c;
            scores_[i] = s;
            keys_[i] = score_key(s + size_penalty(c, p.acceptedShapes), (uint32_t)i);
        }
        std::partial_sort(keys_.begin(), keys_.begin() + K, keys_.end());

        std::vector<Candidate> cur((size_t)K);
        std::vector<float> curScore((size_t)K), curAdj((size_t)K);
        for (int k = 0; k < K; ++k) {
            uint32_t idx = key_index(keys_[k]);
            cur[k] = cands_[idx];
            curScore[k] = scores_[idx];
            curAdj[k] = scores_[idx] + size_penalty(cands_[idx], p.acceptedShapes);
        }
        res.randomBest = curScore[0];

        for (int r = 0; r < cfg_.rounds; ++r) {
            for (int k = 0; k < K; ++k) {
                if (!(curScore[k] < 0.0f)) continue;
                float bestAdj = 0, bestScore = 0;
                int bestM = -1;
                Candidate bestC = {};
                for (int m = 0; m < M; ++m) {
                    const uint64_t j = (uint64_t)k * M + m;
                    Rng rng = make_rng(p.seed, p.attempt, 2u, (uint64_t)r * K * M + j);
                    Candidate c = mutate(cur[k], rng, w_, h_, cfg_.forceOpaque != 0,
                                         cfg_.steps.move[r], cfg_.steps.radius[r],
                                         cfg_.steps.theta[r]);
                    float s = score(c, false);
                    float adj = s + size_penalty(c, p.acceptedShapes);
                    if (bestM < 0 || better(adj, m, bestAdj, bestM)) {
                        bestAdj = adj;
                        bestScore = s;
                        bestM = m;
                        bestC = c;
                    }
                }
                if (bestM >= 0 && bestAdj <= curAdj[k]) {
                    if (bestAdj < curAdj[k]) ++res.improvements;
                    cur[k] = bestC;
                    curScore[k] = bestScore;
                    curAdj[k] = bestAdj;
                }
            }
        }

        int kb = 0;
        for (int k = 1; k < K; ++k)
            if (better(curAdj[k], k, curAdj[kb], kb)) kb = k;
        res.climbBest = curScore[kb];
        if (!(res.climbBest < kMinImproveDelta)) {
            res.status = kNoImprovement;
            return res;
        }
        Candidate fin = quantize(cur[kb], w_, h_, cfg_.forceOpaque);
        res.finalScore = score(fin, true);
        res.final = fin;
        if (is_rejected(res.finalScore)) res.status = kQuantizedRejected;
        else if (!(res.finalScore < kMinImproveDelta)) res.status = kQuantizedNoGain;
        else res.status = kApplied;
        if (res.status == kApplied) {
            Geom g = paint(fin);
            refresh_error_map(&g);
        }
        return res;
    }

    void apply_shapes(const std::vector<Candidate>& shapes) override {
        for (const auto& c : shapes) paint(c);
        refresh_error_map(nullptr);
    }

    void reset_canvas(const std::vector<Px>& current) override {
        current_ = current;
        dyn_ = build_dynamic_table(target_.data(), current_.data(), mask_.data(), w_, h_);
        refresh_error_map(nullptr);
    }

    void read_current(std::vector<Px>& out) override { out = current_; }

    void score_batch(const std::vector<Candidate>& in, bool roundColor, std::vector<float>& scores,
                     std::vector<Candidate>& coloured) override {
        scores.resize(in.size());
        coloured = in;
        for (size_t i = 0; i < in.size(); ++i) scores[i] = score(coloured[i], roundColor);
    }

    void read_tables(std::vector<DynEntry>& dyn) override { dyn = dyn_; }
    void read_cdf(std::vector<double>& cdf) override { cdf = cdf_; }

private:
    // Paints a shape row run by row run and rewrites the changed row totals.
    Geom paint(const Candidate& c) {
        Geom g = make_geom(c, w_, h_);
        for (int y = g.yMin; y <= g.yMax; ++y) {
            int lo, hi;
            if (!row_span(g, y, lo, hi)) continue;
            for (int x = lo; x <= hi; ++x) {
                const long long px = (long long)y * w_ + x;
                if (mask_[px]) blend(current_[px], c);
            }
            rebuild_row(dyn_.data(), target_.data(), current_.data(), mask_.data(), w_, y, lo);
        }
        return g;
    }

    // Recomputes the error-map cells a shape touched (all cells if g is null)
    // and the CDF the random centres are drawn from.
    void refresh_error_map(const Geom* g) {
        const int gw = cfg_.gridW, gh = cfg_.gridH;
        for (int gy = 0; gy < gh; ++gy)
            for (int gx = 0; gx < gw; ++gx)
                if (!g || cell_overlaps(w_, h_, gw, gh, gx, gy, *g))
                    grid_[(size_t)gy * gw + gx] = grid_cell(target_.data(), current_.data(),
                                                            mask_.data(), w_, h_, gw, gh, gx, gy);
        build_cdf_cpu(grid_.data(), cdf_.data(), gw * gh);
    }

    int w_ = 0, h_ = 0;
    RunConfig cfg_;
    std::vector<Px> target_, current_;
    std::vector<uint8_t> mask_;
    std::vector<StatEntry> st_;
    std::vector<DynEntry> dyn_;
    std::vector<float> grid_;
    std::vector<double> cdf_;
    std::vector<Candidate> cands_;
    std::vector<float> scores_;
    std::vector<uint64_t> keys_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::unique_ptr<Backend>(new CpuBackend()); }

}  // namespace fpg
