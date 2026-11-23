// The layer search on an NVIDIA GPU.
//
// One layer is a fixed list of steps, recorded once as a CUDA graph and
// replayed for every layer:
//   1. k_random       make and score the random candidates (one warp each)
//   2. radix sort     order them by score + size penalty
//   3. k_init_starts  take the best K as hill-climb starts
//   4. k_mutate/k_accept, once per round: tweak every start M times, keep
//                     each start's best tweak if it is no worse
//   5. k_finalize     pick the best start, round it, score it again
//   6. k_apply_rows   paint it and rewrite the changed row totals
//   7. k_grid, k_cdf  update the error map the next layer samples from
// The CPU sends the layer's parameters in and reads one small result back.
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../backend.hpp"
#include "../tables.hpp"

namespace fpg {
namespace {

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock = 256;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}
#define CK(x) check((x), #x)

int blocks_for_warps(long long warps) { return (int)((warps * kWarp + kBlock - 1) / kBlock); }

// ------------------------------------------------------------------ kernels

__device__ __forceinline__ void warp_reduce(Stats& s) {
#pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) {
        Stats t;
        t.s0 = __shfl_xor_sync(kFull, s.s0, o);
        t.s1 = __shfl_xor_sync(kFull, s.s1, o);
        t.s2 = __shfl_xor_sync(kFull, s.s2, o);
        t.s3 = __shfl_xor_sync(kFull, s.s3, o);
        t.ss = __shfl_xor_sync(kFull, s.ss, o);
        t.ts = __shfl_xor_sync(kFull, s.ts, o);
        t.t0 = __shfl_xor_sync(kFull, s.t0, o);
        t.t1 = __shfl_xor_sync(kFull, s.t1, o);
        t.t2 = __shfl_xor_sync(kFull, s.t2, o);
        t.t3 = __shfl_xor_sync(kFull, s.t3, o);
        t.n = __shfl_xor_sync(kFull, s.n, o);
        t.nt = __shfl_xor_sync(kFull, s.nt, o);
        s.add(t);
    }
}

// Scores c with the whole warp; every lane returns the same score and colour.
__device__ float warp_score(const DynEntry* __restrict__ dyn, const StatEntry* __restrict__ st,
                            int width, int height, Candidate& c, bool roundColor, int lane) {
    Geom g = make_geom(c, width, height);
    Stats s;
    s.clear();
    accumulate_lane(dyn, st, width, g, lane, s);
    warp_reduce(s);
    return eval_stats(s, c.a, roundColor, c);
}

__device__ __forceinline__ DynEntry shfl_entry(const DynEntry& v, int src) {
    DynEntry o;
    o.s0 = __shfl_sync(kFull, v.s0, src);
    o.s1 = __shfl_sync(kFull, v.s1, src);
    o.s2 = __shfl_sync(kFull, v.s2, src);
    o.s3 = __shfl_sync(kFull, v.s3, src);
    o.ss = __shfl_sync(kFull, v.ss, src);
    o.ts = __shfl_sync(kFull, v.ts, src);
    return o;
}

// Rewrites row y's totals from x0 on, one warp per row (see rebuild_row).
__device__ void warp_rebuild_row(DynEntry* dyn, const Px* target, const Px* current,
                                 const uint8_t* mask, int width, int y, int x0, int lane) {
    DynEntry total = chunk_total(target, current, mask, width, y, x0, lane);
    DynEntry offset = dyn[(long long)y * ((long long)width + 1) + x0];
    for (int j = 0; j < kWarp; ++j) {
        DynEntry t = shfl_entry(total, j);
        if (j < lane) entry_add(offset, t);
    }
    chunk_write(dyn, target, current, mask, width, y, x0, lane, offset);
}

__global__ void __launch_bounds__(kBlock)
k_random(const LayerParams* __restrict__ P, const double* __restrict__ cdf,
         const DynEntry* __restrict__ dyn, const StatEntry* __restrict__ st,
         Candidate* __restrict__ cands, float* __restrict__ scores, uint64_t* __restrict__ keys,
         int count) {
    const int w = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (w >= count) return;
    const LayerParams p = *P;
    Candidate c = random_candidate(p, cdf, (uint32_t)w);
    const float s = warp_score(dyn, st, p.width, p.height, c, false, lane);
    if (lane == 0) {
        cands[w] = c;
        scores[w] = s;
        keys[w] = score_key(s + size_penalty(c, p.acceptedShapes), (uint32_t)w);
    }
}

__global__ void k_init_starts(const LayerParams* __restrict__ P, const uint64_t* __restrict__ sorted,
                              const Candidate* __restrict__ cands, const float* __restrict__ scores,
                              Candidate* cur, float* curScore, float* curAdj, int* improved,
                              LayerResult* res, int K) {
    const int k = threadIdx.x;
    if (k >= K) return;
    const uint32_t idx = key_index(sorted[k]);
    cur[k] = cands[idx];
    curScore[k] = scores[idx];
    curAdj[k] = scores[idx] + size_penalty(cands[idx], P->acceptedShapes);
    improved[k] = 0;
    if (k == 0) res->randomBest = scores[idx];
}

__global__ void __launch_bounds__(kBlock)
k_mutate(const LayerParams* __restrict__ P, const ClimbSteps* __restrict__ steps, int round,
         const Candidate* __restrict__ cur, const float* __restrict__ curScore,
         const DynEntry* __restrict__ dyn, const StatEntry* __restrict__ st,
         Candidate* __restrict__ mutC, float* __restrict__ mutS, float* __restrict__ mutAdj, int K,
         int M) {
    const int w = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (w >= K * M) return;
    const int k = w / M;
    if (!(curScore[k] < 0.0f)) return;  // this start found nothing to improve on
    const LayerParams p = *P;
    Rng rng = make_rng(p.seed, p.attempt, 2u, (uint64_t)round * K * M + (uint64_t)w);
    Candidate c = mutate(cur[k], rng, p.width, p.height, p.forceOpaque != 0, steps->move[round],
                         steps->radius[round], steps->theta[round]);
    const float s = warp_score(dyn, st, p.width, p.height, c, false, lane);
    if (lane == 0) {
        mutC[w] = c;
        mutS[w] = s;
        mutAdj[w] = s + size_penalty(c, p.acceptedShapes);
    }
}

// One warp per start: keep the best tweak of this round if it is no worse.
__global__ void k_accept(Candidate* cur, float* curScore, float* curAdj, int* improved,
                         const Candidate* __restrict__ mutC, const float* __restrict__ mutS,
                         const float* __restrict__ mutAdj, int K, int M) {
    const int k = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (k >= K) return;
    if (!(curScore[k] < 0.0f)) return;
    float ba = 0.0f;
    int bm = -1;
    for (int m = lane; m < M; m += kWarp) {
        const float a = mutAdj[k * M + m];
        if (bm < 0 || better(a, m, ba, bm)) {
            ba = a;
            bm = m;
        }
    }
    for (int o = kWarp / 2; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(kFull, ba, o);
        const int om = __shfl_xor_sync(kFull, bm, o);
        if (om >= 0 && (bm < 0 || better(oa, om, ba, bm))) {
            ba = oa;
            bm = om;
        }
    }
    if (lane == 0 && bm >= 0 && ba <= curAdj[k]) {
        if (ba < curAdj[k]) improved[k] += 1;
        cur[k] = mutC[k * M + bm];
        curScore[k] = mutS[k * M + bm];
        curAdj[k] = ba;
    }
}

__global__ void k_finalize(const LayerParams* __restrict__ P, const Candidate* __restrict__ cur,
                           const float* __restrict__ curScore, const float* __restrict__ curAdj,
                           const int* __restrict__ improved, int K,
                           const DynEntry* __restrict__ dyn, const StatEntry* __restrict__ st,
                           LayerResult* res, ApplyJob* job) {
    const int lane = threadIdx.x;
    const LayerParams p = *P;
    int kb = 0, imp = 0;
    for (int k = 0; k < K; ++k) {
        imp += improved[k];
        if (k > 0 && better(curAdj[k], k, curAdj[kb], kb)) kb = k;
    }
    const float climbBest = curScore[kb];
    if (!(climbBest < kMinImproveDelta)) {
        if (lane == 0) {
            res->climbBest = climbBest;
            res->finalScore = 0.0f;
            res->improvements = imp;
            res->status = kNoImprovement;
            job->enabled = 0;
        }
        return;
    }
    Candidate fin = quantize(cur[kb], p.width, p.height, p.forceOpaque != 0);
    const float fs = warp_score(dyn, st, p.width, p.height, fin, true, lane);
    if (lane == 0) {
        int status = kApplied;
        if (is_rejected(fs)) status = kQuantizedRejected;
        else if (!(fs < kMinImproveDelta)) status = kQuantizedNoGain;
        res->final = fin;
        res->climbBest = climbBest;
        res->finalScore = fs;
        res->improvements = imp;
        res->status = status;
        job->c = fin;
        job->enabled = status == kApplied ? 1 : 0;
    }
}

// One warp per image row: paint the row's run of the shape, then rewrite
// the row's totals.
__global__ void __launch_bounds__(kBlock)
k_apply_rows(const ApplyJob* __restrict__ job, const Px* __restrict__ target, Px* current,
             const uint8_t* __restrict__ mask, DynEntry* dyn, int width, int height) {
    const int y = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (y >= height || !job->enabled) return;
    const Candidate c = job->c;
    const Geom g = make_geom(c, width, height);
    int lo, hi;
    if (!row_span(g, y, lo, hi)) return;
    const long long p0 = (long long)y * width;
    for (int x = lo + lane; x <= hi; x += kWarp) {
        if (!mask[p0 + x]) continue;
        Px v = current[p0 + x];
        blend(v, c);
        current[p0 + x] = v;
    }
    __syncwarp();
    warp_rebuild_row(dyn, target, current, mask, width, y, lo, lane);
}

__global__ void __launch_bounds__(kBlock)
k_rebuild_all(const Px* __restrict__ target, const Px* __restrict__ current,
              const uint8_t* __restrict__ mask, DynEntry* dyn, int width, int height) {
    const int y = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (y >= height) return;
    warp_rebuild_row(dyn, target, current, mask, width, y, 0, lane);
}

__global__ void k_grid(const ApplyJob* __restrict__ job, const Px* __restrict__ target,
                       const Px* __restrict__ current, const uint8_t* __restrict__ mask,
                       float* grid, int width, int height, int gridW, int gridH, int full) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gridW * gridH) return;
    const int gx = i % gridW, gy = i / gridW;
    if (!full) {
        if (!job->enabled) return;
        const Geom g = make_geom(job->c, width, height);
        if (!cell_overlaps(width, height, gridW, gridH, gx, gy, g)) return;
    }
    grid[i] = grid_cell(target, current, mask, width, height, gridW, gridH, gx, gy);
}

// The CDF of the error map, one warp (see build_cdf_cpu).
__global__ void k_cdf(const ApplyJob* __restrict__ job, const float* __restrict__ grid, double* cdf,
                      int n, int full) {
    const int lane = threadIdx.x;
    if (!full && !job->enabled) return;
    const int chunk = (n + kWarp - 1) / kWarp;
    const int a = lane * chunk;
    const int b = (lane + 1) * chunk < n ? (lane + 1) * chunk : n;
    double total = 0.0;
    for (int i = a; i < b; ++i) total += grid[i] > 0.0f ? (double)grid[i] : 0.0;
    double offset = 0.0;
    for (int j = 0; j < kWarp; ++j) {
        const double t = __shfl_sync(kFull, total, j);
        if (j < lane) offset += t;
    }
    for (int i = a; i < b; ++i) {
        offset += grid[i] > 0.0f ? (double)grid[i] : 0.0;
        cdf[i] = offset;
    }
}

__global__ void __launch_bounds__(kBlock)
k_score_list(const Candidate* __restrict__ in, Candidate* out, float* scores, int n,
             const DynEntry* __restrict__ dyn, const StatEntry* __restrict__ st, int width,
             int height, int roundColor) {
    const int w = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (w >= n) return;
    Candidate c = in[w];
    const float s = warp_score(dyn, st, width, height, c, roundColor != 0, lane);
    if (lane == 0) {
        out[w] = c;
        scores[w] = s;
    }
}

// ------------------------------------------------------------------ backend

template <class T>
struct DevBuf {
    T* p = nullptr;
    size_t n = 0;
    void alloc(size_t count) {
        free();
        n = count;
        if (count) CK(cudaMalloc((void**)&p, count * sizeof(T)));
    }
    void free() {
        if (p) cudaFree(p);
        p = nullptr;
        n = 0;
    }
    ~DevBuf() { free(); }
};

class CudaBackend : public Backend {
public:
    CudaBackend() {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        if (e != cudaSuccess || count == 0)
            throw std::runtime_error(std::string("no NVIDIA GPU found (") +
                                     (e != cudaSuccess ? cudaGetErrorString(e) : "0 devices") +
                                     "). Update the NVIDIA driver, or run with --cpu.");
        device_ = 0;
        if (const char* d = std::getenv("FPG_DEVICE")) device_ = std::atoi(d);
        if (device_ < 0 || device_ >= count) device_ = 0;
        CK(cudaSetDevice(device_));
        CK(cudaGetDeviceProperties(&props_, device_));
        // A blocking stream, so plain cudaMemcpy calls stay in order with it.
        CK(cudaStreamCreate(&stream_));
        CK(cudaHostAlloc((void**)&hParams_, sizeof(LayerParams), cudaHostAllocDefault));
        CK(cudaHostAlloc((void**)&hResult_, sizeof(LayerResult), cudaHostAllocDefault));
        useGraph_ = !(std::getenv("FPG_NO_GRAPH") && std::string(std::getenv("FPG_NO_GRAPH")) == "1");
    }

    ~CudaBackend() override {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (hParams_) cudaFreeHost(hParams_);
        if (hResult_) cudaFreeHost(hResult_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::string describe() const override {
        char buf[256];
        std::snprintf(buf, sizeof buf, "CUDA (%s, sm_%d%d, %.0f MB)", props_.name, props_.major,
                      props_.minor, props_.totalGlobalMem / 1048576.0);
        return buf;
    }

    void init(const PreparedImage& img, const RunConfig& cfg) override {
        w_ = img.width;
        h_ = img.height;
        cfg_ = cfg;
        const size_t px = (size_t)w_ * h_;
        const size_t entries = (size_t)(w_ + 1) * h_;
        const size_t R = (size_t)cfg_.randomCount;
        const size_t KM = (size_t)cfg_.starts * std::max(1, cfg_.perRound);
        const size_t G = (size_t)cfg_.gridW * cfg_.gridH;

        size_t sortBytes = 0;
        CK(cub::DeviceRadixSort::SortKeys(nullptr, sortBytes, (const uint64_t*)nullptr,
                                          (uint64_t*)nullptr, (int)R, 0, 64, stream_));
        const double need = double(px) * (2 * sizeof(Px) + 1) +
                            double(entries) * (sizeof(DynEntry) + sizeof(StatEntry)) +
                            double(R) * (sizeof(Candidate) + sizeof(float) + 2 * sizeof(uint64_t)) +
                            double(KM) * (sizeof(Candidate) + 2 * sizeof(float)) + double(sortBytes) +
                            double(G) * (sizeof(float) + sizeof(double));
        size_t freeB = 0, totalB = 0;
        CK(cudaMemGetInfo(&freeB, &totalB));
        if (need > 0.95 * double(freeB)) {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "not enough GPU memory: this image and settings need about %.0f MB and "
                          "%.0f MB is free. Lower maxResolution or randomSamples.",
                          need / 1048576.0, freeB / 1048576.0);
            throw std::runtime_error(buf);
        }

        target_.alloc(px);
        current_.alloc(px);
        mask_.alloc(px);
        dyn_.alloc(entries);
        st_.alloc(entries);
        grid_.alloc(G);
        cdf_.alloc(G);
        cands_.alloc(R);
        scores_.alloc(R);
        keys_.alloc(R);
        sorted_.alloc(R);
        sortTemp_.alloc(std::max<size_t>(sortBytes, 1));
        sortBytes_ = sortBytes;
        cur_.alloc(cfg_.starts);
        curScore_.alloc(cfg_.starts);
        curAdj_.alloc(cfg_.starts);
        improved_.alloc(cfg_.starts);
        mutC_.alloc(KM);
        mutS_.alloc(KM);
        mutAdj_.alloc(KM);
        params_.alloc(1);
        result_.alloc(1);
        job_.alloc(1);
        steps_.alloc(1);

        CK(cudaMemcpy(target_.p, img.target.data(), px * sizeof(Px), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(mask_.p, img.mask.data(), px, cudaMemcpyHostToDevice));
        std::vector<StatEntry> st = build_static_table(img.target.data(), img.mask.data(), w_, h_);
        CK(cudaMemcpy(st_.p, st.data(), entries * sizeof(StatEntry), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(steps_.p, &cfg_.steps, sizeof(ClimbSteps), cudaMemcpyHostToDevice));
        ApplyJob off = {};
        CK(cudaMemcpy(job_.p, &off, sizeof off, cudaMemcpyHostToDevice));
        reset_canvas(img.current);
    }

    LayerResult run_layer(const LayerParams& p) override {
        *hParams_ = p;
        if (useGraph_) {
            if (!graphExec_) {
                CK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                enqueue_layer();
                CK(cudaStreamEndCapture(stream_, &graph_));
                CK(cudaGraphInstantiate(&graphExec_, graph_, 0));
            }
            CK(cudaGraphLaunch(graphExec_, stream_));
        } else {
            enqueue_layer();
        }
        CK(cudaStreamSynchronize(stream_));
        CK(cudaGetLastError());
        return *hResult_;
    }

    void apply_shapes(const std::vector<Candidate>& shapes) override {
        if (!shapes.empty()) {
            std::vector<ApplyJob> jobs(shapes.size());
            for (size_t i = 0; i < shapes.size(); ++i) {
                jobs[i] = ApplyJob{};
                jobs[i].c = shapes[i];
                jobs[i].enabled = 1;
            }
            DevBuf<ApplyJob> dj;
            dj.alloc(jobs.size());
            CK(cudaMemcpy(dj.p, jobs.data(), jobs.size() * sizeof(ApplyJob), cudaMemcpyHostToDevice));
            for (size_t i = 0; i < jobs.size(); ++i) {
                k_apply_rows<<<blocks_for_warps(h_), kBlock, 0, stream_>>>(
                    dj.p + i, target_.p, current_.p, mask_.p, dyn_.p, w_, h_);
            }
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(stream_));
        }
        refresh_full();
    }

    void reset_canvas(const std::vector<Px>& current) override {
        const size_t px = (size_t)w_ * h_;
        CK(cudaMemcpy(current_.p, current.data(), px * sizeof(Px), cudaMemcpyHostToDevice));
        CK(cudaMemsetAsync(dyn_.p, 0, dyn_.n * sizeof(DynEntry), stream_));
        k_rebuild_all<<<blocks_for_warps(h_), kBlock, 0, stream_>>>(target_.p, current_.p, mask_.p,
                                                                    dyn_.p, w_, h_);
        CK(cudaGetLastError());
        refresh_full();
    }

    void read_current(std::vector<Px>& out) override {
        out.resize((size_t)w_ * h_);
        CK(cudaMemcpy(out.data(), current_.p, out.size() * sizeof(Px), cudaMemcpyDeviceToHost));
    }

    void score_batch(const std::vector<Candidate>& in, bool roundColor, std::vector<float>& scores,
                     std::vector<Candidate>& coloured) override {
        const int n = (int)in.size();
        scores.assign(in.size(), 0.0f);
        coloured.assign(in.size(), Candidate{});
        if (n == 0) return;
        DevBuf<Candidate> din, dout;
        DevBuf<float> ds;
        din.alloc(in.size());
        dout.alloc(in.size());
        ds.alloc(in.size());
        CK(cudaMemcpy(din.p, in.data(), in.size() * sizeof(Candidate), cudaMemcpyHostToDevice));
        k_score_list<<<blocks_for_warps(n), kBlock, 0, stream_>>>(din.p, dout.p, ds.p, n, dyn_.p,
                                                                   st_.p, w_, h_, roundColor ? 1 : 0);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(stream_));
        CK(cudaMemcpy(scores.data(), ds.p, in.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(coloured.data(), dout.p, in.size() * sizeof(Candidate), cudaMemcpyDeviceToHost));
    }

    void read_tables(std::vector<DynEntry>& dyn) override {
        dyn.resize(dyn_.n);
        CK(cudaMemcpy(dyn.data(), dyn_.p, dyn_.n * sizeof(DynEntry), cudaMemcpyDeviceToHost));
    }

    void read_cdf(std::vector<double>& cdf) override {
        cdf.resize(cdf_.n);
        CK(cudaMemcpy(cdf.data(), cdf_.p, cdf_.n * sizeof(double), cudaMemcpyDeviceToHost));
    }

private:
    void enqueue_layer() {
        const int R = cfg_.randomCount, K = cfg_.starts, M = cfg_.perRound;
        CK(cudaMemcpyAsync(params_.p, hParams_, sizeof(LayerParams), cudaMemcpyHostToDevice, stream_));
        k_random<<<blocks_for_warps(R), kBlock, 0, stream_>>>(params_.p, cdf_.p, dyn_.p, st_.p,
                                                              cands_.p, scores_.p, keys_.p, R);
        size_t bytes = sortBytes_;
        CK(cub::DeviceRadixSort::SortKeys(sortTemp_.p, bytes, keys_.p, sorted_.p, R, 0, 64, stream_));
        k_init_starts<<<1, kWarp, 0, stream_>>>(params_.p, sorted_.p, cands_.p, scores_.p, cur_.p,
                                                curScore_.p, curAdj_.p, improved_.p, result_.p, K);
        for (int r = 0; r < cfg_.rounds && M > 0; ++r) {
            k_mutate<<<blocks_for_warps((long long)K * M), kBlock, 0, stream_>>>(
                params_.p, steps_.p, r, cur_.p, curScore_.p, dyn_.p, st_.p, mutC_.p, mutS_.p,
                mutAdj_.p, K, M);
            k_accept<<<1, K * kWarp, 0, stream_>>>(cur_.p, curScore_.p, curAdj_.p, improved_.p,
                                                   mutC_.p, mutS_.p, mutAdj_.p, K, M);
        }
        k_finalize<<<1, kWarp, 0, stream_>>>(params_.p, cur_.p, curScore_.p, curAdj_.p, improved_.p,
                                             K, dyn_.p, st_.p, result_.p, job_.p);
        k_apply_rows<<<blocks_for_warps(h_), kBlock, 0, stream_>>>(job_.p, target_.p, current_.p,
                                                                   mask_.p, dyn_.p, w_, h_);
        const int G = cfg_.gridW * cfg_.gridH;
        k_grid<<<(G + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(
            job_.p, target_.p, current_.p, mask_.p, grid_.p, w_, h_, cfg_.gridW, cfg_.gridH, 0);
        k_cdf<<<1, kWarp, 0, stream_>>>(job_.p, grid_.p, cdf_.p, G, 0);
        CK(cudaMemcpyAsync(hResult_, result_.p, sizeof(LayerResult), cudaMemcpyDeviceToHost, stream_));
        CK(cudaGetLastError());
    }

    void refresh_full() {
        const int G = cfg_.gridW * cfg_.gridH;
        k_grid<<<(G + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(
            job_.p, target_.p, current_.p, mask_.p, grid_.p, w_, h_, cfg_.gridW, cfg_.gridH, 1);
        k_cdf<<<1, kWarp, 0, stream_>>>(job_.p, grid_.p, cdf_.p, G, 1);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(stream_));
    }

    int device_ = 0;
    cudaDeviceProp props_ = {};
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    bool useGraph_ = true;
    LayerParams* hParams_ = nullptr;
    LayerResult* hResult_ = nullptr;

    int w_ = 0, h_ = 0;
    RunConfig cfg_;
    size_t sortBytes_ = 0;
    DevBuf<Px> target_, current_;
    DevBuf<uint8_t> mask_;
    DevBuf<DynEntry> dyn_;
    DevBuf<StatEntry> st_;
    DevBuf<float> grid_;
    DevBuf<double> cdf_;
    DevBuf<Candidate> cands_;
    DevBuf<float> scores_;
    DevBuf<uint64_t> keys_, sorted_;
    DevBuf<unsigned char> sortTemp_;
    DevBuf<Candidate> cur_;
    DevBuf<float> curScore_, curAdj_;
    DevBuf<int> improved_;
    DevBuf<Candidate> mutC_;
    DevBuf<float> mutS_, mutAdj_;
    DevBuf<LayerParams> params_;
    DevBuf<LayerResult> result_;
    DevBuf<ApplyJob> job_;
    DevBuf<ClimbSteps> steps_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend() { return std::unique_ptr<Backend>(new CudaBackend()); }

}  // namespace fpg
