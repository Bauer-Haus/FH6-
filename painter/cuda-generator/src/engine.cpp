#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "geometry_json.hpp"
#include "image_io.hpp"
#include "platform.hpp"
#include "tables.hpp"

namespace fpg {

namespace {
constexpr int kMaxNoImproveRetries = 100;
constexpr int kDefaultClimbStarts = 8;
constexpr int kDefaultClimbRounds = 8;
constexpr int kMaxClimbStarts = 32;  // one warp per start in a single block
constexpr int kMaxFinalPrunes = 5;
}  // namespace

void logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void mutation_steps(int width, int height, float& move, float& radius) {
    double diag = std::sqrt(double(width * width) + double(height * height));
    move = (float)std::max(2.0, diag * 0.012);
    radius = (float)std::max(2.0, diag * 0.010);
}

RunConfig plan_run(const Settings& cfg, int width, int height) {
    RunConfig rc;
    rc.randomCount = std::max(1, cfg.randomSamples);
    rc.starts = cfg.climbStarts > 0 ? cfg.climbStarts : kDefaultClimbStarts;
    rc.starts = std::min(rc.starts, std::min(kMaxClimbStarts, rc.randomCount));
    rc.rounds = cfg.climbRounds > 0 ? cfg.climbRounds : kDefaultClimbRounds;
    rc.rounds = std::min(rc.rounds, kMaxClimbRounds);
    if (cfg.mutatedSamples <= 0) {
        rc.rounds = 0;
        rc.perRound = 0;
    } else {
        // The same number of tweaks per layer as the Go generator spends.
        rc.perRound = std::max(1, cfg.mutatedSamples / (rc.starts * rc.rounds));
    }
    rc.gridW = std::max(1, std::min(cfg.errorGridSize, width));
    rc.gridH = std::max(1, std::min(cfg.errorGridSize, height));
    rc.forceOpaque = cfg.forceOpaqueShapes;
    float move, radius;
    mutation_steps(width, height, move, radius);
    // Tweak sizes shrink from the Go generator's steps to a fifth of them
    // over the rounds: wide moves first, fine-tuning last.
    for (int r = 0; r < kMaxClimbRounds; ++r) {
        double f = rc.rounds > 1 ? std::pow(0.2, double(r) / double(rc.rounds - 1)) : 1.0;
        rc.steps.move[r] = (float)(move * f);
        rc.steps.radius[r] = (float)(radius * f);
        rc.steps.theta[r] = (float)(30.0 * f);
    }
    return rc;
}

LayerParams layer_params(const RunConfig& rc, uint64_t seed, uint64_t attempt, int width,
                         int height, int acceptedShapes, int stopAt, bool forceOpaque) {
    LayerParams p = {};
    p.seed = seed;
    p.attempt = attempt;
    p.width = width;
    p.height = height;
    p.acceptedShapes = acceptedShapes;
    p.forceOpaque = forceOpaque ? 1 : 0;
    p.randomCount = rc.randomCount;
    p.starts = rc.starts;
    p.rounds = rc.rounds;
    p.perRound = rc.perRound;
    p.gridW = rc.gridW;
    p.gridH = rc.gridH;
    // randomCandidatesInto: the largest radius shrinks from 0.25 to 0.05 of
    // the diagonal as the run progresses.
    float progress = stopAt > 0 ? float(acceptedShapes) / float(stopAt) : 0.0f;
    if (progress < 0) progress = 0;
    if (progress > 1) progress = 1;
    float diag = (float)std::sqrt(double(width * width + height * height));
    float scaleFactor = (float)(0.25 - 0.20 * std::pow(double(progress), 1.5));
    float maxRadius = diag * scaleFactor;
    if (maxRadius < 4) maxRadius = 4;
    p.minRadius = 2.0f;
    p.maxRadius = maxRadius;
    return p;
}

double normalize_score(double totalError, double denom) {
    if (denom <= 0) return 0;
    double v = totalError / denom;
    if (v < 0) v = 0;
    return std::round(v * 1000000.0) / 1000000.0;
}

namespace {

ShapeOut background_shape(const PreparedImage& p, double score) {
    ShapeOut s;
    s.type = 1;
    s.data = {0, 0, double(p.width), double(p.height)};
    for (int k = 0; k < 4; ++k) s.color[k] = p.background[k];
    s.score = score;
    return s;
}

ShapeOut to_shape(const Candidate& c, double score) {
    ShapeOut s;
    s.type = 16;
    s.data = {double(c.x), double(c.y), double(c.rx), double(c.ry), double(quant_angle(c.theta))};
    s.color[0] = to_byte(c.r);
    s.color[1] = to_byte(c.g);
    s.color[2] = to_byte(c.b);
    s.color[3] = to_byte(c.a);
    s.score = score;
    return s;
}

Candidate shape_to_candidate(const ShapeOut& s) {
    Candidate c;
    c.x = (float)s.data[0];
    c.y = (float)s.data[1];
    c.rx = (float)s.data[2];
    c.ry = (float)s.data[3];
    c.theta = (float)s.data[4];
    c.r = float(s.color[0]) / 255.0f;
    c.g = float(s.color[1]) / 255.0f;
    c.b = float(s.color[2]) / 255.0f;
    c.a = float(s.color[3]) / 255.0f;
    return c;
}

bool looks_like_background(const ShapeOut& s) {
    if (s.type != 1 || s.data.size() < 4 || s.colorCount < 4) return false;
    if (s.color[3] <= 0) return true;
    return s.data[2] >= 1 && s.data[3] >= 1;
}

// Go's time.Duration.String() after Round(time.Millisecond).
std::string fmt_duration(double seconds) {
    long long ms = (long long)std::llround(seconds * 1000.0);
    if (ms == 0) return "0s";
    if (ms < 1000) return std::to_string(ms) + "ms";
    long long mins = ms / 60000;
    long long rem = ms % 60000;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%lld.%03lld", rem / 1000, rem % 1000);
    std::string s = buf;
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    s += "s";
    if (mins > 0) {
        long long hours = mins / 60;
        mins %= 60;
        s = std::to_string(mins) + "m" + s;
        if (hours > 0) s = std::to_string(hours) + "h" + s;
    }
    return s;
}

std::string resolve_output_base(const Options& opts) {
    return opts.outputPath.empty() ? opts.imagePath : opts.outputPath;
}

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void save_preview(Backend& be, const Options& opts, int w, int h, int step) {
    if (opts.previewPath.empty()) return;
    std::string ext = path_ext(opts.previewPath);
    std::string base = opts.previewPath.substr(0, opts.previewPath.size() - ext.size());
    std::vector<Px> cur;
    be.read_current(cur);
    std::string path = base + "." + std::to_string(step) + ".png";
    if (!save_png(path, cur, w, h)) throw std::runtime_error("write preview " + path);
}

double canvas_error(Backend& be, const PreparedImage& p, int& opaque) {
    std::vector<Px> cur;
    be.read_current(cur);
    return total_error(p.target.data(), cur.data(), p.mask.data(), p.width, p.height, opaque);
}

}  // namespace

void run(const Options& opts) {
    if (opts.imagePath.empty()) throw std::runtime_error("image path is required");
    const std::string root = opts.workspaceRoot.empty() ? "." : opts.workspaceRoot;

    std::string err;
    std::string settingsPath = resolve_settings_path(root, opts.settingsPath, opts.profile, err);
    if (settingsPath.empty()) throw std::runtime_error(err);
    Settings cfg;
    if (!parse_settings(settingsPath, cfg, err)) throw std::runtime_error(err);

    PreparedImage prepared;
    if (!load_and_prepare(opts.imagePath, cfg.maxResolution, prepared, err))
        throw std::runtime_error(err);
    const int W = prepared.width, H = prepared.height;

    RunConfig rc = plan_run(cfg, W, H);
    std::unique_ptr<Backend> be;
#ifdef FPG_WITH_CUDA
    if (!opts.useCpu) be = make_cuda_backend();
#endif
    if (!be) be = make_cpu_backend();
    be->init(prepared, rc);

    if (cfg.errorMetric == "ssim") {
        logf("Warning: the CUDA generator has no SSIM scoring; scoring with MSE.");
    }
    logf("Error metric: MSE");
    if (cfg.enableMultiPrimitiveShapes)
        logf("Warning: enableMultiPrimitiveShapes is not supported yet; using rotated ellipses only.");
    if (!cfg.ignoredKeys.empty()) {
        std::string keys;
        for (const auto& k : cfg.ignoredKeys) keys += (keys.empty() ? "" : ", ") + k;
        logf("Note: these settings have no effect in the CUDA generator: %s", keys.c_str());
    }
    logf("Backend: %s", be->describe().c_str());

    uint64_t seed = opts.seed != 0
                        ? (uint64_t)opts.seed
                        : (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    logf("Seed: %llu", (unsigned long long)seed);

    int opaquePixels = 0;
    double currentError = total_error(prepared.target.data(), prepared.current.data(),
                                      prepared.mask.data(), W, H, opaquePixels);
    double denom = double(std::max(1, opaquePixels * 4));

    std::vector<ShapeOut> shapes = {background_shape(prepared, normalize_score(currentError, denom))};
    int accepted = 0;

    std::string resumePath = opts.resumePath.empty() ? cfg.loadGeometry : opts.resumePath;
    if (!resumePath.empty()) {
        std::vector<ShapeOut> loaded;
        if (!load_geometry(resumePath, loaded, err)) throw std::runtime_error(err);
        if (loaded.empty()) throw std::runtime_error("checkpoint has no shapes");
        std::vector<Candidate> restore;
        for (size_t i = 0; i < loaded.size(); ++i) {
            const ShapeOut& s = loaded[i];
            if (i == 0 && looks_like_background(s)) continue;
            if (s.data.size() < 5 || s.colorCount < 4) continue;
            Candidate fin = quantize(shape_to_candidate(s), W, H, cfg.forceOpaqueShapes);
            restore.push_back(fin);
            shapes.push_back(to_shape(fin, s.score));
        }
        if (restore.empty()) throw std::runtime_error("checkpoint has no drawable shapes to restore");
        accepted = (int)restore.size();
        if (accepted >= cfg.stopAt) {
            throw std::runtime_error("checkpoint already has " + std::to_string(accepted) +
                                     " shapes (target stopAt=" + std::to_string(cfg.stopAt) + ")");
        }
        be->apply_shapes(restore);
        currentError = canvas_error(*be, prepared, opaquePixels);
        denom = double(std::max(1, opaquePixels * 4));
        logf("Resumed from checkpoint: %s (%d/%d shapes)", resumePath.c_str(), accepted, cfg.stopAt);
    }

    float move, radius;
    mutation_steps(W, H, move, radius);
    logf("Loaded image: %s (%dx%d), transparency=%s", opts.imagePath.c_str(), W, H,
         prepared.hasTransparency ? "true" : "false");
    logf("Settings: stopAt=%d randomSamples=%d mutatedSamples=%d saveAt=%d saveEvery(preview)=%d",
         cfg.stopAt, cfg.randomSamples, cfg.mutatedSamples, (int)cfg.saveAt.size(), cfg.saveEvery);
    logf("Compatibility mode: forceOpaqueShapes=%s", cfg.forceOpaqueShapes ? "true" : "false");
    logf("Hill climb: %d starts x %d rounds x %d mutations (move +/- %.1fpx, radius +/- %.1fpx, "
         "theta +/- 30deg, shrinking to 1/5)",
         rc.starts, rc.rounds, rc.perRound, move, radius);
    logf("Pipeline: whole layer on the GPU (CUDA graph), row running-total scoring");
    logf("Scoring mode: DeltaE with GPU-computed optimal color (negative = better)");

    const bool profile = std::getenv("FPG_PROFILE") && std::string(std::getenv("FPG_PROFILE")) == "1";
    double profTotal = 0;
    int profSteps = 0;
    const double runStart = now_seconds();
    int noImprove = 0, finalPrunes = 0, lastPrunedMilestone = 0;
    uint64_t attempt = 0;

    while (accepted < cfg.stopAt) {
        const int step = accepted + 1;
        const double stepStart = now_seconds();
        LayerParams lp = layer_params(rc, seed, attempt++, W, H, accepted, cfg.stopAt,
                                      cfg.forceOpaqueShapes);
        LayerResult res = be->run_layer(lp);
        const double layerTime = now_seconds() - stepStart;

        logf("[%d/%d] Random best delta: %.6f", step, cfg.stopAt, res.randomBest);
        if (rc.rounds > 0 && res.randomBest < 0) {
            logf("[%d/%d] Hill climb best delta after %d rounds: %.6f (%d improvement(s))", step,
                 cfg.stopAt, rc.rounds, res.climbBest, res.improvements);
        }
        if (res.status != kApplied) {
            ++noImprove;
            if (res.status == kNoImprovement) {
                logf("[%d/%d] No improvement (delta %.6f). Retry %d/%d", step, cfg.stopAt,
                     res.climbBest, noImprove, kMaxNoImproveRetries);
            } else {
                logf("[%d/%d] Quantized candidate rejected after re-eval (delta %.6f). Retry %d/%d",
                     step, cfg.stopAt, res.finalScore, noImprove, kMaxNoImproveRetries);
            }
            if (noImprove >= kMaxNoImproveRetries) {
                logf("Stopped early: reached max retries without improvement (%d)", kMaxNoImproveRetries);
                break;
            }
            continue;
        }
        noImprove = 0;

        currentError += double(res.finalScore);
        if (currentError < 0) currentError = 0;
        shapes.push_back(to_shape(res.final, normalize_score(currentError, denom)));
        ++accepted;
        logf("[%d/%d] Added rotated ellipse #%d (delta %.6f)", accepted, cfg.stopAt,
             (int)shapes.size() - 1, res.finalScore);

        if (cfg.saveAt.count(accepted)) {
            std::string path = build_output_path(resolve_output_base(opts), accepted);
            if (!save_geometry(path, shapes)) throw std::runtime_error("write " + path);
            logf("[%d/%d] Saved geometry checkpoint for shape count %d", accepted, cfg.stopAt, accepted);
        }
        if (cfg.saveEvery >= 1 && accepted % cfg.saveEvery == 0 && !opts.previewPath.empty()) {
            save_preview(*be, opts, W, H, accepted);
            logf("[%d/%d] Saved preview snapshot", accepted, cfg.stopAt);
        }

        const bool milestone = accepted % 500 == 0 && accepted > lastPrunedMilestone;
        const bool finalPass = accepted == cfg.stopAt && finalPrunes < kMaxFinalPrunes;
        if (milestone || finalPass) {
            if (finalPass) {
                logf("[%d/%d] Reached target! Running final occlusion culling and compaction pass (%d/%d)...",
                     accepted, cfg.stopAt, finalPrunes + 1, kMaxFinalPrunes);
            } else {
                logf("[%d/%d] Scanning for completely occluded shapes to recycle...", accepted, cfg.stopAt);
            }
            if (milestone) lastPrunedMilestone = accepted;
            std::vector<ShapeOut> pruned = prune_occluded(shapes, W, H, prepared.mask.data());
            const int removed = (int)(shapes.size() - pruned.size());
            if (removed > 0) {
                logf("[%d/%d] Recycled %d occluded shapes%s! Active shapes: %d -> %d", accepted,
                     cfg.stopAt, removed, finalPass ? " in final pass" : "", (int)shapes.size(),
                     (int)pruned.size());
                if (finalPass) ++finalPrunes;
                shapes = pruned;
                accepted = (int)shapes.size() - 1;
                be->reset_canvas(prepared.current);
                std::vector<Candidate> redo;
                for (size_t i = 1; i < shapes.size(); ++i) redo.push_back(shape_to_candidate(shapes[i]));
                be->apply_shapes(redo);
                currentError = canvas_error(*be, prepared, opaquePixels);
            } else if (finalPass) {
                logf("[%d/%d] Final pass: No occluded shapes found. Ready to finish.", accepted, cfg.stopAt);
            } else {
                logf("[%d/%d] No occluded shapes found to recycle.", accepted, cfg.stopAt);
            }
        }

        if (profile) {
            profTotal += layerTime;
            if (++profSteps % 10 == 0) {
                logf("[profile @%d] GPU layer %.3f ms average over 10", accepted, profTotal * 100.0);
                profTotal = 0;
            }
        }
        logf("[%d/%d] Step completed in %s", accepted, cfg.stopAt,
             fmt_duration(now_seconds() - stepStart).c_str());
    }

    if (accepted < cfg.stopAt) {
        logf("Finished early with %d/%d shapes due to no-improvement stopping rule", accepted, cfg.stopAt);
    }
    const std::string finalPath = build_final_output_path(resolve_output_base(opts));
    if (!save_geometry(finalPath, shapes)) throw std::runtime_error("write " + finalPath);
    if (!opts.previewPath.empty()) {
        std::vector<Px> cur;
        be->read_current(cur);
        if (!save_png(opts.previewPath, cur, W, H))
            throw std::runtime_error("write preview " + opts.previewPath);
    }
    logf("Total time: %s for %d shapes", fmt_duration(now_seconds() - runStart).c_str(), accepted);
}

}  // namespace fpg
