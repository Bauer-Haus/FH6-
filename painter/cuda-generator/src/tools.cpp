#include "tools.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "backend.hpp"
#include "engine.hpp"
#include "geometry_json.hpp"
#include "tables.hpp"

namespace fpg {

PreparedImage synthetic_image(int width, int height, int maxResolution) {
    Image16 img;
    img.width = width;
    img.height = height;
    img.pix.resize((size_t)width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double u = double(x) / width, v = double(y) / height;
            double r = 0.5 + 0.5 * std::sin(6.0 * u + 1.0);
            double g = 0.5 + 0.5 * std::cos(5.0 * v + 3.0 * u);
            double b = v;
            // a few discs
            const double discs[4][5] = {{0.3, 0.4, 0.15, 0.9, 0.1},
                                        {0.6, 0.6, 0.10, 0.1, 0.8},
                                        {0.45, 0.25, 0.07, 1.0, 1.0},
                                        {0.2, 0.8, 0.12, 0.0, 0.0}};
            for (const auto& d : discs) {
                double dx = (u - d[0]) * width, dy = (v - d[1]) * height;
                if (dx * dx + dy * dy < (d[2] * width) * (d[2] * width)) {
                    r = d[3];
                    g = d[4];
                    b = 1.0 - d[3];
                }
            }
            // transparent top-right corner with a soft edge
            double cx = (1.0 - u) * width, cy = v * height;
            double dist = std::sqrt(cx * cx + cy * cy);
            double edge = 0.25 * width;
            double a = dist < edge ? 0.0 : (dist < edge + 6 ? (dist - edge) / 6.0 : 1.0);
            uint32_t a16 = (uint32_t)std::lround(a * 65535.0);
            uint32_t* p = &img.pix[((size_t)y * width + x) * 4];
            p[0] = (uint32_t)std::lround(r * 65535.0) * a16 / 0xffff;
            p[1] = (uint32_t)std::lround(g * 65535.0) * a16 / 0xffff;
            p[2] = (uint32_t)std::lround(b * 65535.0) * a16 / 0xffff;
            p[3] = a16;
        }
    }
    return prepare(img, maxResolution);
}

int run_score(const std::string& geometryPath, const std::string& imagePath) {
    std::vector<ShapeOut> shapes;
    std::string err;
    if (!load_geometry(geometryPath, shapes, err)) {
        std::printf("Error: %s\n", err.c_str());
        return 1;
    }
    if (shapes.empty() || shapes[0].type != 1 || shapes[0].data.size() < 4) {
        std::printf("Error: %s does not start with the background shape\n", geometryPath.c_str());
        return 1;
    }
    const int W = (int)shapes[0].data[2], H = (int)shapes[0].data[3];
    Image16 img;
    if (!load_image16(imagePath, img, err)) {
        std::printf("Error: %s\n", err.c_str());
        return 1;
    }
    PreparedImage p = prepare(img, std::max(W, H));
    if (p.width != W || p.height != H) {
        std::printf("Error: the geometry is for a %dx%d image, but %s prepares to %dx%d\n", W, H,
                    imagePath.c_str(), p.width, p.height);
        return 1;
    }
    std::vector<Px> canvas = p.current;
    int drawn = 0, skipped = 0;
    for (size_t i = 1; i < shapes.size(); ++i) {
        const ShapeOut& s = shapes[i];
        if (s.type != 16 || s.data.size() < 5 || s.colorCount < 4) {
            ++skipped;
            continue;
        }
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
        reference_apply(canvas.data(), p.mask.data(), W, H, c);
        ++drawn;
    }
    int opaque = 0;
    double e = total_error(p.target.data(), canvas.data(), p.mask.data(), W, H, opaque);
    double rmse = std::sqrt(e / std::max(1.0, double(opaque) * 4.0)) * 255.0;
    double psnr = rmse > 0 ? 20.0 * std::log10(255.0 / rmse) : 99.0;
    std::printf("Score: %s\n", geometryPath.c_str());
    std::printf("  shapes=%d skipped=%d size=%dx%d opaquePixels=%d\n", drawn, skipped, W, H, opaque);
    std::printf("  RMSE=%.4f (0-255 scale)  PSNR=%.3f dB\n", rmse, psnr);
    return 0;
}

#ifndef FPG_WITH_CUDA

int run_selftest(const std::string&) {
    std::printf("This build has no CUDA support, so there is no GPU to test.\n");
    return 2;
}

#else

namespace {

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

bool same_candidate(const Candidate& a, const Candidate& b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

double seconds_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<Candidate> test_candidates(int W, int H, int count) {
    std::vector<Candidate> out;
    LayerParams p = {};
    p.seed = 99;
    p.width = W;
    p.height = H;
    p.gridW = p.gridH = 1;
    p.minRadius = 0.5f;
    p.maxRadius = 0.4f * (float)std::max(W, H);
    double zero = 0.0;  // a CDF with total 0 makes centres uniform
    for (int i = 0; i < count; ++i) {
        Candidate c = random_candidate(p, &zero, (uint32_t)i);
        Rng rng = make_rng(7, 0, 9, (uint64_t)i);
        c.a = rng.range(0.3f, 1.0f);
        if (i % 7 == 0) c.ry = 0.63f;  // thin
        if (i % 11 == 0) c.rx = c.ry = 0.63f;  // single pixel
        out.push_back(c);
    }
    const float fw = (float)W, fh = (float)H;
    const Candidate edge[] = {
        {0, 0, 50, 20, 33, 0, 0, 0, 1},       {fw - 1, fh - 1, 80, 3, 120, 0, 0, 0, 0.5f},
        {fw / 2, fh / 2, 2000, 2000, 0, 0, 0, 0, 1}, {fw / 2, 0, 400, 0.63f, 90, 0, 0, 0, 0.7f},
        {10.5f, 10.5f, 0.63f, 0.63f, 45, 0, 0, 0, 1}, {fw / 3, fh / 3, 120, 60, 359.9f, 0, 0, 0, 0.3f},
    };
    for (const auto& c : edge) out.push_back(c);
    return out;
}

struct Check {
    int failures = 0;
    void report(bool ok, const char* name, const std::string& detail) {
        std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail.empty() ? "" : ": ",
                    detail.c_str());
        std::fflush(stdout);
        if (!ok) ++failures;
    }
};

void timing_run(Backend& gpu, const PreparedImage& img, const RunConfig& rc, double progress,
                int layers, std::vector<double>& times) {
    const int stopAt = 3000;
    int accepted = (int)(progress * stopAt);
    for (int i = 0; i < layers; ++i) {
        LayerParams lp = layer_params(rc, 4242, (uint64_t)(progress * 1e6) + i, img.width,
                                      img.height, accepted, stopAt, false);
        double t0 = seconds_now();
        LayerResult r = gpu.run_layer(lp);
        times.push_back(seconds_now() - t0);
        if (r.status == kApplied) ++accepted;
    }
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int run_selftest(const std::string& imagePath) {
    Check check;
    try {
        std::unique_ptr<Backend> gpu = make_cuda_backend();
        std::printf("GPU: %s\n", gpu->describe().c_str());

        PreparedImage img;
        if (imagePath.empty()) {
            img = synthetic_image(480, 360, 2000);
        } else {
            std::string err;
            if (!load_and_prepare(imagePath, 480, img, err)) throw std::runtime_error(err);
        }
        const int W = img.width, H = img.height;
        std::printf("Test image: %dx%d%s\n", W, H, imagePath.empty() ? " (synthetic)" : "");

        Settings s;
        s.randomSamples = 20000;
        s.mutatedSamples = 2000;
        RunConfig rc = plan_run(s, W, H);
        std::unique_ptr<Backend> cpu = make_cpu_backend();
        cpu->init(img, rc);
        gpu->init(img, rc);

        // 1. The starting tables and error map.
        {
            std::vector<DynEntry> a, b;
            std::vector<double> ca, cb;
            cpu->read_tables(a);
            gpu->read_tables(b);
            cpu->read_cdf(ca);
            gpu->read_cdf(cb);
            bool ok = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(DynEntry)) == 0 &&
                      ca == cb;
            check.report(ok, "running-total tables and error map after upload", "");
        }

        // 2. Scoring: GPU against the CPU (same bits) and the per-pixel reference.
        {
            std::vector<Candidate> cands = test_candidates(W, H, 4000);
            std::vector<float> sg, sc;
            std::vector<Candidate> cg, cc;
            gpu->score_batch(cands, false, sg, cg);
            cpu->score_batch(cands, false, sc, cc);
            int sameBits = 0, refBad = 0;
            double worst = 0;
            for (size_t i = 0; i < cands.size(); ++i) {
                if (same_bits(sg[i], sc[i]) && same_candidate(cg[i], cc[i])) ++sameBits;
                Candidate rc2 = cands[i];
                Stats st = reference_stats(img.target.data(), img.current.data(), img.mask.data(), W, H, rc2);
                float ref = eval_stats(st, rc2.a, false, rc2);
                bool rejG = is_rejected(sg[i]), rejR = is_rejected(ref);
                double diff = (rejG || rejR) ? (rejG == rejR ? 0.0 : 1e30)
                                             : std::fabs(double(sg[i]) - ref) / std::max(1e-3, std::fabs(double(ref)));
                worst = std::max(worst, diff);
                if (diff > 1e-4) ++refBad;
            }
            char buf[200];
            std::snprintf(buf, sizeof buf, "%d/%zu identical to the CPU", sameBits, cands.size());
            check.report(sameBits == (int)cands.size(), "GPU scores match the CPU bit for bit", buf);
            std::snprintf(buf, sizeof buf, "worst relative difference %.2e, %d over 1e-4", worst, refBad);
            check.report(refBad == 0, "GPU scores match pixel-by-pixel scoring", buf);
        }

        // 3. Whole layers: random batch, climb, rounding and painting.
        {
            const int layers = 40;
            int accepted = 0, same = 0, applied = 0;
            std::string detail;
            bool ok = true;
            double cpuTime = 0, gpuTime = 0;
            for (int i = 0; i < layers; ++i) {
                LayerParams lp = layer_params(rc, 12345, (uint64_t)i, W, H, accepted, 300, false);
                double t0 = seconds_now();
                LayerResult a = cpu->run_layer(lp);
                double t1 = seconds_now();
                LayerResult b = gpu->run_layer(lp);
                double t2 = seconds_now();
                cpuTime += t1 - t0;
                gpuTime += t2 - t1;
                bool match = a.status == b.status && same_bits(a.randomBest, b.randomBest) &&
                             same_bits(a.climbBest, b.climbBest) && a.improvements == b.improvements;
                if (match && a.status != kNoImprovement)
                    match = same_bits(a.finalScore, b.finalScore) && same_candidate(a.final, b.final);
                if (!match) {
                    double rel = std::fabs(double(a.climbBest) - b.climbBest) /
                                 std::max(1e-9, std::fabs(double(a.climbBest)));
                    char buf[300];
                    std::snprintf(buf, sizeof buf,
                                  "layer %d differs (CPU status %d delta %.9g, GPU status %d delta %.9g)", i,
                                  a.status, a.climbBest, b.status, b.climbBest);
                    detail = buf;
                    if (rel < 1e-4) {
                        detail += "; a near-tie from rounding, so later layers are not compared";
                    } else {
                        ok = false;
                    }
                    break;
                }
                ++same;
                if (a.status == kApplied) {
                    ++accepted;
                    ++applied;
                }
            }
            char buf[200];
            std::snprintf(buf, sizeof buf, "%d/%d layers identical, %d shapes added; CPU %.2fs, GPU %.3fs",
                          same, layers, applied, cpuTime, gpuTime);
            check.report(ok, "GPU layers match the CPU", detail.empty() ? buf : std::string(buf) + "; " + detail);

            if (same == layers) {
                std::vector<Px> ca, cb;
                cpu->read_current(ca);
                gpu->read_current(cb);
                std::vector<DynEntry> ta, tb;
                cpu->read_tables(ta);
                gpu->read_tables(tb);
                std::vector<double> da, db;
                cpu->read_cdf(da);
                gpu->read_cdf(db);
                bool same2 = std::memcmp(ca.data(), cb.data(), ca.size() * sizeof(Px)) == 0 &&
                             std::memcmp(ta.data(), tb.data(), ta.size() * sizeof(DynEntry)) == 0 &&
                             da == db;
                check.report(same2, "canvas, tables and error map after painting", "");

                // The tables the GPU keeps up to date match tables rebuilt from scratch.
                std::vector<DynEntry> fresh =
                    build_dynamic_table(img.target.data(), cb.data(), img.mask.data(), W, H);
                double worst = 0;
                for (size_t k = 0; k < fresh.size(); ++k)
                    worst = std::max(worst, std::fabs(fresh[k].ss - tb[k].ss));
                check.report(worst < 1e-6, "updated tables match a full rebuild",
                             "largest difference " + std::to_string(worst));
            }
        }

        // 4. Speed, with the "c. balanced" sample counts on a 1200x800 image.
        {
            PreparedImage big = synthetic_image(1200, 800, 2000);
            Settings bs;
            bs.randomSamples = 50000;
            bs.mutatedSamples = 3000;
            RunConfig brc = plan_run(bs, big.width, big.height);
            std::unique_ptr<Backend> g2 = make_cuda_backend();
            g2->init(big, brc);
            std::vector<double> warm;
            timing_run(*g2, big, brc, 0.0, 1, warm);  // records the CUDA graph
            std::vector<double> t0, t1, t2;
            timing_run(*g2, big, brc, 0.0, 30, t0);
            timing_run(*g2, big, brc, 0.5, 30, t1);
            timing_run(*g2, big, brc, 0.95, 30, t2);
            double m0 = median(t0) * 1000, m1 = median(t1) * 1000, m2 = median(t2) * 1000;
            std::printf("[INFO] speed, 1200x800, 50000 random + 3000 climb tweaks per layer:\n");
            std::printf("       first layer %.1f ms (includes setup)\n", warm[0] * 1000);
            std::printf("       per layer: start %.2f ms, middle %.2f ms, end %.2f ms\n", m0, m1, m2);
            std::printf("       about %.0f s for 3000 layers\n", (m0 + m1 + m2) / 3.0 * 3000 / 1000);
        }
    } catch (const std::exception& e) {
        std::printf("[FAIL] %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", check.failures == 0 ? "SELFTEST PASSED" : "SELFTEST FAILED");
    return check.failures == 0 ? 0 : 1;
}

#endif

}  // namespace fpg
