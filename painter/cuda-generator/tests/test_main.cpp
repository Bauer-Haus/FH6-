// CPU tests for the CUDA generator. They need no GPU: they check the shared
// maths against values from the Go generator, the row-run method against
// pixel-by-pixel scoring, and a whole run on the CPU backend.
//
// Usage: fpg_tests <path to tests/data>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../src/backend.hpp"
#include "../src/engine.hpp"
#include "../src/geometry_json.hpp"
#include "../src/image_io.hpp"
#include "../src/platform.hpp"
#include "../src/settings.hpp"
#include "../src/tables.hpp"
#include "../src/tools.hpp"
#include "go_reference.inc"

using namespace fpg;

namespace {

int g_failures = 0;
int g_checks = 0;
std::string g_data;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::printf("    FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                             \
    } while (0)

#define CHECK_MSG(cond, ...)                                                          \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::printf("    FAILED %s:%d: %s: ", __FILE__, __LINE__, #cond);          \
            std::printf(__VA_ARGS__);                                                 \
            std::printf("\n");                                                        \
        }                                                                             \
    } while (0)

bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// ------------------------------------------------------------------ the tests

void test_go_values() {
    for (const auto& c : kSnapCases) CHECK_MSG(same_float(snap_rx(c[0]), c[1]), "snap(%.9g)", c[0]);
    for (const auto& c : kPenaltyCases) {
        Candidate k = {};
        k.rx = c.rx;
        k.ry = c.ry;
        CHECK_MSG(same_float(size_penalty(k, c.accepted), c.out), "penalty(%g,%g,%d)", c.rx, c.ry,
                  c.accepted);
    }
    for (const auto& c : kByteCases) CHECK_MSG(to_byte(c.in) == c.out, "byte(%.9g)", c.in);
    for (const auto& c : kAngleCases) CHECK_MSG(quant_angle(c.in) == c.out, "angle(%.9g)", c.in);
    for (const auto& c : kQuantXYCases) {
        Candidate k = {};
        k.x = c.in;
        k.y = c.in;
        k.rx = k.ry = 1;
        Candidate q = quantize(k, c.size, c.size, false);
        CHECK_MSG(same_float(q.x, c.out) && same_float(q.y, c.out), "quantize(%g)", c.in);
    }
    for (const auto& c : kStepCases) {
        float m, r;
        mutation_steps(c.w, c.h, m, r);
        CHECK(same_float(m, c.move) && same_float(r, c.radius));
    }
    for (const auto& c : kNormCases) CHECK(normalize_score(c.total, c.denom) == c.out);
    for (const auto& c : kJsonFloatCases) {
        std::string got = format_go_float(c.v);
        CHECK_MSG(got == c.text, "format(%.17g) = %s, Go: %s", c.v, got.c_str(), c.text);
    }
}

bool read_prepared_bin(const std::string& path, PreparedImage& p) {
    std::string raw;
    if (!read_file(path, raw) || raw.size() < 28) return false;
    int32_t hdr[7];
    std::memcpy(hdr, raw.data(), sizeof hdr);
    p.width = hdr[0];
    p.height = hdr[1];
    p.hasTransparency = hdr[2] != 0;
    for (int k = 0; k < 4; ++k) p.background[k] = (uint8_t)hdr[3 + k];
    const size_t n = (size_t)p.width * p.height;
    if (raw.size() != 28 + n * 16 + n) return false;
    p.target.resize(n);
    std::memcpy(p.target.data(), raw.data() + 28, n * 16);
    p.mask.assign(raw.begin() + 28 + n * 16, raw.end());
    return true;
}

void test_image_prepare() {
    for (int maxRes : {0, 40, 64}) {
        PreparedImage want, got;
        CHECK(read_prepared_bin(path_join(g_data, "alpha_test.prepared." + std::to_string(maxRes) + ".bin"), want));
        std::string err;
        CHECK_MSG(load_and_prepare(path_join(g_data, "alpha_test.png"), maxRes, got, err), "%s", err.c_str());
        CHECK_MSG(got.width == want.width && got.height == want.height, "size %dx%d vs Go %dx%d",
                  got.width, got.height, want.width, want.height);
        if (got.width != want.width || got.height != want.height) continue;
        CHECK(got.hasTransparency == want.hasTransparency);
        CHECK(std::memcmp(got.background, want.background, 4) == 0);
        CHECK(got.mask == want.mask);
        int diff = 0;
        float worst = 0;
        for (size_t i = 0; i < got.target.size(); ++i) {
            const float* a = &got.target[i].r;
            const float* b = &want.target[i].r;
            for (int k = 0; k < 4; ++k) {
                if (!same_float(a[k], b[k])) ++diff;
                worst = std::max(worst, std::fabs(a[k] - b[k]));
            }
        }
        CHECK_MSG(diff == 0, "maxRes %d: %d values differ from Go (worst %g)", maxRes, diff, worst);
    }
}

PreparedImage test_picture() { return synthetic_image(301, 203, 2000); }

std::vector<Candidate> random_shapes(int W, int H, int n, uint64_t seed) {
    std::vector<Candidate> out;
    for (int i = 0; i < n; ++i) {
        Rng r = make_rng(seed, 0, 3, (uint64_t)i);
        Candidate c;
        c.x = r.range(-20, (float)W + 20);
        c.y = r.range(-20, (float)H + 20);
        c.rx = snap_rx(r.range(0.3f, i % 3 ? 60.0f : 300.0f));
        c.ry = snap_rx(r.range(0.3f, i % 4 ? 60.0f : 300.0f));
        c.theta = r.range(0, 360);
        c.a = r.range(0.3f, 1.0f);
        c.r = r.f01();
        c.g = r.f01();
        c.b = r.f01();
        if (i % 9 == 0) c.theta = (float)(int)c.theta;  // whole degrees, as exported
        if (i % 13 == 0) c.theta = 90.0f * (i % 4);
        out.push_back(c);
    }
    return out;
}

void test_row_spans_match_pixels() {
    const int W = 211, H = 157;
    int rows = 0, bad = 0;
    for (const Candidate& c : random_shapes(W, H, 3000, 11)) {
        Geom g = make_geom(c, W, H);
        for (int y = 0; y < H; ++y) {
            const float dy = ((float)y + 0.5f) - g.cy;
            int lo = 0, hi = -1;
            bool any = row_span(g, y, lo, hi);
            for (int x = 0; x < W; ++x) {
                bool want = y >= g.yMin && y <= g.yMax && x >= g.xMin && x <= g.xMax && inside(g, x, dy);
                bool got = any && x >= lo && x <= hi;
                if (want != got) {
                    ++bad;
                    break;
                }
            }
            ++rows;
        }
    }
    CHECK_MSG(bad == 0, "%d of %d rows differ from the per-pixel test", bad, rows);
}

void test_scores_match_reference() {
    PreparedImage img = test_picture();
    RunConfig rc;
    rc.randomCount = 1;
    rc.starts = 1;
    rc.gridW = rc.gridH = 16;
    auto cpu = make_cpu_backend();
    cpu->init(img, rc);
    // paint some shapes first so the canvas is not flat
    std::vector<Candidate> paint = random_shapes(img.width, img.height, 40, 5);
    cpu->apply_shapes(paint);
    std::vector<Px> canvas;
    cpu->read_current(canvas);

    std::vector<Candidate> cands = random_shapes(img.width, img.height, 2000, 6);
    std::vector<float> scores;
    std::vector<Candidate> coloured;
    cpu->score_batch(cands, false, scores, coloured);
    int bad = 0, countBad = 0, rejected = 0;
    double worst = 0;
    for (size_t i = 0; i < cands.size(); ++i) {
        Candidate c = cands[i];
        Stats ref = reference_stats(img.target.data(), canvas.data(), img.mask.data(), img.width, img.height, c);
        float want = eval_stats(ref, c.a, false, c);
        if (is_rejected(want) || is_rejected(scores[i])) {
            if (is_rejected(want) != is_rejected(scores[i])) ++bad;
            ++rejected;
            continue;
        }
        double rel = std::fabs((double)scores[i] - want) / std::max(1e-3, std::fabs((double)want));
        worst = std::max(worst, rel);
        if (rel > 1e-5) ++bad;
        if (std::fabs(c.r - coloured[i].r) > 1e-5f) ++countBad;
    }
    CHECK_MSG(bad == 0, "%d of %zu scores differ (worst relative %.3g)", bad, cands.size(), worst);
    CHECK_MSG(countBad == 0, "%d colours differ", countBad);
    CHECK_MSG(rejected > 50 && rejected < 1900, "%d rejected: the test should cover both kinds", rejected);
}

void test_painting_matches_reference() {
    PreparedImage img = test_picture();
    RunConfig rc;
    rc.randomCount = 1;
    rc.starts = 1;
    rc.gridW = rc.gridH = 16;
    auto cpu = make_cpu_backend();
    cpu->init(img, rc);
    std::vector<Candidate> shapes = random_shapes(img.width, img.height, 300, 8);
    cpu->apply_shapes(shapes);
    std::vector<Px> got;
    cpu->read_current(got);
    std::vector<Px> want = img.current;
    for (const auto& c : shapes) reference_apply(want.data(), img.mask.data(), img.width, img.height, c);
    CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(Px)) == 0);

    std::vector<DynEntry> kept, fresh;
    cpu->read_tables(kept);
    fresh = build_dynamic_table(img.target.data(), got.data(), img.mask.data(), img.width, img.height);
    double worst = 0;
    for (size_t i = 0; i < kept.size(); ++i) {
        worst = std::max(worst, std::fabs(kept[i].s0 - fresh[i].s0));
        worst = std::max(worst, std::fabs(kept[i].ss - fresh[i].ss));
        worst = std::max(worst, std::fabs(kept[i].ts - fresh[i].ts));
    }
    CHECK_MSG(worst < 1e-9, "kept tables drift from a rebuild by %g", worst);
}

void test_layers_lower_the_error() {
    PreparedImage img = test_picture();
    Settings s;
    s.randomSamples = 300;
    s.mutatedSamples = 160;
    s.errorGridSize = 16;
    RunConfig rc = plan_run(s, img.width, img.height);
    CHECK(rc.starts == 8 && rc.rounds == 8 && rc.perRound == 2);
    auto cpu = make_cpu_backend();
    cpu->init(img, rc);
    int opaque = 0;
    double err = total_error(img.target.data(), img.current.data(), img.mask.data(), img.width, img.height, opaque);
    int accepted = 0, mismatched = 0;
    for (int i = 0; i < 25; ++i) {
        LayerParams lp = layer_params(rc, 77, (uint64_t)i, img.width, img.height, accepted, 100, false);
        LayerResult r = cpu->run_layer(lp);
        std::vector<Px> cur;
        cpu->read_current(cur);
        double now = total_error(img.target.data(), cur.data(), img.mask.data(), img.width, img.height, opaque);
        if (r.status == kApplied) {
            ++accepted;
            CHECK(now < err);
            // The score is the real change in error, for the colour painted.
            double rel = std::fabs((now - err) - r.finalScore) / std::max(1e-6, std::fabs((double)r.finalScore));
            if (rel > 1e-3) ++mismatched;
        } else {
            CHECK(now == err);
        }
        err = now;
    }
    CHECK_MSG(accepted >= 20, "only %d of 25 layers added a shape", accepted);
    CHECK_MSG(mismatched == 0, "%d layers' scores differ from the real error change", mismatched);
}

void test_same_seed_same_run() {
    PreparedImage img = test_picture();
    Settings s;
    s.randomSamples = 200;
    s.mutatedSamples = 64;
    RunConfig rc = plan_run(s, img.width, img.height);
    auto a = make_cpu_backend();
    auto b = make_cpu_backend();
    a->init(img, rc);
    b->init(img, rc);
    for (int i = 0; i < 5; ++i) {
        LayerParams lp = layer_params(rc, 5, (uint64_t)i, img.width, img.height, i, 50, false);
        LayerResult ra = a->run_layer(lp), rb = b->run_layer(lp);
        CHECK(std::memcmp(&ra.final, &rb.final, sizeof ra.final) == 0 && ra.status == rb.status);
    }
}

void test_settings() {
    Settings s = parse_settings_text(
        "\xEF\xBB\xBF" "description = A test\n"
        "; a comment\n# another\n"
        "maxResolution = 1200\nrandomSamples=50000\nmutatedSamples = 3000\n"
        "saveAt = 500, x, 1000,-3\nsaveEvery = 0\nstopAt = nope\nforceOpaqueShapes = yes\n"
        "useWorkGroupEval = false\nerrorMetric = SSIM\nclimbStarts = 12\nclimbRounds = 5\n"
        "progressiveSamplingCurve = -1\nmaxThreads = 4\n");
    CHECK(s.description == "A test");
    CHECK(s.maxResolution == 1200);
    CHECK(s.randomSamples == 50000);
    CHECK(s.mutatedSamples == 3000);
    CHECK((s.saveAt == std::set<int>{500, 1000}));
    CHECK(s.saveEvery == 1);
    CHECK(s.stopAt == 3000);
    CHECK(s.forceOpaqueShapes);
    CHECK(s.errorMetric == "ssim");
    CHECK(s.climbStarts == 12 && s.climbRounds == 5);
    CHECK(s.progressiveSamplingCurve == 2.0f);
    CHECK(s.ignoredKeys.size() == 2);
    Settings d = parse_settings_text("");
    CHECK(d.randomSamples == 3000 && d.stopAt == 3000 && d.saveAt.size() == 6);
    CHECK((parse_settings_text("saveAt = junk").saveAt == std::set<int>{3000}));
}

void test_json() {
    std::vector<ShapeOut> shapes(3);
    shapes[0].type = 1;
    shapes[0].data = {0, 0, 1200, 800};
    int c0[4] = {12, 34, 56, 0};
    std::memcpy(shapes[0].color, c0, sizeof c0);
    shapes[0].score = 0.123457;
    shapes[1].data = {600, 400, (double)snap_rx(150), (double)snap_rx(33.3f), 45};
    int c1[4] = {255, 0, 128, 200};
    std::memcpy(shapes[1].color, c1, sizeof c1);
    shapes[1].score = 0.000012;
    shapes[2].data = {0, 799, 0.63, 1.26, 360};
    int c2[4] = {1, 2, 3, 255};
    std::memcpy(shapes[2].color, c2, sizeof c2);
    shapes[2].score = 0;
    std::string want;
    CHECK(read_file(path_join(g_data, "sample_geometry.json"), want));
    std::string got = geometry_to_json(shapes);
    CHECK_MSG(got == want, "JSON text differs from Go's json.MarshalIndent:\n%s", got.c_str());

    std::vector<ShapeOut> back;
    std::string err;
    CHECK_MSG(parse_geometry(got, back, err), "%s", err.c_str());
    CHECK(back.size() == 3);
    if (back.size() == 3) {
        CHECK(back[1].data == shapes[1].data && back[1].color[3] == 200 && back[1].score == 0.000012);
        CHECK(back[0].type == 1 && back[2].colorCount == 4);
    }
    CHECK(!parse_geometry("{\"shapes\": [ {\"type\": 16, ", back, err));

    CHECK(build_output_path("C:/out/pic.png", 500) == "C:/out/pic.500.json");
    CHECK(build_final_output_path("/tmp/out/pic") == "/tmp/out/pic.json");
    CHECK(build_output_path("/a.b/c", 3) == "/a.b/c.3.json");
}

void test_prune() {
    std::vector<uint8_t> mask(100 * 100, 1);
    std::vector<ShapeOut> shapes(4);
    shapes[0].type = 1;
    shapes[0].data = {0, 0, 100, 100};
    auto ell = [](double x, double y, double r, int a) {
        ShapeOut s;
        s.data = {x, y, r, r, 0};
        s.color[3] = a;
        return s;
    };
    shapes[1] = ell(50, 50, 10, 255);  // hidden by shape 3
    shapes[2] = ell(10, 10, 5, 255);   // visible
    shapes[3] = ell(50, 50, 20, 255);
    std::vector<ShapeOut> out = prune_occluded(shapes, 100, 100, mask.data());
    CHECK(out.size() == 3);
    shapes[3].color[3] = 200;  // see-through: nothing is hidden
    CHECK(prune_occluded(shapes, 100, 100, mask.data()).size() == 4);
}

void test_whole_run_on_cpu(const std::string& tmp) {
    make_dirs(tmp);
    const std::string ini = path_join(tmp, "test.ini");
    write_file_atomic(ini,
                      "stopAt = 24\nrandomSamples = 400\nmutatedSamples = 128\nsaveAt = 10,24\n"
                      "saveEvery = 12\nmaxResolution = 64\nerrorGridSize = 8\n");
    Options o;
    o.imagePath = path_join(g_data, "alpha_test.png");
    o.settingsPath = ini;
    o.outputPath = path_join(tmp, "run.png");
    o.previewPath = path_join(tmp, "preview.png");
    o.seed = 3;
    o.useCpu = true;
    bool ok = true;
    try {
        run(o);
    } catch (const std::exception& e) {
        std::printf("    run: %s\n", e.what());
        ok = false;
    }
    CHECK(ok);
    std::vector<ShapeOut> g10, gFinal;
    std::string err;
    CHECK(load_geometry(path_join(tmp, "run.10.json"), g10, err));
    CHECK(load_geometry(path_join(tmp, "run.json"), gFinal, err));
    CHECK(g10.size() == 11);
    CHECK(gFinal.size() >= 2 && gFinal.size() <= 25);
    CHECK(is_dir(tmp) && read_file(path_join(tmp, "preview.12.png"), err) && read_file(path_join(tmp, "preview.png"), err));
    if (!gFinal.empty()) {
        CHECK(gFinal[0].type == 1 && gFinal[0].data[2] == 64 && gFinal[0].color[3] == 0);
        for (size_t i = 1; i < gFinal.size(); ++i) {
            const ShapeOut& s = gFinal[i];
            CHECK(s.type == 16 && s.data[0] == std::floor(s.data[0]) && s.data[4] == std::floor(s.data[4]));
            CHECK(snap_rx((float)s.data[2]) == (float)s.data[2]);
        }
    }

    // Resume from the 10-shape checkpoint to the same target.
    Options r = o;
    r.resumePath = path_join(tmp, "run.10.json");
    r.outputPath = path_join(tmp, "resumed.png");
    r.previewPath.clear();
    ok = true;
    try {
        run(r);
    } catch (const std::exception& e) {
        std::printf("    resume: %s\n", e.what());
        ok = false;
    }
    CHECK(ok);
    std::vector<ShapeOut> resumed;
    CHECK(load_geometry(path_join(tmp, "resumed.json"), resumed, err));
    CHECK(resumed.size() >= 11);
    if (resumed.size() >= 11 && g10.size() == 11)
        for (int i = 1; i <= 10; ++i) CHECK(resumed[i].data == g10[i].data);
}

}  // namespace

int main(int argc, char** argv) {
    g_data = argc > 1 ? argv[1] : "tests/data";
    const std::string tmp = argc > 2 ? argv[2] : path_join(dir_name(g_data), "tmp-test-output");
    struct T {
        const char* name;
        std::function<void()> fn;
    } tests[] = {
        {"values match the Go generator", test_go_values},
        {"image preparation matches Go", test_image_prepare},
        {"row runs match the per-pixel test", test_row_spans_match_pixels},
        {"row-total scores match pixel-by-pixel scores", test_scores_match_reference},
        {"painting by row runs matches pixel painting", test_painting_matches_reference},
        {"each layer lowers the error by its score", test_layers_lower_the_error},
        {"same seed, same layers", test_same_seed_same_run},
        {"settings file", test_settings},
        {"geometry JSON", test_json},
        {"occluded shapes are pruned", test_prune},
        {"whole run and resume on the CPU", [&] { test_whole_run_on_cpu(tmp); }},
    };
    for (auto& t : tests) {
        int before = g_failures;
        std::printf("%s\n", t.name);
        std::fflush(stdout);
        t.fn();
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
