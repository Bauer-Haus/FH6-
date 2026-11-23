// Shared maths for the CPU and the GPU.
//
// Every function here compiles for both (FPG_HD). The CPU backend, the tests
// and the CUDA kernels all call the same code, and the build turns off
// multiply-add fusing on both sides (-ffp-contract=off / --fmad=false), so a
// CPU run and a GPU run with the same seed make the same decisions.
//
// Ported from painter/generator (Go): engine.go and gpu/kernels.go. Each
// function names the Go function or kernel it matches.
#pragma once

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define FPG_HD __host__ __device__ inline
#else
#define FPG_HD inline
#endif

namespace fpg {

// evaluate_candidates_v3 writes this score for shapes it rejects.
constexpr float kRejectScore = 3.402823466e+38f;
// engine.go: minImproveDelta. A layer must lower the error by more than this.
constexpr float kMinImproveDelta = -1e-7f;
constexpr int kWarp = 32;
constexpr int kMaxClimbRounds = 64;

// One rotated ellipse. Colour channels are 0..1. theta is in degrees.
struct Candidate {
    float x, y, rx, ry, theta, r, g, b, a;
};

struct alignas(16) Px {
    float r, g, b, a;
};

// Row running totals. Entry x of a row holds the sum over pixels 0..x-1 of
// that row, counting opaque pixels only (transparent pixels count in nt).
// DynEntry changes as shapes are painted; StatEntry is fixed for the run.
struct DynEntry {
    double s0, s1, s2, s3;  // sum of current, per channel
    double ss;              // sum over the 4 channels of current^2
    double ts;              // sum over the 4 channels of target*current
};

struct StatEntry {
    double t0, t1, t2, t3;  // sum of target, per channel
    int n;                  // opaque pixels
    int nt;                 // transparent pixels
};

// Statistics of the pixels inside one shape (what evaluate_candidates_v3
// gathers pixel by pixel). The per-channel sums of current^2 and
// target*current only appear summed over channels in the score formula,
// so they are kept as one number each.
struct Stats {
    double s0, s1, s2, s3, ss, ts, t0, t1, t2, t3;
    int n, nt;

    FPG_HD void clear() {
        s0 = s1 = s2 = s3 = ss = ts = t0 = t1 = t2 = t3 = 0.0;
        n = nt = 0;
    }
    FPG_HD void add(const Stats& o) {
        s0 += o.s0; s1 += o.s1; s2 += o.s2; s3 += o.s3;
        ss += o.ss; ts += o.ts;
        t0 += o.t0; t1 += o.t1; t2 += o.t2; t3 += o.t3;
        n += o.n; nt += o.nt;
    }
};

// Everything one layer needs to know. Written by the host before each layer.
struct LayerParams {
    uint64_t seed;
    uint64_t attempt;      // counts every layer attempt, including retries
    int width, height;
    int acceptedShapes;
    int forceOpaque;
    int randomCount;       // random candidates per layer
    int starts;            // hill-climb starting shapes (best K random)
    int rounds;            // hill-climb rounds
    int perRound;          // tweaks per start per round
    int gridW, gridH;
    float minRadius, maxRadius;
};

// Tweak sizes for each hill-climb round. Fixed for a run.
struct ClimbSteps {
    float move[kMaxClimbRounds];
    float radius[kMaxClimbRounds];
    float theta[kMaxClimbRounds];
};

enum LayerStatus : int {
    kApplied = 0,
    kNoImprovement = 1,     // the best shape did not lower the error
    kQuantizedRejected = 2, // rounding to whole pixels made it spill
    kQuantizedNoGain = 3,   // rounding to whole pixels lost the gain
};

struct LayerResult {
    Candidate final;   // the shape as exported (rounded); valid when applied
    float randomBest;  // best raw score of the random batch
    float climbBest;   // best raw score after the climb
    float finalScore;  // raw score of the rounded shape
    int status;
    int improvements;  // strict improvements made by the climb
    int pad[2];
};

// The shape the apply kernel paints, if enabled.
struct ApplyJob {
    Candidate c;
    int enabled;
    int pad[3];
};

// ---------------------------------------------------------------- helpers

FPG_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
FPG_HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
FPG_HD double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
FPG_HD int maxi(int a, int b) { return a > b ? a : b; }

// Go math.Round: halves round away from zero, as C round() does.
FPG_HD double go_round(double v) { return round(v); }

// engine.go: snapToValidRX. Radii are whole multiples of 0.63 so that
// radius/63 has at most two decimals in the game's units.
FPG_HD float snap_rx(float v) {
    float k = (float)go_round((double)v * (100.0 / 63.0));
    if (k < 1.0f) k = 1.0f;
    return k * (float)(63.0 / 100.0);
}

// engine.go: f32ToByte.
FPG_HD uint8_t to_byte(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (uint8_t)go_round((double)(v * 255.0f));
}

// engine.go: the angle rounding in quantizeCandidate and toShape.
FPG_HD int quant_angle(float theta) {
    int angle = (int)go_round((double)theta) % 360;
    if (angle < 0) angle += 360;
    if (angle == 0 && theta > 359.5f) angle = 360;
    return angle;
}

// engine.go: scaleCompatibilityPenalty.
FPG_HD float size_penalty(const Candidate& c, int acceptedShapes) {
    if (c.rx < 0.0f || c.ry < 0.0f) return 0.0f;
    double rx = (double)maxi(1, (int)go_round((double)c.rx));
    double ry = (double)maxi(1, (int)go_round((double)c.ry));
    double span = rx > ry ? rx : ry;
    if (acceptedShapes < 8 && span >= 96.0) return (float)(span * span * 0.1);
    if (span >= 96.0) return (float)(span * 0.05);
    return 0.0f;
}

// engine.go: isRejectedEvalScore.
FPG_HD bool is_rejected(float score) {
    if (score != score) return true;  // NaN
    if (isinf(score)) return true;
    return score >= kRejectScore * 0.5f;
}

// engine.go: quantizeCandidate (ellipses only).
FPG_HD Candidate quantize(Candidate c, int width, int height, bool forceOpaque) {
    c.x = (float)clampi((int)go_round((double)c.x), 0, maxi(0, width - 1));
    c.y = (float)clampi((int)go_round((double)c.y), 0, maxi(0, height - 1));
    c.rx = snap_rx(c.rx);
    c.ry = snap_rx(c.ry);
    c.theta = (float)quant_angle(c.theta);
    if (forceOpaque) c.a = 1.0f;
    c.r = (float)to_byte(c.r) / 255.0f;
    c.g = (float)to_byte(c.g) / 255.0f;
    c.b = (float)to_byte(c.b) / 255.0f;
    c.a = (float)to_byte(c.a) / 255.0f;
    return c;
}

// --------------------------------------------------------------- geometry

// A candidate set up for pixel tests, with the same bounding box and the
// same per-pixel test as evaluate_candidates_v3.
struct Geom {
    float cx, cy, cosT, sinT, invRX2, invRY2;
    float qa, qb, qc;  // row equation: qa*dx^2 + (qb*dy)*dx + (qc*dy^2 - 1) <= 0
    int xMin, xMax, yMin, yMax;
};

FPG_HD Geom make_geom(const Candidate& c, int width, int height) {
    Geom g;
    float rx = c.rx > 1.0f ? c.rx : 1.0f;
    float ry = c.ry > 1.0f ? c.ry : 1.0f;
    float theta = c.theta * 0.01745329251994329577f;
    g.cx = c.x;
    g.cy = c.y;
    g.cosT = (float)cos((double)theta);
    g.sinT = (float)sin((double)theta);
    g.invRX2 = 1.0f / (rx * rx);
    g.invRY2 = 1.0f / (ry * ry);
    float rx2 = rx * rx, ry2 = ry * ry;
    float cos2 = g.cosT * g.cosT, sin2 = g.sinT * g.sinT;
    float ex = sqrtf(rx2 * cos2 + ry2 * sin2);
    float ey = sqrtf(rx2 * sin2 + ry2 * cos2);
    g.xMin = maxi(0, (int)floorf(c.x - ex - 1.0f));
    g.yMin = maxi(0, (int)floorf(c.y - ey - 1.0f));
    int xMax = (int)ceilf(c.x + ex + 1.0f);
    int yMax = (int)ceilf(c.y + ey + 1.0f);
    g.xMax = xMax < width - 1 ? xMax : width - 1;
    g.yMax = yMax < height - 1 ? yMax : height - 1;
    g.qa = cos2 * g.invRX2 + sin2 * g.invRY2;
    g.qb = 2.0f * g.cosT * g.sinT * (g.invRX2 - g.invRY2);
    g.qc = sin2 * g.invRX2 + cos2 * g.invRY2;
    return g;
}

// The per-pixel test from evaluate_candidates_v3 / apply_candidate_v2.
// dy is ((float)y + 0.5f) - cy.
FPG_HD bool inside(const Geom& g, int x, float dy) {
    float dx = ((float)x + 0.5f) - g.cx;
    float xr = dx * g.cosT + dy * g.sinT;
    float yr = -dx * g.sinT + dy * g.cosT;
    return xr * xr * g.invRX2 + yr * yr * g.invRY2 <= 1.0f;
}

// The run of pixels [lo, hi] that row y has inside the shape, within the
// bounding box. The ends come from solving the ellipse equation for the row;
// then the end pixels are checked with inside(), so the run holds exactly
// the pixels the per-pixel test accepts.
FPG_HD bool row_span(const Geom& g, int y, int& lo, int& hi) {
    if (y < g.yMin || y > g.yMax || g.xMin > g.xMax) return false;
    float dy = ((float)y + 0.5f) - g.cy;
    float qb = g.qb * dy;
    float qc = g.qc * dy * dy - 1.0f;
    float disc = qb * qb - 4.0f * g.qa * qc;
    float mid = g.cx - 0.5f - qb / (2.0f * g.qa);
    int a, b;
    if (disc >= 0.0f) {
        float h = sqrtf(disc) / (2.0f * g.qa);
        a = (int)ceilf(mid - h);
        b = (int)floorf(mid + h);
    } else {
        a = 1;
        b = 0;
    }
    if (a < g.xMin) a = g.xMin;
    if (b > g.xMax) b = g.xMax;
    if (a > b) {
        // Rows that just touch the shape: try the pixels nearest its middle.
        int m = clampi((int)floorf(mid + 0.5f), g.xMin, g.xMax);
        if (inside(g, m, dy)) {
            a = b = m;
        } else if (m > g.xMin && inside(g, m - 1, dy)) {
            a = b = m - 1;
        } else if (m < g.xMax && inside(g, m + 1, dy)) {
            a = b = m + 1;
        } else {
            return false;
        }
    }
    while (a <= b && !inside(g, a, dy)) ++a;
    while (b >= a && !inside(g, b, dy)) --b;
    if (a > b) return false;
    while (a > g.xMin && inside(g, a - 1, dy)) --a;
    while (b < g.xMax && inside(g, b + 1, dy)) ++b;
    lo = a;
    hi = b;
    return true;
}

// Adds the rows this warp lane owns (yMin+lane, yMin+lane+32, ...).
FPG_HD void accumulate_lane(const DynEntry* dyn, const StatEntry* st, int width,
                            const Geom& g, int lane, Stats& acc) {
    const long long stride = (long long)width + 1;
    for (int y = g.yMin + lane; y <= g.yMax; y += kWarp) {
        int lo, hi;
        if (!row_span(g, y, lo, hi)) continue;
        const long long r = (long long)y * stride;
        const DynEntry& d0 = dyn[r + lo];
        const DynEntry& d1 = dyn[r + hi + 1];
        const StatEntry& t0 = st[r + lo];
        const StatEntry& t1 = st[r + hi + 1];
        acc.s0 += d1.s0 - d0.s0;
        acc.s1 += d1.s1 - d0.s1;
        acc.s2 += d1.s2 - d0.s2;
        acc.s3 += d1.s3 - d0.s3;
        acc.ss += d1.ss - d0.ss;
        acc.ts += d1.ts - d0.ts;
        acc.t0 += t1.t0 - t0.t0;
        acc.t1 += t1.t1 - t0.t1;
        acc.t2 += t1.t2 - t0.t2;
        acc.t3 += t1.t3 - t0.t3;
        acc.n += t1.n - t0.n;
        acc.nt += t1.nt - t0.nt;
    }
}

// The score of drawing the shape with alpha a over the current canvas,
// using the identity from evaluate_candidates_v3:
//   sum dErr = a^2 [N c^2 - 2c Ss + Sss] - 2a [c St - c Ss - Sst + Sss]
// for each channel, with the optimal colour for c (alpha channel: c = 1).
// Negative is better. Writes the colour to out.r/g/b. With roundColor the
// colour is rounded to 8 bits first and the score is for that colour, so the
// score matches exactly what gets painted and exported.
FPG_HD float eval_stats(const Stats& s, float alpha, bool roundColor, Candidate& out) {
    // Hard reject (evaluate_candidates_v3): no opaque pixel, or too much spill
    // into transparent pixels, which the game would draw over the car.
    if (s.n == 0 || s.nt > 64 || s.nt * 400 > s.n) {
        out.r = out.g = out.b = 0.0f;
        return kRejectScore;
    }
    const float caf = clampf(alpha, 1e-3f, 1.0f);
    const double a = (double)caf;
    const double n = (double)s.n;
    const double invA = 1.0 - a;
    float fr = (float)clampd((s.t0 / n - (s.s0 / n) * invA) / a, 0.0, 1.0);
    float fg = (float)clampd((s.t1 / n - (s.s1 / n) * invA) / a, 0.0, 1.0);
    float fb = (float)clampd((s.t2 / n - (s.s2 / n) * invA) / a, 0.0, 1.0);
    if (roundColor) {
        fr = (float)to_byte(fr) / 255.0f;
        fg = (float)to_byte(fg) / 255.0f;
        fb = (float)to_byte(fb) / 255.0f;
    }
    out.r = fr;
    out.g = fg;
    out.b = fb;
    const double r = fr, g = fg, b = fb;
    const double a2 = a * a, twoA = 2.0 * a;
    double d = 0.0;
    d += a2 * (n * r * r - 2.0 * r * s.s0) - twoA * (r * s.t0 - r * s.s0);
    d += a2 * (n * g * g - 2.0 * g * s.s1) - twoA * (g * s.t1 - g * s.s1);
    d += a2 * (n * b * b - 2.0 * b * s.s2) - twoA * (b * s.t2 - b * s.s2);
    d += a2 * (n - 2.0 * s.s3) - twoA * (s.t3 - s.s3);
    d += (a2 - twoA) * s.ss + twoA * s.ts;
    if (s.nt > 0) {
        const double nt = (double)s.nt;
        const double spill = nt / (n + nt);
        d += a2 * nt * (1.0 + 2.0 * spill) * (r * r + g * g + b * b + 1.0);
    }
    return (float)d;
}

// apply_candidate_v2 for one pixel.
FPG_HD void blend(Px& p, const Candidate& c) {
    const float alpha = clampf(c.a, 0.0f, 1.0f);
    const float inv = 1.0f - alpha;
    p.r = p.r * inv + clampf(c.r, 0.0f, 1.0f) * alpha;
    p.g = p.g * inv + clampf(c.g, 0.0f, 1.0f) * alpha;
    p.b = p.b * inv + clampf(c.b, 0.0f, 1.0f) * alpha;
    p.a = p.a * inv + alpha;
}

// ---------------------------------------------------- rebuilding row totals
//
// After a shape is painted, a row's running totals are rewritten from the
// first changed pixel x0 on. On the GPU one warp does a row: each lane sums a
// chunk of the row, the chunk offsets are added up in lane order, then each
// lane writes its chunk. The CPU does the same steps in the same order, so
// both get the same numbers to the last bit.

FPG_HD void entry_add(DynEntry& acc, const DynEntry& v) {
    acc.s0 += v.s0;
    acc.s1 += v.s1;
    acc.s2 += v.s2;
    acc.s3 += v.s3;
    acc.ss += v.ss;
    acc.ts += v.ts;
}

FPG_HD DynEntry pixel_entry(const Px* target, const Px* current, const uint8_t* mask, long long p) {
    DynEntry v = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (!mask[p]) return v;
    const Px s = current[p];
    const Px t = target[p];
    const double sr = s.r, sg = s.g, sb = s.b, sa = s.a;
    v.s0 = sr;
    v.s1 = sg;
    v.s2 = sb;
    v.s3 = sa;
    v.ss = ((sr * sr + sg * sg) + sb * sb) + sa * sa;
    v.ts = (((double)t.r * sr + (double)t.g * sg) + (double)t.b * sb) + (double)t.a * sa;
    return v;
}

// The pixels [a, b) of row-tail x0..width-1 that lane `lane` handles.
FPG_HD void chunk_bounds(int width, int x0, int lane, int& a, int& b) {
    const int n = width - x0;
    const int chunk = (n + kWarp - 1) / kWarp;
    a = x0 + lane * chunk;
    b = a + chunk;
    if (a > width) a = width;
    if (b > width) b = width;
}

// Step 1: the sum of a lane's chunk, starting from zero.
FPG_HD DynEntry chunk_total(const Px* target, const Px* current, const uint8_t* mask, int width,
                            int y, int x0, int lane) {
    int a, b;
    chunk_bounds(width, x0, lane, a, b);
    const long long p0 = (long long)y * width;
    DynEntry acc = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int x = a; x < b; ++x) entry_add(acc, pixel_entry(target, current, mask, p0 + x));
    return acc;
}

// Step 3: write a lane's chunk, starting from `offset` (the running total
// at the chunk's first pixel).
FPG_HD void chunk_write(DynEntry* dyn, const Px* target, const Px* current, const uint8_t* mask,
                        int width, int y, int x0, int lane, DynEntry offset) {
    int a, b;
    chunk_bounds(width, x0, lane, a, b);
    const long long r = (long long)y * ((long long)width + 1);
    const long long p0 = (long long)y * width;
    for (int x = a; x < b; ++x) {
        entry_add(offset, pixel_entry(target, current, mask, p0 + x));
        dyn[r + x + 1] = offset;
    }
}

// The CPU version of the warp rebuild (step 2 adds the chunk totals in lane
// order, as the GPU does).
inline void rebuild_row(DynEntry* dyn, const Px* target, const Px* current, const uint8_t* mask,
                        int width, int y, int x0) {
    DynEntry totals[kWarp];
    for (int l = 0; l < kWarp; ++l) totals[l] = chunk_total(target, current, mask, width, y, x0, l);
    DynEntry offset = dyn[(long long)y * ((long long)width + 1) + x0];
    for (int l = 0; l < kWarp; ++l) {
        chunk_write(dyn, target, current, mask, width, y, x0, l, offset);
        entry_add(offset, totals[l]);
    }
}

// The same chunked running total for the error-map CDF.
inline void build_cdf_cpu(const float* grid, double* cdf, int n) {
    const int chunk = (n + kWarp - 1) / kWarp;
    double totals[kWarp];
    for (int l = 0; l < kWarp; ++l) {
        double t = 0.0;
        for (int i = l * chunk; i < n && i < (l + 1) * chunk; ++i) t += grid[i] > 0.0f ? (double)grid[i] : 0.0;
        totals[l] = t;
    }
    double offset = 0.0;
    for (int l = 0; l < kWarp; ++l) {
        double run = offset;
        for (int i = l * chunk; i < n && i < (l + 1) * chunk; ++i) {
            run += grid[i] > 0.0f ? (double)grid[i] : 0.0;
            cdf[i] = run;
        }
        offset += totals[l];
    }
}

// compute_error_grid for one cell.
FPG_HD float grid_cell(const Px* target, const Px* current, const uint8_t* mask,
                       int width, int height, int gridW, int gridH, int gx, int gy) {
    int x0 = (int)(((long long)gx * width) / gridW);
    int x1 = (int)(((long long)(gx + 1) * width) / gridW);
    int y0 = (int)(((long long)gy * height) / gridH);
    int y1 = (int)(((long long)(gy + 1) * height) / gridH);
    float sum = 0.0f;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            long long p = (long long)y * width + x;
            if (!mask[p]) continue;
            Px t = target[p], s = current[p];
            float dr = t.r - s.r, dg = t.g - s.g, db = t.b - s.b, da = t.a - s.a;
            sum += dr * dr + dg * dg + db * db + da * da;
        }
    }
    return sum;
}

FPG_HD bool cell_overlaps(int width, int height, int gridW, int gridH, int gx, int gy,
                          const Geom& g) {
    int x0 = (int)(((long long)gx * width) / gridW);
    int x1 = (int)(((long long)(gx + 1) * width) / gridW);
    int y0 = (int)(((long long)gy * height) / gridH);
    int y1 = (int)(((long long)(gy + 1) * height) / gridH);
    return x0 <= g.xMax && x1 - 1 >= g.xMin && y0 <= g.yMax && y1 - 1 >= g.yMin;
}

// -------------------------------------------------------------- randomness

// Counter-based random numbers: every candidate has its own stream, keyed by
// (seed, attempt, stream, index), so a run repeats exactly for a given seed
// whatever the GPU, and the CPU backend makes the same draws.
FPG_HD uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Rng {
    uint64_t state;
    FPG_HD uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        return mix64(state);
    }
    FPG_HD float f01() { return (float)(next() >> 40) * (1.0f / 16777216.0f); }
    FPG_HD double d01() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    FPG_HD float range(float lo, float hi) { return lo + (hi - lo) * f01(); }
};

FPG_HD Rng make_rng(uint64_t seed, uint64_t attempt, uint32_t stream, uint64_t index) {
    uint64_t k = mix64(seed ^ 0x5DEECE66DULL);
    k = mix64(k ^ (attempt * 0xD1B54A32D192ED03ULL));
    k = mix64(k ^ ((uint64_t)stream * 0x8CB92BA72F3D8DD7ULL));
    Rng r;
    r.state = mix64(k ^ (index * 0x9E3779B97F4A7C15ULL));
    return r;
}

// errorSampler.sample: a point drawn in proportion to the error map.
// cdf holds the running total of the grid cells.
FPG_HD void sample_center(Rng& rng, const double* cdf, int gridW, int gridH, int width,
                          int height, float& x, float& y) {
    const int n = gridW * gridH;
    const double total = n > 0 ? cdf[n - 1] : 0.0;
    if (!(total > 0.0)) {
        x = rng.f01() * (float)width;
        y = rng.f01() * (float)height;
        return;
    }
    const double u = rng.d01() * total;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    const int gx = lo % gridW, gy = lo / gridW;
    int x0 = (int)(((long long)gx * width) / gridW);
    int x1 = (int)(((long long)(gx + 1) * width) / gridW);
    int y0 = (int)(((long long)gy * height) / gridH);
    int y1 = (int)(((long long)(gy + 1) * height) / gridH);
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;
    if (x1 > width) x1 = width;
    if (y1 > height) y1 = height;
    x = (float)x0 + rng.f01() * (float)(x1 - x0);
    y = (float)y0 + rng.f01() * (float)(y1 - y0);
}

// randomCandidatesInto, for candidate `index` of this attempt.
FPG_HD Candidate random_candidate(const LayerParams& p, const double* cdf, uint32_t index) {
    Rng rng = make_rng(p.seed, p.attempt, 1u, index);
    Candidate c;
    sample_center(rng, cdf, p.gridW, p.gridH, p.width, p.height, c.x, c.y);
    const float w = (float)p.width, h = (float)p.height;
    if (c.x < 0.0f) c.x = 0.0f;
    if (c.y < 0.0f) c.y = 0.0f;
    if (c.x > w - 1.0f) c.x = w - 1.0f;
    if (c.y > h - 1.0f) c.y = h - 1.0f;
    c.a = p.forceOpaque ? 1.0f : rng.range(0.3f, 1.0f);
    c.rx = snap_rx(rng.range(p.minRadius, p.maxRadius));
    c.ry = snap_rx(rng.range(p.minRadius, p.maxRadius));
    c.theta = rng.f01() * 360.0f;
    c.r = c.g = c.b = 0.0f;
    return c;
}

// mutatedCandidates, for one tweak, with this round's step sizes.
FPG_HD Candidate mutate(const Candidate& base, Rng& rng, int width, int height, bool forceOpaque,
                        float moveStep, float radiusStep, float thetaStep) {
    Candidate c = base;
    const float w = (float)width, h = (float)height;
    c.x += rng.range(-moveStep, moveStep);
    c.y += rng.range(-moveStep, moveStep);
    if (c.x < 0.0f) c.x = 0.0f;
    if (c.y < 0.0f) c.y = 0.0f;
    if (c.x > w - 1.0f) c.x = w - 1.0f;
    if (c.y > h - 1.0f) c.y = h - 1.0f;
    double nrx = (double)(c.rx + rng.range(-radiusStep, radiusStep));
    double nry = (double)(c.ry + rng.range(-radiusStep, radiusStep));
    c.rx = snap_rx((float)(nrx > 0.63 ? nrx : 0.63));
    c.ry = snap_rx((float)(nry > 0.63 ? nry : 0.63));
    c.theta += rng.range(-thetaStep, thetaStep);
    if (c.theta < 0.0f) c.theta += 360.0f;
    if (c.theta >= 360.0f) c.theta -= 360.0f;
    if (forceOpaque) c.a = 1.0f;
    return c;
}

// A sort key that orders by (score, index): smaller is better, ties keep
// the lower index (as pickBestIndex does).
FPG_HD uint64_t score_key(float score, uint32_t index) {
    if (score == 0.0f) score = 0.0f;  // -0 sorts as +0
    union {
        float f;
        uint32_t u;
    } v;
    v.f = score;
    uint32_t bits = (v.u & 0x80000000u) ? ~v.u : (v.u | 0x80000000u);
    return ((uint64_t)bits << 32) | (uint64_t)index;
}

FPG_HD uint32_t key_index(uint64_t key) { return (uint32_t)(key & 0xFFFFFFFFull); }

// (score, index) is better than (score2, index2)?
FPG_HD bool better(float s, int i, float s2, int i2) { return s < s2 || (s == s2 && i < i2); }

}  // namespace fpg
