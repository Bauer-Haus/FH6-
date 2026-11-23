/*
 * fh6_fastcore.c - optional compiled core for fh6_sticker.py.
 *
 * fh6_sticker.py builds this file on first use with the system C compiler and
 * loads it through ctypes. If no compiler is found it uses its numpy code,
 * which gives the same kind of result, just slower.
 *
 * Shapes are 6 doubles: cx, cy, w, h, rot (degrees), is_circle. The pixel rule
 * and formulas are the same as shape_spans() in fh6_sticker.py: a pixel is
 * inside a shape when its centre is, and each shape covers one run of pixels
 * per row. ct/cq/ce are the per-row running totals kept by _Scorer:
 *   ct  (H, W+1, 3)  target colour
 *   cq  (H, W+1)     target colour squared, summed over channels
 *   ce  (H, W+1)     current squared error
 */

#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int W, H;
    const double *ct, *cq, *ce;
} Canvas;

/* x range where -r <= k*x + c <= r; returns 0 when empty. */
static int line_range(double k, double c, double r, double *lo, double *hi) {
    if (fabs(k) > 1e-9) {
        double a = (-r - c) / k, b = (r - c) / k;
        *lo = a < b ? a : b;
        *hi = a < b ? b : a;
        return 1;
    }
    *lo = -INFINITY;
    *hi = INFINITY;
    return fabs(c) <= r;
}

static double score_one(const double *p, const Canvas *cv, double *col) {
    const int W = cv->W, H = cv->H, stride = W + 1;
    const double cx = p[0], cy = p[1], hw = p[2] / 2, hh = p[3] / 2;
    const int circ = p[5] > 0.5;
    const double a = p[4] * (M_PI / 180.0), ca = cos(a), sa = sin(a);
    const double ia = 1 / (hw * hw), ib = 1 / (hh * hh);
    const double A = ca * ca * ia + sa * sa * ib;
    const double reach = hypot(p[2], p[3]) / 2;
    int y0 = (int)floor(cy - reach), y1 = (int)ceil(cy + reach) + 1;
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;

    long n = 0;
    double s0 = 0, s1 = 0, s2 = 0, q = 0, e = 0;
    for (int r = y0; r < y1; r++) {
        const double Y = r + 0.5 - cy;
        double lo, hi;
        if (circ) {
            const double B = 2 * ca * sa * Y * (ia - ib);
            const double C = Y * Y * (sa * sa * ia + ca * ca * ib) - 1;
            const double disc = B * B - 4 * A * C;
            if (disc < 0) continue;
            const double root = sqrt(disc);
            lo = (-B - root) / (2 * A);
            hi = (-B + root) / (2 * A);
        } else {
            double lo1, hi1, lo2, hi2;
            if (!line_range(ca, sa * Y, hw, &lo1, &hi1)) continue;
            if (!line_range(-sa, ca * Y, hh, &lo2, &hi2)) continue;
            lo = lo1 > lo2 ? lo1 : lo2;
            hi = hi1 < hi2 ? hi1 : hi2;
        }
        double fx0 = ceil(lo + cx - 0.5), fx1 = floor(hi + cx - 0.5) + 1;
        if (fx0 < 0) fx0 = 0;
        if (fx0 > W) fx0 = W;
        if (fx1 < 0) fx1 = 0;
        if (fx1 > W) fx1 = W;
        const int x0 = (int)fx0, x1 = (int)fx1;
        if (x1 <= x0) continue;
        const long i0 = (long)r * stride + x0, i1 = (long)r * stride + x1;
        n += x1 - x0;
        s0 += cv->ct[i1 * 3] - cv->ct[i0 * 3];
        s1 += cv->ct[i1 * 3 + 1] - cv->ct[i0 * 3 + 1];
        s2 += cv->ct[i1 * 3 + 2] - cv->ct[i0 * 3 + 2];
        q += cv->cq[i1] - cv->cq[i0];
        e += cv->ce[i1] - cv->ce[i0];
    }
    const double nn = n > 0 ? (double)n : 1.0;
    col[0] = s0 / nn;
    col[1] = s1 / nn;
    col[2] = s2 / nn;
    if (n < 2) return INFINITY;
    return q - (s0 * s0 + s1 * s1 + s2 * s2) / nn - e;
}

/* ------------------------------------------------------------------ random */

static uint64_t next_u64(uint64_t *s) { /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double uniform(uint64_t *s) { return (next_u64(s) >> 11) * (1.0 / 9007199254740992.0); }

static double gauss(uint64_t *s) {
    const double u1 = uniform(s), u2 = uniform(s);
    return sqrt(-2 * log(1 - u1)) * cos(2 * M_PI * u2);
}

static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* One random tweak, same moves as mutate() in fh6_sticker.py. */
static void mutate(double *m, int W, int H, uint64_t *s) {
    switch (next_u64(s) & 3) {
    case 0:
        m[0] = clampd(m[0] + gauss(s) * W * 0.05, 0, W);
        m[1] = clampd(m[1] + gauss(s) * H * 0.05, 0, H);
        break;
    case 1:
        m[2] = clampd(m[2] + gauss(s) * W * 0.05, 1.5, W * 1.5);
        break;
    case 2:
        m[3] = clampd(m[3] + gauss(s) * H * 0.05, 1.5, H * 1.5);
        break;
    default:
        m[4] = fmod(m[4] + gauss(s) * 20, 180);
        if (m[4] < 0) m[4] += 180;
    }
}

/* ------------------------------------------------------------------ exports */

EXPORT void fc_score(const double *P, int n, int W, int H, const double *ct, const double *cq,
                     const double *ce, double *delta, double *colors) {
    const Canvas cv = {W, H, ct, cq, ce};
    for (int i = 0; i < n; i++) delta[i] = score_one(P + 6 * i, &cv, colors + 3 * i);
}

/* Hill-climb shape p in place: `rounds` times, try `batch` random tweaks of it
 * and keep the best unless it raises *delta. Accepting equal scores lets the
 * shape drift across tweaks that cover the same pixels, which finds better fits
 * than stopping at the first plateau. col is the colour for the kept shape. */
EXPORT void fc_climb(double *p, double *delta, double *col, int rounds, int batch, int W, int H,
                     const double *ct, const double *cq, const double *ce, uint64_t seed) {
    const Canvas cv = {W, H, ct, cq, ce};
    uint64_t s = seed;
    double m[6], bm[6], c[3], bc[3];
    for (int r = 0; r < rounds; r++) {
        double bd = INFINITY;
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < 6; k++) m[k] = p[k];
            mutate(m, W, H, &s);
            const double d = score_one(m, &cv, c);
            if (d < bd) {
                bd = d;
                for (int k = 0; k < 6; k++) bm[k] = m[k];
                bc[0] = c[0], bc[1] = c[1], bc[2] = c[2];
            }
        }
        if (bd <= *delta) { /* equal: keep moving across flat stretches */
            *delta = bd;
            for (int k = 0; k < 6; k++) p[k] = bm[k];
            col[0] = bc[0], col[1] = bc[1], col[2] = bc[2];
        }
    }
}
