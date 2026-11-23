#include "tables.hpp"

#include <cmath>

namespace fpg {

std::vector<StatEntry> build_static_table(const Px* target, const uint8_t* mask, int width,
                                          int height) {
    const long long stride = (long long)width + 1;
    std::vector<StatEntry> st((size_t)(stride * height));
    for (int y = 0; y < height; ++y) {
        StatEntry acc = {0.0, 0.0, 0.0, 0.0, 0, 0};
        st[(size_t)(y * stride)] = acc;
        for (int x = 0; x < width; ++x) {
            const long long p = (long long)y * width + x;
            if (mask[p]) {
                acc.t0 += (double)target[p].r;
                acc.t1 += (double)target[p].g;
                acc.t2 += (double)target[p].b;
                acc.t3 += (double)target[p].a;
                acc.n += 1;
            } else {
                acc.nt += 1;
            }
            st[(size_t)(y * stride + x + 1)] = acc;
        }
    }
    return st;
}

std::vector<DynEntry> build_dynamic_table(const Px* target, const Px* current,
                                          const uint8_t* mask, int width, int height) {
    const long long stride = (long long)width + 1;
    std::vector<DynEntry> dyn((size_t)(stride * height), DynEntry{0, 0, 0, 0, 0, 0});
    for (int y = 0; y < height; ++y) rebuild_row(dyn.data(), target, current, mask, width, y, 0);
    return dyn;
}

double total_error(const Px* target, const Px* current, const uint8_t* mask, int width,
                   int height, int& opaquePixels) {
    double total = 0.0;
    opaquePixels = 0;
    const long long n = (long long)width * height;
    for (long long p = 0; p < n; ++p) {
        if (!mask[p]) continue;
        ++opaquePixels;
        double dr = (double)(target[p].r - current[p].r);
        double dg = (double)(target[p].g - current[p].g);
        double db = (double)(target[p].b - current[p].b);
        double da = (double)(target[p].a - current[p].a);
        total += dr * dr + dg * dg + db * db + da * da;
    }
    return total;
}

Stats reference_stats(const Px* target, const Px* current, const uint8_t* mask, int width,
                      int height, const Candidate& c) {
    Geom g = make_geom(c, width, height);
    Stats s;
    s.clear();
    for (int y = g.yMin; y <= g.yMax; ++y) {
        const float dy = ((float)y + 0.5f) - g.cy;
        for (int x = g.xMin; x <= g.xMax; ++x) {
            if (!inside(g, x, dy)) continue;
            const long long p = (long long)y * width + x;
            if (!mask[p]) {
                ++s.nt;
                continue;
            }
            const Px t = target[p], v = current[p];
            s.t0 += t.r;
            s.t1 += t.g;
            s.t2 += t.b;
            s.t3 += t.a;
            s.s0 += v.r;
            s.s1 += v.g;
            s.s2 += v.b;
            s.s3 += v.a;
            s.ss += (double)v.r * v.r + (double)v.g * v.g + (double)v.b * v.b + (double)v.a * v.a;
            s.ts += (double)t.r * v.r + (double)t.g * v.g + (double)t.b * v.b + (double)t.a * v.a;
            ++s.n;
        }
    }
    return s;
}

void reference_apply(Px* current, const uint8_t* mask, int width, int height, const Candidate& c) {
    Geom g = make_geom(c, width, height);
    for (int y = g.yMin; y <= g.yMax; ++y) {
        const float dy = ((float)y + 0.5f) - g.cy;
        for (int x = g.xMin; x <= g.xMax; ++x) {
            const long long p = (long long)y * width + x;
            if (!mask[p] || !inside(g, x, dy)) continue;
            blend(current[p], c);
        }
    }
}

// engine.go: shapeContainsPoint and shapeExtents.
static bool contains(int type, float xr, float yr, float rx, float ry, float invRX2, float invRY2) {
    switch (type) {
    case 0:
        return std::fabs(xr) <= rx && std::fabs(yr) <= ry;
    case 2: {
        if (yr < -ry || yr > ry) return false;
        float halfWidth = rx * (yr + ry) / (2.0f * ry);
        return std::fabs(xr) <= halfWidth;
    }
    default:
        return xr * xr * invRX2 + yr * yr * invRY2 <= 1.0f;
    }
}

std::vector<ShapeOut> prune_occluded(const std::vector<ShapeOut>& shapes, int width, int height,
                                     const uint8_t* mask) {
    if (shapes.size() <= 1) return shapes;
    std::vector<uint8_t> cov((size_t)width * height, 0);
    std::vector<bool> keep(shapes.size(), false);
    keep[0] = true;
    for (size_t j = shapes.size() - 1; j >= 1; --j) {
        const ShapeOut& s = shapes[j];
        if ((s.type != 0 && s.type != 2 && s.type != 16) || s.data.size() < 5) {
            keep[j] = true;
            continue;
        }
        float cx = (float)s.data[0], cy = (float)s.data[1];
        float rx = (float)s.data[2], ry = (float)s.data[3];
        float theta = (float)s.data[4];
        int alpha = s.color[3];
        if (rx < 1) rx = 1;
        if (ry < 1) ry = 1;
        float t = theta * (float)(3.14159265358979323846 / 180.0);
        float cosT = (float)std::cos((double)t);
        float sinT = (float)std::sin((double)t);
        float invRX2 = 1.0f / (rx * rx);
        float invRY2 = 1.0f / (ry * ry);
        float ex, ey;
        if (s.type == 16) {
            ex = (float)std::sqrt((double)(rx * rx * cosT * cosT + ry * ry * sinT * sinT));
            ey = (float)std::sqrt((double)(rx * rx * sinT * sinT + ry * ry * cosT * cosT));
        } else {
            float ac = std::fabs(cosT), as = std::fabs(sinT);
            ex = rx * ac + ry * as;
            ey = rx * as + ry * ac;
        }
        int xMin = clampi((int)std::floor((double)(cx - ex - 1)), 0, width - 1);
        int xMax = clampi((int)std::ceil((double)(cx + ex + 1)), 0, width - 1);
        int yMin = clampi((int)std::floor((double)(cy - ey - 1)), 0, height - 1);
        int yMax = clampi((int)std::ceil((double)(cy + ey + 1)), 0, height - 1);

        bool occluded = true, anyInside = false;
        for (int y = yMin; y <= yMax && occluded; ++y) {
            for (int x = xMin; x <= xMax; ++x) {
                const long long p = (long long)y * width + x;
                if (!mask[p]) continue;
                float dx = (float)x + 0.5f - cx, dy = (float)y + 0.5f - cy;
                float xr = dx * cosT + dy * sinT;
                float yr = -dx * sinT + dy * cosT;
                if (contains(s.type, xr, yr, rx, ry, invRX2, invRY2)) {
                    anyInside = true;
                    if (cov[(size_t)p] == 0) {
                        occluded = false;
                        break;
                    }
                }
            }
        }
        if (!anyInside) occluded = true;
        keep[j] = !occluded;
        if (!occluded && alpha == 255) {
            for (int y = yMin; y <= yMax; ++y) {
                for (int x = xMin; x <= xMax; ++x) {
                    const long long p = (long long)y * width + x;
                    if (!mask[p]) continue;
                    float dx = (float)x + 0.5f - cx, dy = (float)y + 0.5f - cy;
                    float xr = dx * cosT + dy * sinT;
                    float yr = -dx * sinT + dy * cosT;
                    if (contains(s.type, xr, yr, rx, ry, invRX2, invRY2)) cov[(size_t)p] = 1;
                }
            }
        }
    }
    std::vector<ShapeOut> out;
    out.reserve(shapes.size());
    for (size_t j = 0; j < shapes.size(); ++j)
        if (keep[j]) out.push_back(shapes[j]);
    return out;
}

}  // namespace fpg
