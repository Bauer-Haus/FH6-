#include "image_io.hpp"

#include <cmath>

#include "platform.hpp"

#ifdef _WIN32
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fpg {

bool load_image16(const std::string& path, Image16& out, std::string& err) {
    int w = 0, h = 0, n = 0;
    FILE* f = fopen_utf8(path, "rb");
    if (!f) {
        err = "open " + path + ": cannot read file";
        return false;
    }
    const bool is16 = stbi_is_16_bit_from_file(f) != 0;
    std::fseek(f, 0, SEEK_SET);
    out.pix.clear();
    if (is16) {
        stbi_us* data = stbi_load_from_file_16(f, &w, &h, &n, 4);
        if (data) {
            out.pix.resize((size_t)w * h * 4);
            for (size_t i = 0; i < (size_t)w * h; ++i) {
                // color.NRGBA64.RGBA()
                uint32_t a = data[i * 4 + 3];
                out.pix[i * 4 + 0] = (uint32_t)data[i * 4 + 0] * a / 0xffff;
                out.pix[i * 4 + 1] = (uint32_t)data[i * 4 + 1] * a / 0xffff;
                out.pix[i * 4 + 2] = (uint32_t)data[i * 4 + 2] * a / 0xffff;
                out.pix[i * 4 + 3] = a;
            }
            stbi_image_free(data);
        }
    } else {
        stbi_uc* data = stbi_load_from_file(f, &w, &h, &n, 4);
        if (data) {
            out.pix.resize((size_t)w * h * 4);
            for (size_t i = 0; i < (size_t)w * h; ++i) {
                // color.NRGBA.RGBA()
                uint32_t a = (uint32_t)data[i * 4 + 3] * 0x101;
                out.pix[i * 4 + 0] = (uint32_t)data[i * 4 + 0] * a / 0xff;
                out.pix[i * 4 + 1] = (uint32_t)data[i * 4 + 1] * a / 0xff;
                out.pix[i * 4 + 2] = (uint32_t)data[i * 4 + 2] * a / 0xff;
                out.pix[i * 4 + 3] = a;
            }
            stbi_image_free(data);
        }
    }
    std::fclose(f);
    if (out.pix.empty()) {
        const char* why = stbi_failure_reason();
        err = std::string("image: unknown format or unreadable file (") + (why ? why : "?") + ")";
        return false;
    }
    out.width = w;
    out.height = h;
    return true;
}

namespace {

struct Contrib {
    int coord;
    double weight;
};
struct Source {
    int i, j;
    double invTotalWeight, invTotalWeightFFFF;
};
struct Distrib {
    std::vector<Source> sources;
    std::vector<Contrib> contribs;
};

double catmull_rom(double t) {
    if (t < 1) return double((double(1.5 * t) - 2.5) * t * t) + 1;
    return double((double(double(double(-0.5 * t) + 2.5) * t) - 4) * t) + 2;
}

// x/image/draw newDistrib.
Distrib new_distrib(int dw, int sw) {
    const double support = 2.0;
    double scale = double(sw) / double(dw);
    double halfWidth = support, kernelArgScale = 1.0;
    if (scale > 1) {
        halfWidth *= scale;
        kernelArgScale = 1 / scale;
    }
    Distrib d;
    std::vector<double> centers((size_t)dw);
    std::vector<int> lo((size_t)dw), hi((size_t)dw);
    for (int x = 0; x < dw; ++x) {
        double center = double((double(x) + 0.5) * scale) - 0.5;
        int i = (int)std::floor(center - halfWidth);
        if (i < 0) i = 0;
        int j = (int)std::ceil(center + halfWidth);
        if (j > sw) {
            j = sw;
            if (j < i) j = i;
        }
        centers[x] = center;
        lo[x] = i;
        hi[x] = j;
    }
    d.sources.resize((size_t)dw);
    for (int k = 0; k < dw; ++k) {
        double total = 0.0;
        int l = (int)d.contribs.size();
        for (int coord = lo[k]; coord < hi[k]; ++coord) {
            double t = std::fabs((centers[k] - double(coord)) * kernelArgScale);
            if (t >= support) continue;
            double weight = catmull_rom(t);
            if (weight == 0) continue;
            total += weight;
            d.contribs.push_back({coord, weight});
        }
        total = 1 / total;
        d.sources[k] = {l, (int)d.contribs.size(), total, total / 0xffff};
    }
    return d;
}

uint32_t ftou(double f) {
    int32_t i = (int32_t)(double(0xffff * f) + 0.5);
    if (i > 0xffff) return 0xffff;
    if (i > 0) return (uint32_t)i;
    return 0;
}

}  // namespace

Image16 resize_catmull_rom(const Image16& src, int dw, int dh) {
    const int sw = src.width, sh = src.height;
    Distrib hz = new_distrib(dw, sw);
    Distrib vt = new_distrib(dh, sh);
    // scaleX: rows of the source into a dw x sh buffer.
    std::vector<double> tmp((size_t)dw * sh * 4);
    size_t t = 0;
    for (int y = 0; y < sh; ++y) {
        for (const Source& s : hz.sources) {
            double pr = 0, pg = 0, pb = 0, pa = 0;
            for (int c = s.i; c < s.j; ++c) {
                const Contrib& ct = hz.contribs[c];
                const uint32_t* p = &src.pix[((size_t)y * sw + ct.coord) * 4];
                pr += double(double(p[0]) * ct.weight);
                pg += double(double(p[1]) * ct.weight);
                pb += double(double(p[2]) * ct.weight);
                pa += double(double(p[3]) * ct.weight);
            }
            tmp[t * 4 + 0] = pr * s.invTotalWeightFFFF;
            tmp[t * 4 + 1] = pg * s.invTotalWeightFFFF;
            tmp[t * 4 + 2] = pb * s.invTotalWeightFFFF;
            tmp[t * 4 + 3] = pa * s.invTotalWeightFFFF;
            ++t;
        }
    }
    // scaleY_RGBA_Over onto an all-zero image.RGBA, read back with RGBA().
    Image16 out;
    out.width = dw;
    out.height = dh;
    out.pix.assign((size_t)dw * dh * 4, 0);
    for (int dx = 0; dx < dw; ++dx) {
        for (int dy = 0; dy < dh; ++dy) {
            const Source& s = vt.sources[dy];
            double pr = 0, pg = 0, pb = 0, pa = 0;
            for (int c = s.i; c < s.j; ++c) {
                const Contrib& ct = vt.contribs[c];
                const double* p = &tmp[((size_t)ct.coord * dw + dx) * 4];
                pr += double(p[0] * ct.weight);
                pg += double(p[1] * ct.weight);
                pb += double(p[2] * ct.weight);
                pa += double(p[3] * ct.weight);
            }
            if (pr > pa) pr = pa;
            if (pg > pa) pg = pa;
            if (pb > pa) pb = pa;
            uint32_t v[4] = {ftou(pr * s.invTotalWeight), ftou(pg * s.invTotalWeight),
                             ftou(pb * s.invTotalWeight), ftou(pa * s.invTotalWeight)};
            uint32_t* o = &out.pix[((size_t)dy * dw + dx) * 4];
            for (int k = 0; k < 4; ++k) {
                uint32_t b8 = (v[k] >> 8) & 0xff;  // dst.Pix byte
                o[k] = b8 | (b8 << 8);             // color.RGBA.RGBA()
            }
        }
    }
    return out;
}

PreparedImage prepare(const Image16& loaded, int maxResolution) {
    const Image16* src = &loaded;
    Image16 scaled;
    if (maxResolution > 0) {
        int maxDim = loaded.width > loaded.height ? loaded.width : loaded.height;
        if (maxDim > maxResolution) {
            double scale = double(maxResolution) / double(maxDim);
            int nw = (int)std::round(double(loaded.width) * scale);
            int nh = (int)std::round(double(loaded.height) * scale);
            if (nw < 1) nw = 1;
            if (nh < 1) nh = 1;
            scaled = resize_catmull_rom(loaded, nw, nh);
            src = &scaled;
        }
    }
    PreparedImage p;
    const int w = src->width, h = src->height;
    p.width = w;
    p.height = h;
    p.target.resize((size_t)w * h);
    p.current.resize((size_t)w * h);
    p.mask.assign((size_t)w * h, 0);
    double sumR = 0, sumG = 0, sumB = 0, sumA = 0, opaqueCount = 0;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        float r = float(src->pix[i * 4 + 0]) / 65535.0f;
        float g = float(src->pix[i * 4 + 1]) / 65535.0f;
        float b = float(src->pix[i * 4 + 2]) / 65535.0f;
        float a = float(src->pix[i * 4 + 3]) / 65535.0f;
        if (a > 0 && a < 1) {
            r = r / a < 1 ? r / a : 1;
            g = g / a < 1 ? g / a : 1;
            b = b / a < 1 ? b / a : 1;
        }
        p.target[i] = {r, g, b, a};
        if (a > 0.001f) {
            p.mask[i] = 1;
            sumR += r;
            sumG += g;
            sumB += b;
            sumA += a;
            opaqueCount += 1;
        } else {
            p.hasTransparency = true;
        }
    }
    auto c255 = [](double v) { return v < 0 ? 0.0 : (v > 255 ? 255.0 : v); };
    if (opaqueCount > 0) {
        p.background[0] = (uint8_t)c255(sumR / opaqueCount * 255.0);
        p.background[1] = (uint8_t)c255(sumG / opaqueCount * 255.0);
        p.background[2] = (uint8_t)c255(sumB / opaqueCount * 255.0);
        p.background[3] = (uint8_t)c255(sumA / opaqueCount * 255.0);
    }
    // A transparent source gets background alpha 0, so the importer skips the
    // background rectangle; its RGB stays the mean colour (see imageutil).
    if (p.hasTransparency) p.background[3] = 0;
    for (auto& c : p.current) {
        if (p.hasTransparency) c = {0, 0, 0, 0};
        else c = {float(p.background[0]) / 255.0f, float(p.background[1]) / 255.0f,
                  float(p.background[2]) / 255.0f, 1.0f};
    }
    return p;
}

bool load_and_prepare(const std::string& path, int maxResolution, PreparedImage& out,
                      std::string& err) {
    Image16 img;
    if (!load_image16(path, img, err)) return false;
    if (img.width <= 0 || img.height <= 0) {
        err = "invalid image size";
        return false;
    }
    out = prepare(img, maxResolution);
    return true;
}

static void png_sink(void* ctx, void* data, int size) {
    static_cast<std::string*>(ctx)->append(static_cast<const char*>(data), (size_t)size);
}

bool save_png(const std::string& path, const std::vector<Px>& pix, int width, int height) {
    std::vector<uint8_t> rgba((size_t)width * height * 4);
    for (size_t i = 0; i < pix.size(); ++i) {
        rgba[i * 4 + 0] = to_byte(pix[i].r);
        rgba[i * 4 + 1] = to_byte(pix[i].g);
        rgba[i * 4 + 2] = to_byte(pix[i].b);
        rgba[i * 4 + 3] = to_byte(pix[i].a);
    }
    std::string buf;
    stbi_write_png_compression_level = 2;  // previews are rewritten often
    if (!stbi_write_png_to_func(png_sink, &buf, width, height, 4, rgba.data(), width * 4))
        return false;
    std::string dir = dir_name(path);
    if (!dir.empty()) make_dirs(dir);
    return write_file_atomic(path, buf);
}

}  // namespace fpg
