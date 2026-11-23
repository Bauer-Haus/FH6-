// Image loading and preparation (port of internal/imageutil) and PNG output.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core.hpp"

namespace fpg {

struct PreparedImage {
    int width = 0, height = 0;
    std::vector<Px> target;    // straight (not premultiplied) colour, 0..1
    std::vector<Px> current;   // the starting canvas
    std::vector<uint8_t> mask; // 1 where the target is opaque (alpha > 0.001)
    bool hasTransparency = false;
    uint8_t background[4] = {0, 0, 0, 255};
};

// Premultiplied 16-bit RGBA, as Go's color.Color.RGBA() returns it.
struct Image16 {
    int width = 0, height = 0;
    std::vector<uint32_t> pix;  // 4 values per pixel, each 0..0xffff
};

bool load_image16(const std::string& path, Image16& out, std::string& err);

// The Catmull-Rom resize of golang.org/x/image/draw (Kernel.Scale onto an
// empty image.RGBA), including its 8-bit result.
Image16 resize_catmull_rom(const Image16& src, int dw, int dh);

// imageutil.LoadAndPrepare.
bool load_and_prepare(const std::string& path, int maxResolution, PreparedImage& out,
                      std::string& err);
// The part of LoadAndPrepare after decoding.
PreparedImage prepare(const Image16& src, int maxResolution);

// Writes straight RGBA (0..1) as a PNG, atomically.
bool save_png(const std::string& path, const std::vector<Px>& pix, int width, int height);

}  // namespace fpg
