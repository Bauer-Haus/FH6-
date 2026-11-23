// The extra modes: --selftest, --score and --bench-cpu.
#pragma once

#include <string>

#include "image_io.hpp"

namespace fpg {

// A made-up test picture: gradients, discs and a transparent corner.
PreparedImage synthetic_image(int width, int height, int maxResolution);

// Checks the GPU against the CPU reference. Returns the process exit code.
int run_selftest(const std::string& imagePath);

// Prints the RMSE of a geometry file against an image. Returns the exit code.
int run_score(const std::string& geometryPath, const std::string& imagePath);

}  // namespace fpg
