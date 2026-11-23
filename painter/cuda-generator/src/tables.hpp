// Host-side builders shared by both backends, and the slow per-pixel
// reference used to check them.
#pragma once

#include <vector>

#include "core.hpp"
#include "geometry_json.hpp"

namespace fpg {

// The fixed table (target sums and pixel counts) for every row.
std::vector<StatEntry> build_static_table(const Px* target, const uint8_t* mask, int width,
                                          int height);
// The changing table (canvas sums), built with the same steps as the GPU.
std::vector<DynEntry> build_dynamic_table(const Px* target, const Px* current,
                                          const uint8_t* mask, int width, int height);

// engine.go: computeTotalError. Returns the summed squared error over opaque
// pixels and sets opaquePixels.
double total_error(const Px* target, const Px* current, const uint8_t* mask, int width,
                   int height, int& opaquePixels);

// Pixel-by-pixel statistics of a shape: the slow way, as
// evaluate_candidates_v3 gathers them. Used to check the row tables.
Stats reference_stats(const Px* target, const Px* current, const uint8_t* mask, int width,
                      int height, const Candidate& c);

// Paints a shape pixel by pixel (render.ApplyEllipse with the kernel's
// bounding box). Used to check the row-run painting.
void reference_apply(Px* current, const uint8_t* mask, int width, int height, const Candidate& c);

// engine.go: pruneOccludedShapes. Removes shapes hidden by later opaque ones.
std::vector<ShapeOut> prune_occluded(const std::vector<ShapeOut>& shapes, int width, int height,
                                     const uint8_t* mask);

}  // namespace fpg
