// Geometry JSON: {"shapes":[{type,data,color,score}]}, written in the same
// layout as Go's json.MarshalIndent(payload, "", "  ").
#pragma once

#include <string>
#include <vector>

namespace fpg {

struct ShapeOut {
    int type = 16;              // 1 = background rectangle, 16 = rotated ellipse
    std::vector<double> data;   // background: [0,0,w,h]; ellipse: [cx,cy,rx,ry,angle]
    int color[4] = {0, 0, 0, 0};
    int colorCount = 4;         // how many colour values a loaded file had
    double score = 0.0;
};

// Go's encoding/json formatting of a float64.
std::string format_go_float(double v);

std::string geometry_to_json(const std::vector<ShapeOut>& shapes);
bool save_geometry(const std::string& path, const std::vector<ShapeOut>& shapes);
bool load_geometry(const std::string& path, std::vector<ShapeOut>& shapes, std::string& err);
bool parse_geometry(const std::string& text, std::vector<ShapeOut>& shapes, std::string& err);

// output.BuildOutputPath / BuildFinalOutputPath.
std::string build_output_path(const std::string& base, int shapeCount);
std::string build_final_output_path(const std::string& base);

}  // namespace fpg
