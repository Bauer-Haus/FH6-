// The layer search, behind one interface with two versions: the CUDA one
// (the real generator) and a CPU one that makes the same decisions, used by
// the tests, by --selftest to check the GPU, and by --cpu runs.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core.hpp"
#include "image_io.hpp"

namespace fpg {

struct RunConfig {
    int randomCount = 0;  // random candidates per layer
    int starts = 0;       // hill-climb starting shapes
    int rounds = 0;       // hill-climb rounds (0 = no climb)
    int perRound = 0;     // tweaks per start per round
    int gridW = 64, gridH = 64;
    bool forceOpaque = false;
    ClimbSteps steps = {};
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string describe() const = 0;
    // Uploads the image and builds the running-total tables and error map.
    virtual void init(const PreparedImage& img, const RunConfig& cfg) = 0;
    // One layer: random batch, climb, round, and paint if it helps.
    virtual LayerResult run_layer(const LayerParams& p) = 0;
    // Paints the shapes in order, then refreshes the error map.
    virtual void apply_shapes(const std::vector<Candidate>& shapes) = 0;
    // Replaces the canvas and rebuilds everything that depends on it.
    virtual void reset_canvas(const std::vector<Px>& current) = 0;
    virtual void read_current(std::vector<Px>& out) = 0;

    // For tests and --selftest.
    virtual void score_batch(const std::vector<Candidate>& in, bool roundColor,
                             std::vector<float>& scores, std::vector<Candidate>& coloured) = 0;
    virtual void read_tables(std::vector<DynEntry>& dyn) = 0;
    virtual void read_cdf(std::vector<double>& cdf) = 0;
};

std::unique_ptr<Backend> make_cpu_backend();

#ifdef FPG_WITH_CUDA
// Throws std::runtime_error if no usable GPU is found.
std::unique_ptr<Backend> make_cuda_backend();
#endif

}  // namespace fpg
