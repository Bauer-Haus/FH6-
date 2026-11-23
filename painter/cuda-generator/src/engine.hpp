// The run loop: a port of engine.Run from the Go generator.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "backend.hpp"
#include "settings.hpp"

namespace fpg {

struct Options {
    std::string imagePath;
    std::string settingsPath;
    std::string profile;
    std::string outputPath;
    std::string previewPath;
    std::string workspaceRoot;
    int64_t seed = 0;
    std::string resumePath;
    bool useCpu = false;
};

// Prints a line and flushes, so the app reading the pipe sees it at once.
void logf(const char* fmt, ...);

// The hill-climb plan and the other per-run numbers, from the settings.
RunConfig plan_run(const Settings& cfg, int width, int height);
// The per-layer numbers that change as the run goes on.
LayerParams layer_params(const RunConfig& rc, uint64_t seed, uint64_t attempt, int width,
                         int height, int acceptedShapes, int stopAt, bool forceOpaque);

// engine.go: normalizeScore (the "score" saved with each shape).
double normalize_score(double totalError, double denom);

// engine.go: mutationSteps.
void mutation_steps(int width, int height, float& move, float& radius);

// Throws std::runtime_error on failure (the message is printed as "Error: ...").
void run(const Options& opts);

}  // namespace fpg
