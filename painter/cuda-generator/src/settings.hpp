// Settings file (INI) reader. A port of painter/generator/internal/config.
#pragma once

#include <set>
#include <string>
#include <vector>

namespace fpg {

struct Settings {
    std::string description = "Default profile";
    int maxPreviewSize = 500;
    int maxResolution = 2000;
    int maxThreads = 0;
    int mutatedSamples = 1000;
    bool forceOpaqueShapes = false;
    bool enableMultiPrimitiveShapes = false;
    int posterizeLevels = 20;
    int previewEvery = 10;
    int randomSamples = 3000;
    std::set<int> saveAt = {500, 1000, 1500, 2000, 2500, 3000};
    int saveEvery = 10;
    int stopAt = 3000;
    bool useWorkGroupEval = true;
    bool enableProgressiveSampling = false;
    int progressiveSamplingStart = 4;
    int progressiveSamplingEnd = 1;
    float progressiveSamplingTransition = 0.333f;
    float progressiveSamplingCurve = 2.5f;
    int errorGridSize = 64;
    std::string loadGeometry;
    std::string errorMetric = "mse";
    float ssimWeight = 0.5f;
    int randomSampleBudget = 0;

    // New in the CUDA generator. 0 means "pick automatically".
    int climbStarts = 0;
    int climbRounds = 0;

    // Keys that were set in the file but have no effect here.
    std::vector<std::string> ignoredKeys;
};

// config.ResolveSettingsPath. Returns "" and sets err on failure.
std::string resolve_settings_path(const std::string& rootDir, const std::string& explicitPath,
                                  const std::string& profile, std::string& err);

// config.ParseSettings. Returns false and sets err if the file can't be read.
bool parse_settings(const std::string& path, Settings& out, std::string& err);

// Same as parse_settings, from text (used by the tests).
Settings parse_settings_text(const std::string& text);

}  // namespace fpg
