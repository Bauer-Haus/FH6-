#include "settings.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <sstream>

#include "platform.hpp"

namespace fpg {
namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string lower(std::string s) {
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

// strconv.Atoi: an optional sign and decimal digits, nothing else.
int parse_int(const std::string& value, int fallback) {
    std::string v = trim(value);
    if (v.empty()) return fallback;
    size_t i = (v[0] == '+' || v[0] == '-') ? 1 : 0;
    if (i == v.size()) return fallback;
    for (size_t j = i; j < v.size(); ++j)
        if (!std::isdigit((unsigned char)v[j])) return fallback;
    errno = 0;
    long long n = std::strtoll(v.c_str(), nullptr, 10);
    if (errno != 0 || n > 2147483647LL || n < -2147483648LL) return fallback;
    return (int)n;
}

float parse_float(const std::string& value, float fallback) {
    std::string v = trim(value);
    if (v.empty()) return fallback;
    char* end = nullptr;
    errno = 0;
    double d = std::strtod(v.c_str(), &end);
    if (errno != 0 || end == v.c_str() || *end != '\0') return fallback;
    return (float)d;
}

bool parse_bool(const std::string& value, bool fallback) {
    std::string v = lower(trim(value));
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return fallback;
}

std::set<int> parse_save_at(const std::string& value) {
    std::set<int> out;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, ',')) {
        int n = parse_int(part, -1);
        if (n >= 1) out.insert(n);
    }
    if (out.empty()) out.insert(3000);
    return out;
}

void note_ignored(Settings& cfg, const std::string& key) {
    if (std::find(cfg.ignoredKeys.begin(), cfg.ignoredKeys.end(), key) == cfg.ignoredKeys.end())
        cfg.ignoredKeys.push_back(key);
}

void apply_line(Settings& cfg, const std::string& rawLine) {
    std::string line = trim(rawLine);
    if (line.empty() || line[0] == '#' || line[0] == ';') return;
    size_t eq = line.find('=');
    if (eq == std::string::npos) return;
    std::string key = trim(line.substr(0, eq));
    std::string value = trim(line.substr(eq + 1));

    if (key == "description") cfg.description = value;
    else if (key == "maxPreviewSize") cfg.maxPreviewSize = parse_int(value, cfg.maxPreviewSize);
    else if (key == "maxResolution") cfg.maxResolution = parse_int(value, cfg.maxResolution);
    else if (key == "maxThreads") {
        cfg.maxThreads = parse_int(value, cfg.maxThreads);
        note_ignored(cfg, key);
    } else if (key == "mutatedSamples") cfg.mutatedSamples = parse_int(value, cfg.mutatedSamples);
    else if (key == "forceOpaqueShapes") cfg.forceOpaqueShapes = parse_bool(value, cfg.forceOpaqueShapes);
    else if (key == "enableMultiPrimitiveShapes")
        cfg.enableMultiPrimitiveShapes = parse_bool(value, cfg.enableMultiPrimitiveShapes);
    else if (key == "posterizeLevels") cfg.posterizeLevels = parse_int(value, cfg.posterizeLevels);
    else if (key == "previewEvery") cfg.previewEvery = parse_int(value, cfg.previewEvery);
    else if (key == "randomSamples") cfg.randomSamples = parse_int(value, cfg.randomSamples);
    else if (key == "saveAt") cfg.saveAt = parse_save_at(value);
    else if (key == "saveEvery") cfg.saveEvery = parse_int(value, cfg.saveEvery);
    else if (key == "stopAt") cfg.stopAt = parse_int(value, cfg.stopAt);
    else if (key == "useWorkGroupEval") {
        cfg.useWorkGroupEval = parse_bool(value, cfg.useWorkGroupEval);
        note_ignored(cfg, key);
    } else if (key == "enableProgressiveSampling") {
        cfg.enableProgressiveSampling = parse_bool(value, cfg.enableProgressiveSampling);
        if (cfg.enableProgressiveSampling) note_ignored(cfg, key);
    } else if (key == "progressiveSamplingStart")
        cfg.progressiveSamplingStart = parse_int(value, cfg.progressiveSamplingStart);
    else if (key == "progressiveSamplingEnd")
        cfg.progressiveSamplingEnd = parse_int(value, cfg.progressiveSamplingEnd);
    else if (key == "progressiveSamplingTransition")
        cfg.progressiveSamplingTransition = parse_float(value, cfg.progressiveSamplingTransition);
    else if (key == "progressiveSamplingCurve")
        cfg.progressiveSamplingCurve = parse_float(value, cfg.progressiveSamplingCurve);
    else if (key == "errorGridSize") cfg.errorGridSize = parse_int(value, cfg.errorGridSize);
    else if (key == "loadGeometry") cfg.loadGeometry = value;
    else if (key == "errorMetric") cfg.errorMetric = lower(trim(value));
    else if (key == "randomSampleBudget") {
        cfg.randomSampleBudget = parse_int(value, cfg.randomSampleBudget);
        if (cfg.randomSampleBudget > 0) note_ignored(cfg, key);
    } else if (key == "ssimWeight") cfg.ssimWeight = parse_float(value, cfg.ssimWeight);
    else if (key == "climbStarts") cfg.climbStarts = parse_int(value, cfg.climbStarts);
    else if (key == "climbRounds") cfg.climbRounds = parse_int(value, cfg.climbRounds);
}

void finish(Settings& cfg) {
    if (cfg.randomSamples < 1) cfg.randomSamples = 1;
    if (cfg.mutatedSamples < 0) cfg.mutatedSamples = 0;
    if (cfg.stopAt < 1) cfg.stopAt = 1;
    if (cfg.saveEvery < 1) cfg.saveEvery = 1;
    if (cfg.progressiveSamplingStart < 1) cfg.progressiveSamplingStart = 1;
    if (cfg.progressiveSamplingEnd < 1) cfg.progressiveSamplingEnd = 1;
    if (cfg.progressiveSamplingTransition <= 0) cfg.progressiveSamplingTransition = 0.333f;
    if (cfg.progressiveSamplingCurve <= 0) cfg.progressiveSamplingCurve = 2.0f;
    if (cfg.climbStarts < 0) cfg.climbStarts = 0;
    if (cfg.climbRounds < 0) cfg.climbRounds = 0;
}

}  // namespace

std::string resolve_settings_path(const std::string& rootDir, const std::string& explicitPath,
                                  const std::string& profile, std::string& err) {
    if (!explicitPath.empty()) return explicitPath;
    const std::string settingsDir = path_join(rootDir, "settings");
    if (!profile.empty()) {
        std::vector<std::string> names;
        if (!list_files(settingsDir, names)) {
            err = "cannot read settings folder " + settingsDir;
            return "";
        }
        std::sort(names.begin(), names.end());
        const std::string needle = lower(profile);
        for (const auto& name : names) {
            std::string l = lower(name);
            if (l.find(needle) != std::string::npos && l.size() >= 4 &&
                l.compare(l.size() - 4, 4, ".ini") == 0)
                return path_join(settingsDir, name);
        }
        err = "profile \"" + profile + "\" not found in " + settingsDir;
        return "";
    }
    return path_join(settingsDir, "_default.ini");
}

Settings parse_settings_text(const std::string& text) {
    Settings cfg;
    std::string body = text;
    if (body.size() >= 3 && (unsigned char)body[0] == 0xEF && (unsigned char)body[1] == 0xBB &&
        (unsigned char)body[2] == 0xBF)
        body = body.substr(3);
    std::stringstream ss(body);
    std::string line;
    while (std::getline(ss, line)) apply_line(cfg, line);
    finish(cfg);
    return cfg;
}

bool parse_settings(const std::string& path, Settings& out, std::string& err) {
    std::string text;
    if (!read_file(path, text)) {
        err = "open " + path + ": cannot read file";
        return false;
    }
    out = parse_settings_text(text);
    return true;
}

}  // namespace fpg
