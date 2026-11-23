// forza-painter-geometrize-cuda: a drop-in replacement for the Go generator
// (same command line, settings, output files and log lines), with the
// layer search running on an NVIDIA GPU.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"
#include "platform.hpp"
#include "tools.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

const char* kVersion = "Version: cuda-1.0.0";

void usage() {
    std::printf(
        "Usage: forza-painter-geometrize-cuda [--settings path.ini|--profile name] [--output path]\n"
        "         [--preview path] [--seed n] [--resume checkpoint.json] [--cpu] <image-path>\n"
        "       forza-painter-geometrize-cuda --selftest [image]\n"
        "       forza-painter-geometrize-cuda --score <geometry.json> <image>\n");
}

std::string strip_dashes(const std::string& a) {
    if (a.rfind("--", 0) == 0) return a.substr(2);
    if (a.rfind("-", 0) == 0) return a.substr(1);
    return a;
}

// main.go's normalizePreviewPath: a folder gets "preview.png" inside it.
std::string normalize_preview(const std::string& path) {
    std::string abs = fpg::abs_path(path);
    if (abs.empty()) return abs;
    char last = abs.back();
    if (last == '/' || last == '\\' || fpg::is_dir(abs)) return fpg::path_join(abs, "preview.png");
    return abs;
}

int real_main(const std::vector<std::string>& args) {
    std::printf("%s\n", kVersion);
    std::fflush(stdout);

    if (!args.empty() && (args[0] == "--selftest" || args[0] == "-selftest"))
        return fpg::run_selftest(args.size() > 1 ? args[1] : "");
    if (!args.empty() && (args[0] == "--score" || args[0] == "-score")) {
        if (args.size() < 3) {
            usage();
            return 1;
        }
        return fpg::run_score(args[1], args[2]);
    }

    fpg::Options opts;
    std::string backend;
    std::vector<std::string> positional;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        // Options may come before or after the image path (main.go's
        // applyTrailingOptions); anything not starting with '-' is positional.
        if (a.size() < 2 || a[0] != '-') {
            positional.push_back(a);
            continue;
        }
        std::string name = strip_dashes(a), value;
        bool hasValue = false;
        size_t eq = name.find('=');
        if (eq != std::string::npos) {
            value = name.substr(eq + 1);
            name = name.substr(0, eq);
            hasValue = true;
        }
        if (name == "cpu") {
            opts.useCpu = true;
            continue;
        }
        if (name == "help" || name == "h") {
            usage();
            return 0;
        }
        static const char* withValue[] = {"settings", "profile", "output", "preview",
                                          "seed",     "backend", "resume"};
        bool known = false;
        for (const char* k : withValue) known = known || name == k;
        if (!known) {
            std::printf("flag provided but not defined: %s\n", a.c_str());
            usage();
            return 2;
        }
        if (!hasValue) {
            if (i + 1 >= args.size()) {
                std::printf("flag needs an argument: %s\n", a.c_str());
                return 2;
            }
            value = args[++i];
        }
        if (name == "settings") opts.settingsPath = value;
        else if (name == "profile") opts.profile = value;
        else if (name == "output") opts.outputPath = value;
        else if (name == "preview") opts.previewPath = value;
        else if (name == "resume") opts.resumePath = value;
        else if (name == "backend") backend = value;  // accepted for compatibility
        else if (name == "seed") opts.seed = std::strtoll(value.c_str(), nullptr, 10);
    }
    if (positional.empty()) {
        usage();
        return 1;
    }
    opts.imagePath = positional[0];
    opts.workspaceRoot = fpg::current_dir();
    if (!opts.outputPath.empty()) opts.outputPath = fpg::abs_path(opts.outputPath);
    if (!opts.previewPath.empty()) opts.previewPath = normalize_preview(opts.previewPath);

    try {
        fpg::run(opts);
    } catch (const std::exception& e) {
        std::printf("Error: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    std::printf("FINISHED\n");
    std::fflush(stdout);
    return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n > 0 ? (size_t)n - 1 : 0, '\0');
        if (n > 1) WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, &s[0], n, nullptr, nullptr);
        args.push_back(s);
    }
    return real_main(args);
}
#else
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    return real_main(args);
}
#endif
