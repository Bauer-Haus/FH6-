#include "platform.hpp"

#include <atomic>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace fpg {

#ifdef _WIN32
static std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

static std::string narrow(const std::wstring& w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
#endif

char path_sep() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

static bool is_sep(char c) {
#ifdef _WIN32
    return c == '\\' || c == '/';
#else
    return c == '/';
#endif
}

FILE* fopen_utf8(const std::string& path, const char* mode) {
#ifdef _WIN32
    return _wfopen(widen(path).c_str(), widen(mode).c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

bool read_file(const std::string& path, std::string& out) {
    FILE* f = fopen_utf8(path, "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

static bool write_plain(const std::string& path, const std::string& data) {
    FILE* f = fopen_utf8(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

static bool rename_over(const std::string& from, const std::string& to) {
#ifdef _WIN32
    return MoveFileExW(widen(from).c_str(), widen(to).c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

static void remove_file(const std::string& path) {
#ifdef _WIN32
    DeleteFileW(widen(path).c_str());
#else
    std::remove(path.c_str());
#endif
}

bool write_file_atomic(const std::string& path, const std::string& data) {
    static std::atomic<unsigned> counter{0};
    std::string dir = dir_name(path);
    std::string base = path.substr(dir.empty() ? 0 : dir.size() + 1);
    unsigned long long stamp =
        (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
    std::string tmp = path_join(dir.empty() ? "." : dir,
                                "." + base + "." + std::to_string(stamp % 1000000007ULL) + "_" +
                                    std::to_string(counter++) + ".tmp");
    if (write_plain(tmp, data)) {
        if (rename_over(tmp, path)) return true;
    }
    remove_file(tmp);
    return write_plain(path, data);
}

bool make_dirs(const std::string& dir) {
    if (dir.empty() || is_dir(dir)) return true;
    std::string parent = dir_name(dir);
    if (!parent.empty() && parent != dir && !make_dirs(parent)) return false;
#ifdef _WIN32
    return CreateDirectoryW(widen(dir).c_str(), nullptr) != 0 ||
           GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

bool list_files(const std::string& dir, std::vector<std::string>& names) {
    names.clear();
#ifdef _WIN32
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(widen(path_join(dir, "*")).c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) names.push_back(narrow(fd.cFileName));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return true;
#else
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    while (dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        if (!is_dir(path_join(dir, name))) names.push_back(name);
    }
    closedir(d);
    return true;
#endif
}

bool is_dir(const std::string& path) {
#ifdef _WIN32
    DWORD a = GetFileAttributesW(widen(path).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

bool is_abs(const std::string& path) {
#ifdef _WIN32
    if (path.size() >= 3 && path[1] == ':' && is_sep(path[2])) return true;
    return path.size() >= 2 && is_sep(path[0]) && is_sep(path[1]);
#else
    return !path.empty() && path[0] == '/';
#endif
}

std::string current_dir() {
#ifdef _WIN32
    wchar_t buf[32768];
    DWORD n = GetCurrentDirectoryW(32768, buf);
    return n ? narrow(std::wstring(buf, n)) : std::string(".");
#else
    char buf[4096];
    return getcwd(buf, sizeof buf) ? std::string(buf) : std::string(".");
#endif
}

std::string abs_path(const std::string& path) {
    if (path.empty() || is_abs(path)) return path;
#ifdef _WIN32
    std::wstring w = widen(path);
    DWORD n = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
    if (n == 0) return path;
    std::wstring out((size_t)n, L'\0');
    n = GetFullPathNameW(w.c_str(), n, &out[0], nullptr);
    out.resize(n);
    return narrow(out);
#else
    return path_join(current_dir(), path);
#endif
}

std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (is_sep(a.back())) return a + b;
    return a + path_sep() + b;
}

std::string dir_name(const std::string& path) {
    size_t i = path.size();
    while (i > 0 && !is_sep(path[i - 1])) --i;
    if (i == 0) return "";
    if (i == 1) return path.substr(0, 1);
#ifdef _WIN32
    if (i == 3 && path[1] == ':') return path.substr(0, 3);
#endif
    return path.substr(0, i - 1);
}

std::string path_ext(const std::string& path) {
    for (size_t i = path.size(); i > 0; --i) {
        char c = path[i - 1];
        if (is_sep(c)) break;
        if (c == '.') return path.substr(i - 1);
    }
    return "";
}

}  // namespace fpg
