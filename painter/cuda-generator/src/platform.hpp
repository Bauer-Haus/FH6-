// File and path helpers. Paths are UTF-8 everywhere; on Windows they are
// converted to UTF-16 at the system call, so non-ASCII folder names work.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace fpg {

FILE* fopen_utf8(const std::string& path, const char* mode);
bool read_file(const std::string& path, std::string& out);
// output.WriteFileAtomic: write next to path, then rename into place, so the
// app polling the folder never reads a half-written file. Falls back to a
// direct write if the rename fails.
bool write_file_atomic(const std::string& path, const std::string& data);
bool make_dirs(const std::string& dir);
bool list_files(const std::string& dir, std::vector<std::string>& names);
bool is_dir(const std::string& path);
bool is_abs(const std::string& path);
std::string current_dir();
std::string abs_path(const std::string& path);
std::string path_join(const std::string& a, const std::string& b);
std::string dir_name(const std::string& path);
// filepath.Ext: the suffix from the last '.' in the last element, or "".
std::string path_ext(const std::string& path);
char path_sep();

}  // namespace fpg
