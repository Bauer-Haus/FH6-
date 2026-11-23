#include "geometry_json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "platform.hpp"

namespace fpg {

// Shortest decimal digits that read back as v, like strconv's -1 precision.
static void shortest_digits(double v, std::string& digits, int& exp10) {
    char buf[64];
    for (int prec = 0; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*e", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    // buf is "d.ddde[+-]xx" (v is positive here)
    const char* e = std::strchr(buf, 'e');
    exp10 = std::atoi(e + 1);
    digits.clear();
    for (const char* p = buf; p < e; ++p)
        if (*p >= '0' && *p <= '9') digits.push_back(*p);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
}

std::string format_go_float(double v) {
    if (v == 0) return std::signbit(v) ? "-0" : "0";
    if (std::isnan(v) || std::isinf(v)) return "0";  // Go refuses these; never produced here
    std::string out;
    if (v < 0) {
        out.push_back('-');
        v = -v;
    }
    std::string d;
    int e;
    shortest_digits(v, d, e);
    if (v < 1e-6 || v >= 1e21) {
        // 'e' format
        out += d.substr(0, 1);
        if (d.size() > 1) out += "." + d.substr(1);
        char eb[16];
        // strconv writes at least two exponent digits; encoding/json then
        // shortens e-07 to e-7.
        if (e < 0 && e > -10) std::snprintf(eb, sizeof eb, "e-%d", -e);
        else std::snprintf(eb, sizeof eb, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
        out += eb;
        return out;
    }
    // 'f' format: the digits d.ddd x 10^e as plain decimal.
    const int n = (int)d.size();
    if (e >= 0) {
        if (n <= e + 1) {
            out += d + std::string((size_t)(e + 1 - n), '0');
        } else {
            out += d.substr(0, (size_t)e + 1) + "." + d.substr((size_t)e + 1);
        }
    } else {
        out += "0." + std::string((size_t)(-e - 1), '0') + d;
    }
    return out;
}

std::string geometry_to_json(const std::vector<ShapeOut>& shapes) {
    std::string s = "{\n  \"shapes\": [";
    if (shapes.empty()) {
        s += "]\n}";
        return s;
    }
    s += "\n";
    for (size_t i = 0; i < shapes.size(); ++i) {
        const ShapeOut& sh = shapes[i];
        s += "    {\n";
        s += "      \"type\": " + std::to_string(sh.type) + ",\n";
        s += "      \"data\": [";
        if (sh.data.empty()) {
            s += "],\n";
        } else {
            s += "\n";
            for (size_t k = 0; k < sh.data.size(); ++k) {
                s += "        " + format_go_float(sh.data[k]);
                s += (k + 1 < sh.data.size()) ? ",\n" : "\n";
            }
            s += "      ],\n";
        }
        s += "      \"color\": [\n";
        for (int k = 0; k < 4; ++k) {
            s += "        " + std::to_string(sh.color[k]);
            s += (k < 3) ? ",\n" : "\n";
        }
        s += "      ],\n";
        s += "      \"score\": " + format_go_float(sh.score) + "\n";
        s += (i + 1 < shapes.size()) ? "    },\n" : "    }\n";
    }
    s += "  ]\n}";
    return s;
}

bool save_geometry(const std::string& path, const std::vector<ShapeOut>& shapes) {
    std::string dir = dir_name(path);
    if (!dir.empty() && !make_dirs(dir)) return false;
    return write_file_atomic(path, geometry_to_json(shapes));
}

// ------------------------------------------------------------ a small reader

namespace {

struct Reader {
    const std::string& s;
    size_t i = 0;
    std::string err;

    explicit Reader(const std::string& text) : s(text) {}

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    bool eat(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }
    bool fail(const char* why) {
        if (err.empty()) err = std::string(why) + " at byte " + std::to_string(i);
        return false;
    }
    bool string(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) break;
                char c = s[i];
                if (c == 'u') {
                    i += 4;  // keys we care about are ASCII
                    out.push_back('?');
                } else {
                    out.push_back(c == 'n' ? '\n' : c == 't' ? '\t' : c);
                }
                ++i;
            } else {
                out.push_back(s[i++]);
            }
        }
        if (i >= s.size()) return fail("unterminated string");
        ++i;
        return true;
    }
    bool number(double& out) {
        ws();
        const char* start = s.c_str() + i;
        char* end = nullptr;
        out = std::strtod(start, &end);
        if (end == start) return fail("expected number");
        i += (size_t)(end - start);
        return true;
    }
    bool skip() {  // any value
        ws();
        if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '"') {
            std::string t;
            return string(t);
        }
        if (c == '{') {
            ++i;
            if (eat('}')) return true;
            do {
                std::string k;
                if (!string(k) || !eat(':') || !skip()) return false;
            } while (eat(','));
            return eat('}') || fail("expected }");
        }
        if (c == '[') {
            ++i;
            if (eat(']')) return true;
            do {
                if (!skip()) return false;
            } while (eat(','));
            return eat(']') || fail("expected ]");
        }
        if (s.compare(i, 4, "true") == 0 || s.compare(i, 4, "null") == 0) {
            i += 4;
            return true;
        }
        if (s.compare(i, 5, "false") == 0) {
            i += 5;
            return true;
        }
        double d;
        return number(d);
    }
    bool numbers(std::vector<double>& out) {
        out.clear();
        if (!eat('[')) return fail("expected [");
        if (eat(']')) return true;
        do {
            double d;
            if (!number(d)) return false;
            out.push_back(d);
        } while (eat(','));
        return eat(']') || fail("expected ]");
    }
    bool shape(ShapeOut& sh) {
        if (!eat('{')) return fail("expected {");
        sh = ShapeOut();
        if (eat('}')) return true;
        do {
            std::string k;
            if (!string(k) || !eat(':')) return false;
            if (k == "type") {
                double d;
                if (!number(d)) return false;
                sh.type = (int)d;
            } else if (k == "data") {
                if (!numbers(sh.data)) return false;
            } else if (k == "color") {
                std::vector<double> c;
                if (!numbers(c)) return false;
                sh.colorCount = (int)c.size();
                for (size_t n = 0; n < 4; ++n) sh.color[n] = n < c.size() ? (int)c[n] : 0;
            } else if (k == "score") {
                if (!number(sh.score)) return false;
            } else if (!skip()) {
                return false;
            }
        } while (eat(','));
        return eat('}') || fail("expected }");
    }
};

}  // namespace

bool parse_geometry(const std::string& text, std::vector<ShapeOut>& shapes, std::string& err) {
    Reader r(text);
    shapes.clear();
    if (!r.eat('{')) {
        err = "parse checkpoint json: expected {";
        return false;
    }
    if (!r.eat('}')) {
        do {
            std::string k;
            if (!r.string(k) || !r.eat(':')) break;
            if (k == "shapes") {
                if (!r.eat('[')) {
                    r.fail("expected [");
                    break;
                }
                if (r.eat(']')) continue;
                do {
                    ShapeOut sh;
                    if (!r.shape(sh)) break;
                    shapes.push_back(sh);
                } while (r.eat(','));
                if (!r.err.empty() || !r.eat(']')) {
                    r.fail("expected ]");
                    break;
                }
            } else if (!r.skip()) {
                break;
            }
        } while (r.eat(','));
    }
    if (!r.err.empty()) {
        err = "parse checkpoint json: " + r.err;
        return false;
    }
    return true;
}

bool load_geometry(const std::string& path, std::vector<ShapeOut>& shapes, std::string& err) {
    std::string text;
    if (!read_file(path, text)) {
        err = "read checkpoint: cannot read " + path;
        return false;
    }
    return parse_geometry(text, shapes, err);
}

std::string build_output_path(const std::string& base, int shapeCount) {
    std::string ext = path_ext(base);
    return base.substr(0, base.size() - ext.size()) + "." + std::to_string(shapeCount) + ".json";
}

std::string build_final_output_path(const std::string& base) {
    std::string ext = path_ext(base);
    return base.substr(0, base.size() - ext.size()) + ".json";
}

}  // namespace fpg
