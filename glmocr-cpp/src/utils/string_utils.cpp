#include "glmocr/utils/string_utils.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace glmocr {
namespace utils {

std::string ltrim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    return std::string(start, s.end());
}

std::string rtrim(const std::string& s) {
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    return std::string(s.begin(), end);
}

std::string trim(const std::string& s) {
    return rtrim(ltrim(s));
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = s.find(delimiter);
    while (end != std::string::npos) {
        tokens.push_back(s.substr(start, end - start));
        start = end + delimiter.length();
        end = s.find(delimiter, start);
    }
    tokens.push_back(s.substr(start));
    return tokens;
}

std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

bool starts_with(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

std::string clean_repeated_punctuation(const std::string& s) {
    std::string result = s;
    // Replace repeated dots (3+ → 3)
    result = std::regex_replace(result, std::regex(R"((\.)\1{2,})"), "$1$1$1");
    // Replace repeated middle dots (·)
    result = std::regex_replace(result, std::regex(R"((·)\1{2,})"), "$1$1$1");
    // Replace repeated underscores
    result = std::regex_replace(result, std::regex(R"((_)\1{2,})"), "$1$1$1");
    return result;
}

std::string normalize_inline_formula(const std::string& s) {
    std::string result = s;
    // Normalize inline formula markers
    result = std::regex_replace(result, std::regex(R"(\\\()"), "$");
    result = std::regex_replace(result, std::regex(R"(\\\))"), "$");
    return result;
}

std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) return "";
    
    std::string result;
    for (size_t i = 0; i < strings.size(); ++i) {
        if (i > 0) {
            result += delimiter;
        }
        result += strings[i];
    }
    return result;
}

} // namespace utils
} // namespace glmocr
