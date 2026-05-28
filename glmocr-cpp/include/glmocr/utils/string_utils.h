#pragma once

#include <string>
#include <vector>
#include <regex>

namespace glmocr {
namespace utils {

// String trimming
std::string ltrim(const std::string& s);
std::string rtrim(const std::string& s);
std::string trim(const std::string& s);

// String splitting
std::vector<std::string> split(const std::string& s, char delimiter);
std::vector<std::string> split(const std::string& s, const std::string& delimiter);

// String replacement
std::string replace_all(const std::string& str, const std::string& from, const std::string& to);

// Check if string starts/ends with
bool starts_with(const std::string& str, const std::string& prefix);
bool ends_with(const std::string& str, const std::string& suffix);

// Regex-based cleaning
std::string clean_repeated_punctuation(const std::string& s);
std::string normalize_inline_formula(const std::string& s);

// Convert to lowercase
std::string to_lower(const std::string& s);

// Join strings with delimiter
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);

} // namespace utils
} // namespace glmocr
