#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace glmocr {
namespace utils {

// Base64 encoding
std::string base64_encode(const std::vector<uint8_t>& data);
std::string base64_encode(const uint8_t* data, size_t len);

// Base64 decoding
std::vector<uint8_t> base64_decode(const std::string& encoded);

} // namespace utils
} // namespace glmocr
