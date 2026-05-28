#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "glmocr/common.h"

namespace glmocr {
namespace utils {

// Load image from file
Image load_image(const std::string& filepath);

// Load image from memory buffer
Image load_image_from_memory(const std::vector<uint8_t>& data);

// Encode image to base64 string (JPEG format)
std::string encode_image_to_base64(const Image& image, const std::string& format = "JPEG");

// Resize image (simple implementation)
Image resize_image(const Image& image, int new_width, int new_height);

// Crop image region
Image crop_image(const Image& image, int x, int y, int width, int height);

// Check if file is a PDF
bool is_pdf_file(const std::string& filepath);

// Convert PDF to images (placeholder - would need PoDoFo or similar library)
std::vector<Image> pdf_to_images(const std::string& filepath, int dpi = 200, int max_pages = -1);

} // namespace utils
} // namespace glmocr
