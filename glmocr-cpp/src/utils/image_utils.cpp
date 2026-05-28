#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "glmocr/utils/image_utils.h"
#include "glmocr/utils/base64.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <fstream>
#include <cstring>

namespace glmocr {
namespace utils {

Image load_image(const std::string& filepath) {
    Image img;
    int width, height, channels;
    uint8_t* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if (data) {
        img.width = width;
        img.height = height;
        img.channels = channels;
        size_t data_size = width * height * channels;
        img.data.resize(data_size);
        std::memcpy(img.data.data(), data, data_size);
        stbi_image_free(data);
    }
    return img;
}

Image load_image_from_memory(const std::vector<uint8_t>& data) {
    Image img;
    int width, height, channels;
    uint8_t* img_data = stbi_load_from_memory(data.data(), data.size(), &width, &height, &channels, 0);
    if (img_data) {
        img.width = width;
        img.height = height;
        img.channels = channels;
        size_t data_size = width * height * channels;
        img.data.resize(data_size);
        std::memcpy(img.data.data(), img_data, data_size);
        stbi_image_free(img_data);
    }
    return img;
}

std::string encode_image_to_base64(const Image& image, const std::string& format) {
    if (image.data.empty()) return "";
    
    std::vector<uint8_t> encoded;
    auto write_callback = [](void* context, void* data, int size) {
        auto* vec = static_cast<std::vector<uint8_t>*>(context);
        vec->insert(vec->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
    };
    
    bool success = false;
    if (format == "JPEG" || format == "jpeg" || format == "jpg") {
        success = stbi_write_jpg_to_func(write_callback, &encoded, 
            image.width, image.height, image.channels, image.data.data(), 90) != 0;
    } else if (format == "PNG" || format == "png") {
        success = stbi_write_png_to_func(write_callback, &encoded,
            image.width, image.height, image.channels, image.data.data(), image.width * image.channels) != 0;
    }
    
    if (success) {
        return base64_encode(encoded);
    }
    return "";
}

Image resize_image(const Image& image, int new_width, int new_height) {
    // Simple nearest neighbor resize (placeholder - real implementation would use better algorithm)
    Image resized;
    resized.width = new_width;
    resized.height = new_height;
    resized.channels = image.channels;
    resized.data.resize(new_width * new_height * image.channels);
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = static_cast<int>(static_cast<float>(x) / new_width * image.width);
            int src_y = static_cast<int>(static_cast<float>(y) / new_height * image.height);
            src_x = std::min(src_x, image.width - 1);
            src_y = std::min(src_y, image.height - 1);
            
            for (int c = 0; c < image.channels; c++) {
                int dst_idx = (y * new_width + x) * image.channels + c;
                int src_idx = (src_y * image.width + src_x) * image.channels + c;
                resized.data[dst_idx] = image.data[src_idx];
            }
        }
    }
    
    return resized;
}

Image crop_image(const Image& image, int x, int y, int width, int height) {
    Image cropped;
    cropped.width = width;
    cropped.height = height;
    cropped.channels = image.channels;
    cropped.data.resize(width * height * image.channels);
    
    // Clamp coordinates
    x = std::max(0, std::min(x, image.width - 1));
    y = std::max(0, std::min(y, image.height - 1));
    width = std::min(width, image.width - x);
    height = std::min(height, image.height - y);
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            for (int c = 0; c < image.channels; c++) {
                int dst_idx = (j * width + i) * image.channels + c;
                int src_idx = ((y + j) * image.width + (x + i)) * image.channels + c;
                cropped.data[dst_idx] = image.data[src_idx];
            }
        }
    }
    
    return cropped;
}

bool is_pdf_file(const std::string& filepath) {
    // Check if file extension is .pdf
    std::string lower_ext;
    size_t dot_pos = filepath.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = filepath.substr(dot_pos);
        for (char c : ext) {
            lower_ext += std::tolower(c);
        }
    }
    if (lower_ext == ".pdf") return true;
    
    // Also check magic number
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    
    char magic[4];
    file.read(magic, 4);
    return std::strncmp(magic, "%PDF", 4) == 0;
}

std::vector<Image> pdf_to_images(const std::string& filepath, int dpi, int max_pages) {
    // Placeholder implementation - real PDF support would require:
    // - PoDoFo, MuPDF, or similar library
    // - Or call an external tool like pdftoppm
    // For now, return empty vector
    
    // Future implementation note:
    // You could use popen() to call pdftoppm:
    // pdftoppm -png -r <dpi> <pdf> <output_prefix>
    // Then load the generated PNG files
    
    return {};
}

} // namespace utils
} // namespace glmocr
