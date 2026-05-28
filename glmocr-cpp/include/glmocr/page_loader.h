#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "glmocr/common.h"
#include "glmocr/config.h"

namespace glmocr {

class PageLoader {
public:
    PageLoader(const PageLoaderConfig& config);
    ~PageLoader() = default;
    
    // Load a single image or PDF and return all pages as Image objects
    std::vector<Image> load_pages(const std::string& filepath);
    
    // Load multiple sources
    std::vector<Image> load_pages(const std::vector<std::string>& filepaths);
    
    // Build an API request from an image
    nlohmann::json build_request_from_image(
        const Image& image,
        const std::string& task_type = "text");
    
    // Build a request from existing messages (for pass-through mode)
    nlohmann::json build_request(const nlohmann::json& request_data);
    
    // Encode image to data URI format
    std::string image_to_data_uri(const Image& image);
    
private:
    PageLoaderConfig config_;
    
    // Load image and prepare for API (resize to max pixels)
    Image prepare_image_for_api(const Image& image);
};

} // namespace glmocr
