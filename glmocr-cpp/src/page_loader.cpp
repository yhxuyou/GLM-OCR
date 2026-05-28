#include "glmocr/page_loader.h"
#include "glmocr/utils/image_utils.h"
#include "glmocr/utils/string_utils.h"
#include "glmocr/utils/base64.h"

namespace glmocr {

using json = nlohmann::json;

PageLoader::PageLoader(const PageLoaderConfig& config)
    : config_(config) {
}

std::vector<Image> PageLoader::load_pages(const std::string& filepath) {
    std::vector<Image> result;
    
    if (utils::is_pdf_file(filepath)) {
        // Load PDF
        result = utils::pdf_to_images(filepath, config_.pdf_dpi, 
            config_.pdf_max_pages.value_or(-1));
    } else {
        // Load as single image
        Image img = utils::load_image(filepath);
        if (!img.data.empty()) {
            result.push_back(img);
        }
    }
    
    return result;
}

std::vector<Image> PageLoader::load_pages(const std::vector<std::string>& filepaths) {
    std::vector<Image> result;
    for (const auto& filepath : filepaths) {
        auto pages = load_pages(filepath);
        result.insert(result.end(), pages.begin(), pages.end());
    }
    return result;
}

Image PageLoader::prepare_image_for_api(const Image& image) {
    if (image.data.empty()) return image;
    
    int64_t num_pixels = static_cast<int64_t>(image.width) * image.height;
    if (num_pixels <= config_.max_pixels) {
        return image;
    }
    
    // Calculate scaling factor to stay under max_pixels
    double scale = std::sqrt(static_cast<double>(config_.max_pixels) / num_pixels);
    int new_width = static_cast<int>(image.width * scale);
    int new_height = static_cast<int>(image.height * scale);
    
    return utils::resize_image(image, new_width, new_height);
}

std::string PageLoader::image_to_data_uri(const Image& image) {
    std::string base64_data = utils::encode_image_to_base64(image, config_.image_format);
    std::string format_lower = utils::to_lower(config_.image_format);
    if (format_lower == "jpeg") format_lower = "jpg";
    return "data:image/" + format_lower + ";base64," + base64_data;
}

json PageLoader::build_request_from_image(const Image& image, const std::string& task_type) {
    json request;
    
    // Build prompt
    std::string prompt = "";
    auto it = config_.task_prompt_mapping.find(task_type);
    if (it != config_.task_prompt_mapping.end()) {
        prompt = it->second;
    }
    
    // Prepare image
    Image prepared = prepare_image_for_api(image);
    std::string data_uri = image_to_data_uri(prepared);
    
    // Build messages
    request["messages"] = json::array({
        {
            {"role", "user"},
            {"content", json::array({
                {{"type", "image_url"}, {"image_url", {{"url", data_uri}}}}
            })}
        }
    });
    
    if (!prompt.empty()) {
        request["messages"][0]["content"].push_back({
            {"type", "text"},
            {"text", prompt}
        });
    }
    
    // Add generation parameters
    request["max_tokens"] = config_.max_tokens;
    request["temperature"] = config_.temperature;
    request["top_p"] = config_.top_p;
    request["top_k"] = config_.top_k;
    request["repetition_penalty"] = config_.repetition_penalty;
    
    return request;
}

json PageLoader::build_request(const json& request_data) {
    json result = request_data;
    
    // Set default parameters if not present
    if (!result.contains("max_tokens")) {
        result["max_tokens"] = config_.max_tokens;
    }
    if (!result.contains("temperature")) {
        result["temperature"] = config_.temperature;
    }
    if (!result.contains("top_p")) {
        result["top_p"] = config_.top_p;
    }
    if (!result.contains("top_k")) {
        result["top_k"] = config_.top_k;
    }
    if (!result.contains("repetition_penalty")) {
        result["repetition_penalty"] = config_.repetition_penalty;
    }
    
    return result;
}

} // namespace glmocr
