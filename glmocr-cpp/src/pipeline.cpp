#include "glmocr/pipeline.h"
#include "glmocr/utils/image_utils.h"
#include "glmocr/utils/string_utils.h"
#include <algorithm>
#include <chrono>

namespace glmocr {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config) {
    
    page_loader_ = std::make_unique<PageLoader>(config.page_loader);
    ocr_client_ = std::make_unique<OCRClient>(config.ocr_api);
    formatter_ = std::make_unique<ResultFormatter>(config.result_formatter);
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    if (started_) return true;
    
    if (!ocr_client_->start()) {
        return false;
    }
    
    started_ = true;
    return true;
}

void Pipeline::stop() {
    started_ = false;
    ocr_client_->stop();
}

std::vector<Region> Pipeline::detect_layout(const Image& image) {
    // Placeholder implementation - returns a single region covering the whole image
    // In a real implementation, you would use PP-DocLayout-V3 model here
    
    std::vector<Region> regions;
    
    Region whole_page;
    whole_page.index = 0;
    whole_page.label = "text";
    whole_page.native_label = "text";
    whole_page.score = 1.0f;
    whole_page.bbox = BBox(0, 0, 1000, 1000); // normalized
    whole_page.task_type = "text";
    
    regions.push_back(whole_page);
    
    return regions;
}

Image Pipeline::crop_region(const Image& image, const BBox& bbox) {
    // Convert normalized coordinates to pixel coordinates
    int x1 = static_cast<int>(bbox.x1 * image.width / 1000);
    int y1 = static_cast<int>(bbox.y1 * image.height / 1000);
    int x2 = static_cast<int>(bbox.x2 * image.width / 1000);
    int y2 = static_cast<int>(bbox.y2 * image.height / 1000);
    
    // Clamp to image bounds
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(image.width - 1, x2);
    y2 = std::min(image.height - 1, y2);
    
    int width = x2 - x1;
    int height = y2 - y1;
    
    if (width <= 0 || height <= 0) {
        return Image(); // Return empty image
    }
    
    return utils::crop_image(image, x1, y1, width, height);
}

std::string Pipeline::process_page_ocr_only(const Image& image) {
    // Build request
    auto request = page_loader_->build_request_from_image(image, "text");
    
    // Call OCR
    OCRResponse response = ocr_client_->process(request);
    
    if (!response.success) {
        return "";
    }
    
    return response.content;
}

PipelineResult Pipeline::process_ocr_only(const std::string& file_path) {
    if (!started_ && !start()) {
        throw std::runtime_error("Failed to start pipeline");
    }
    
    // Load pages
    auto pages = page_loader_->load_pages(file_path);
    
    if (pages.empty()) {
        return PipelineResult();
    }
    
    // Process each page
    std::vector<std::string> page_contents;
    for (const auto& page : pages) {
        auto content = process_page_ocr_only(page);
        page_contents.push_back(content);
    }
    
    // Format results
    PipelineResult result;
    
    // Build per-page regions
    std::vector<std::vector<Region>> grouped_regions;
    for (size_t i = 0; i < page_contents.size(); ++i) {
        std::vector<Region> page_regions;
        Region r;
        r.index = 0;
        r.label = "text";
        r.content = page_contents[i];
        page_regions.push_back(r);
        grouped_regions.push_back(page_regions);
    }
    
    // Format
    auto formatted = formatter_->process(grouped_regions);
    result.json_result = formatted.json_output;
    result.markdown_output = formatted.markdown_output;
    
    return result;
}

PipelineResult Pipeline::process_image(const Image& image) {
    if (!started_ && !start()) {
        throw std::runtime_error("Failed to start pipeline");
    }
    
    // For now, use OCR-only mode (layout detection is placeholder)
    PipelineResult result;
    
    auto content = process_page_ocr_only(image);
    
    std::vector<Region> regions;
    Region r;
    r.index = 0;
    r.label = "text";
    r.content = content;
    regions.push_back(r);
    
    auto formatted = formatter_->process({regions});
    result.json_result = formatted.json_output;
    result.markdown_output = formatted.markdown_output;
    
    return result;
}

PipelineResult Pipeline::process(const std::string& file_path) {
    // Currently, we use OCR-only mode as placeholder for layout detection
    return process_ocr_only(file_path);
}

std::vector<PipelineResult> Pipeline::process_batch(const std::vector<std::string>& file_paths) {
    std::vector<PipelineResult> results;
    for (const auto& path : file_paths) {
        results.push_back(process(path));
    }
    return results;
}

// GlmOcr implementation
GlmOcr::GlmOcr(const GlmOcrConfig& config)
    : config_(config) {
    pipeline_ = std::make_unique<Pipeline>(config.pipeline);
    pipeline_->start();
}

GlmOcr::~GlmOcr() {
    pipeline_->stop();
}

PipelineResult GlmOcr::parse(const std::string& file_path) {
    return pipeline_->process(file_path);
}

std::vector<PipelineResult> GlmOcr::parse(const std::vector<std::string>& file_paths) {
    return pipeline_->process_batch(file_paths);
}

} // namespace glmocr
