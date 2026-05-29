#include "glmocr/pipeline.h"
#include "glmocr/utils/image_utils.h"
#include "glmocr/utils/string_utils.h"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace glmocr {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config) {

    page_loader_ = std::make_unique<PageLoader>(config.page_loader);
    ocr_client_ = std::make_unique<OCRClient>(config.ocr_api);
    formatter_ = std::make_unique<ResultFormatter>(config.result_formatter);
    layout_detector_ = std::make_unique<LayoutDetector>(config.layout);

    if (config.layout.model_dir) {
        std::string model_path = *config.layout.model_dir;
        if (!model_path.empty()) {
            if (!layout_detector_->load_model(model_path)) {
                std::cerr << "[Pipeline] Warning: Failed to load layout model from "
                          << model_path << ", will use OCR-only mode" << std::endl;
            } else {
                std::cout << "[Pipeline] Layout detector loaded successfully" << std::endl;
            }
        }
    }
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

Image Pipeline::crop_region(const Image& image, const BBox& bbox) {
    int x1 = static_cast<int>(bbox.x1 * image.width / 1000);
    int y1 = static_cast<int>(bbox.y1 * image.height / 1000);
    int x2 = static_cast<int>(bbox.x2 * image.width / 1000);
    int y2 = static_cast<int>(bbox.y2 * image.height / 1000);

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(image.width - 1, x2);
    y2 = std::min(image.height - 1, y2);

    int width = x2 - x1;
    int height = y2 - y1;

    if (width <= 0 || height <= 0) {
        return Image();
    }

    return utils::crop_image(image, x1, y1, width, height);
}

std::string Pipeline::process_page_ocr_only(const Image& image) {
    auto request = page_loader_->build_request_from_image(image, "text");
    OCRResponse response = ocr_client_->process(request);

    if (!response.success) {
        std::cerr << "[Pipeline] OCR request failed: " << response.error_message << std::endl;
        return "";
    }

    return response.content;
}

PipelineResult Pipeline::process_ocr_only(const std::string& file_path) {
    if (!started_ && !start()) {
        throw std::runtime_error("Failed to start pipeline");
    }

    auto pages = page_loader_->load_pages(file_path);

    if (pages.empty()) {
        return PipelineResult();
    }

    std::vector<std::vector<Region>> grouped_regions;
    for (size_t i = 0; i < pages.size(); ++i) {
        auto content = process_page_ocr_only(pages[i]);

        std::vector<Region> page_regions;
        Region r;
        r.index = 0;
        r.label = "text";
        r.content = content;
        page_regions.push_back(r);
        grouped_regions.push_back(page_regions);
    }

    PipelineResult result;
    auto formatted = formatter_->process(grouped_regions);
    result.json_result = formatted.json_output;
    result.markdown_output = formatted.markdown_output;

    return result;
}

PipelineResult Pipeline::process_image(const Image& image) {
    if (!started_ && !start()) {
        throw std::runtime_error("Failed to start pipeline");
    }

    PipelineResult result;

    if (layout_detector_->is_loaded()) {
        // Two-stage pipeline: layout detection -> parallel OCR
        std::cout << "[Pipeline] Running layout detection..." << std::endl;
        auto regions = layout_detector_->detect_regions(image);

        if (regions.empty()) {
            auto content = process_page_ocr_only(image);
            std::vector<Region> page_regions;
            Region r;
            r.index = 0;
            r.label = "text";
            r.content = content;
            page_regions.push_back(r);

            auto formatted = formatter_->process({page_regions});
            result.json_result = formatted.json_output;
            result.markdown_output = formatted.markdown_output;
            return result;
        }

        std::cout << "[Pipeline] Detected " << regions.size()
                  << " regions, running parallel OCR..." << std::endl;

        // Build OCR requests for each region
        std::vector<nlohmann::json> ocr_requests;
        std::vector<Image> cropped_images;

        for (size_t i = 0; i < regions.size(); ++i) {
            auto& region = regions[i];
            Image cropped = crop_region(image, region.bbox);
            cropped_images.push_back(cropped);

            std::string task_type = region.native_label;
            if (region.label == "image" || region.label == "chart") {
                task_type = "image";
            }

            auto request = page_loader_->build_request_from_image(
                cropped.empty() ? image : cropped, task_type);
            ocr_requests.push_back(request);
        }

        // Parallel OCR
        auto ocr_responses = ocr_client_->process_batch(
            ocr_requests, config_.max_workers);

        // Fill region content
        for (size_t i = 0; i < regions.size() && i < ocr_responses.size(); ++i) {
            if (ocr_responses[i].success) {
                regions[i].content = ocr_responses[i].content;
            }
            regions[i].is_image = (regions[i].label == "image" ||
                                   regions[i].label == "chart");
        }

        auto formatted = formatter_->process({regions});
        result.json_result = formatted.json_output;
        result.markdown_output = formatted.markdown_output;

    } else {
        // Fallback: OCR-only mode
        auto content = process_page_ocr_only(image);

        std::vector<Region> page_regions;
        Region r;
        r.index = 0;
        r.label = "text";
        r.content = content;
        page_regions.push_back(r);

        auto formatted = formatter_->process({page_regions});
        result.json_result = formatted.json_output;
        result.markdown_output = formatted.markdown_output;
    }

    return result;
}

PipelineResult Pipeline::process(const std::string& file_path) {
    if (!started_ && !start()) {
        throw std::runtime_error("Failed to start pipeline");
    }

    auto pages = page_loader_->load_pages(file_path);

    if (pages.empty()) {
        return PipelineResult();
    }

    PipelineResult result;
    std::vector<std::vector<Region>> all_page_regions;

    for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
        const auto& page_image = pages[page_idx];

        if (layout_detector_->is_loaded()) {
            // Two-stage: layout detection -> parallel OCR per page
            std::cout << "[Pipeline] Page " << page_idx
                      << ": Running layout detection..." << std::endl;

            auto regions = layout_detector_->detect_regions(page_image);

            if (regions.empty()) {
                auto content = process_page_ocr_only(page_image);
                Region r;
                r.index = 0;
                r.label = "text";
                r.content = content;
                all_page_regions.push_back({r});
                continue;
            }

            std::cout << "[Pipeline] Page " << page_idx << ": Detected "
                      << regions.size() << " regions" << std::endl;

            // Build OCR requests
            std::vector<nlohmann::json> ocr_requests;
            std::vector<Image> cropped_images;

            for (auto& region : regions) {
                Image cropped = crop_region(page_image, region.bbox);
                cropped_images.push_back(cropped);

                std::string task_type = region.native_label;
                if (region.label == "image" || region.label == "chart") {
                    task_type = "image";
                }

                auto request = page_loader_->build_request_from_image(
                    cropped.empty() ? page_image : cropped, task_type);
                ocr_requests.push_back(request);
            }

            // Parallel OCR for all regions on this page
            auto ocr_responses = ocr_client_->process_batch(
                ocr_requests, config_.max_workers);

            // Fill content
            for (size_t i = 0; i < regions.size() && i < ocr_responses.size(); ++i) {
                if (ocr_responses[i].success) {
                    regions[i].content = ocr_responses[i].content;
                }
                regions[i].is_image = (regions[i].label == "image" ||
                                       regions[i].label == "chart");
            }

            all_page_regions.push_back(regions);
        } else {
            // OCR-only fallback
            auto content = process_page_ocr_only(page_image);
            Region r;
            r.index = 0;
            r.label = "text";
            r.content = content;
            all_page_regions.push_back({r});
        }
    }

    auto formatted = formatter_->process(all_page_regions);
    result.json_result = formatted.json_output;
    result.markdown_output = formatted.markdown_output;

    return result;
}

std::vector<PipelineResult> Pipeline::process_batch(const std::vector<std::string>& file_paths) {
    std::vector<PipelineResult> results;
    for (const auto& path : file_paths) {
        results.push_back(process(path));
    }
    return results;
}

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
