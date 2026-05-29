#include "glmocr/pipeline.h"
#include "glmocr/utils/image_utils.h"
#include "glmocr/utils/string_utils.h"
#include <algorithm>
#include <iostream>

namespace glmocr {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config) {

    page_loader_ = std::make_unique<PageLoader>(config.page_loader);
    ocr_client_ = std::make_unique<OCRClient>(config.ocr_api);
    formatter_ = std::make_unique<ResultFormatter>(config.result_formatter);
    layout_detector_ = std::make_unique<LayoutDetector>(config.layout);

    if (config.layout.model_dir && !config.layout.model_dir->empty()) {
        if (layout_detector_->load_model(*config.layout.model_dir)) {
            std::cout << "[Pipeline] Layout detector loaded" << std::endl;
        } else {
            std::cerr << "[Pipeline] Warning: Failed to load layout model" << std::endl;
        }
    }

    // Doc detector
    DocDetector::Config dd_cfg;
    dd_cfg.input_size = config.preprocess.doc_detector.input_size;
    dd_cfg.conf_threshold = config.preprocess.doc_detector.conf_threshold;
    dd_cfg.iou_threshold = config.preprocess.doc_detector.iou_threshold;
    dd_cfg.max_det = config.preprocess.doc_detector.max_det;
    dd_cfg.workers = config.preprocess.doc_detector.workers;
    doc_detector_ = std::make_unique<DocDetector>(dd_cfg);
    if (config.preprocess.enable_doc_detect && config.preprocess.doc_detector.model_path) {
        if (doc_detector_->load_model(*config.preprocess.doc_detector.model_path)) {
            std::cout << "[Pipeline] Doc detector loaded" << std::endl;
        } else {
            std::cerr << "[Pipeline] Warning: Failed to load doc detector model" << std::endl;
        }
    }

    // Orientation detector
    OrientationDetector::Config od_cfg;
    od_cfg.resize_short = config.preprocess.orientation.resize_short;
    od_cfg.crop_size = config.preprocess.orientation.crop_size;
    od_cfg.batch_size = config.preprocess.orientation.batch_size;
    od_cfg.workers = config.preprocess.orientation.workers;
    orientation_detector_ = std::make_unique<OrientationDetector>(od_cfg);
    if (config.preprocess.enable_orientation && config.preprocess.orientation.model_path) {
        if (orientation_detector_->load_model(*config.preprocess.orientation.model_path)) {
            std::cout << "[Pipeline] Orientation detector loaded" << std::endl;
        } else {
            std::cerr << "[Pipeline] Warning: Failed to load orientation model" << std::endl;
        }
    }

    // UnDistorter
    UnDistorter::Config ud_cfg;
    ud_cfg.img_width = config.preprocess.undistort.img_width;
    ud_cfg.img_height = config.preprocess.undistort.img_height;
    ud_cfg.grid_w = config.preprocess.undistort.grid_w;
    ud_cfg.grid_h = config.preprocess.undistort.grid_h;
    ud_cfg.workers = config.preprocess.undistort.workers;
    undistorter_ = std::make_unique<UnDistorter>(ud_cfg);
    if (config.preprocess.enable_undistort && config.preprocess.undistort.model_path) {
        if (undistorter_->load_model(*config.preprocess.undistort.model_path)) {
            std::cout << "[Pipeline] UnDistorter loaded" << std::endl;
        } else {
            std::cerr << "[Pipeline] Warning: Failed to load undistort model" << std::endl;
        }
    }
}

Pipeline::~Pipeline() { stop(); }

bool Pipeline::start() {
    if (started_) return true;
    if (!ocr_client_->start()) return false;
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
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(image.width - 1, x2); y2 = std::min(image.height - 1, y2);
    int w = x2 - x1, h = y2 - y1;
    if (w <= 0 || h <= 0) return Image();
    return utils::crop_image(image, x1, y1, w, h);
}

Image Pipeline::preprocess_image(const Image& image) {
    Image result = image;

    // Step 1: Document detection (crop document region)
    if (doc_detector_->is_loaded()) {
        auto dets = doc_detector_->detect(image);
        if (!dets.empty()) {
            auto& best = dets[0];
            result = doc_detector_->crop_document(image, best);
            std::cout << "[Pipeline] Document detected, cropped to "
                      << result.width << "x" << result.height << std::endl;
        }
    }

    // Step 2: Orientation correction
    if (orientation_detector_->is_loaded()) {
        Orientation orient = orientation_detector_->detect(result);
        if (orient != Orientation::ORIENTATION_0 && orient != Orientation::ORIENTATION_UNKNOWN) {
            result = orientation_detector_->rotate_to_upright(result);
            int deg = static_cast<int>(orient);
            std::cout << "[Pipeline] Orientation corrected: " << deg << " degrees" << std::endl;
        }
    }

    // Step 3: Distortion correction
    if (undistorter_->is_loaded()) {
        result = undistorter_->undistort(result);
        std::cout << "[Pipeline] Distortion corrected" << std::endl;
    }

    return result;
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
    if (!started_ && !start()) throw std::runtime_error("Failed to start pipeline");
    auto pages = page_loader_->load_pages(file_path);
    if (pages.empty()) return PipelineResult();

    std::vector<std::vector<Region>> grouped_regions;
    for (auto& page : pages) {
        auto preprocessed = preprocess_image(page);
        auto content = process_page_ocr_only(preprocessed);
        Region r; r.index = 0; r.label = "text"; r.content = content;
        grouped_regions.push_back({r});
    }

    PipelineResult result;
    auto formatted = formatter_->process(grouped_regions);
    result.json_result = formatted.json_output;
    result.markdown_output = formatted.markdown_output;
    return result;
}

PipelineResult Pipeline::process_image(const Image& image) {
    if (!started_ && !start()) throw std::runtime_error("Failed to start pipeline");

    Image preprocessed = preprocess_image(image);
    PipelineResult result;

    if (layout_detector_->is_loaded()) {
        std::cout << "[Pipeline] Running layout detection..." << std::endl;
        auto regions = layout_detector_->detect_regions(preprocessed);

        if (regions.empty()) {
            auto content = process_page_ocr_only(preprocessed);
            Region r; r.index = 0; r.label = "text"; r.content = content;
            auto formatted = formatter_->process({{r}});
            result.json_result = formatted.json_output;
            result.markdown_output = formatted.markdown_output;
            return result;
        }

        std::cout << "[Pipeline] Detected " << regions.size() << " regions" << std::endl;

        std::vector<nlohmann::json> ocr_requests;
        std::vector<Image> cropped_images;
        for (auto& region : regions) {
            Image cropped = crop_region(preprocessed, region.bbox);
            cropped_images.push_back(cropped);
            std::string task_type = region.native_label;
            if (region.label == "image" || region.label == "chart") task_type = "image";
            ocr_requests.push_back(page_loader_->build_request_from_image(
                cropped.empty() ? preprocessed : cropped, task_type));
        }

        auto ocr_responses = ocr_client_->process_batch(ocr_requests, config_.max_workers);
        for (size_t i = 0; i < regions.size() && i < ocr_responses.size(); ++i) {
            if (ocr_responses[i].success) regions[i].content = ocr_responses[i].content;
            regions[i].is_image = (regions[i].label == "image" || regions[i].label == "chart");
        }

        auto formatted = formatter_->process({regions});
        result.json_result = formatted.json_output;
        result.markdown_output = formatted.markdown_output;
    } else {
        auto content = process_page_ocr_only(preprocessed);
        Region r; r.index = 0; r.label = "text"; r.content = content;
        auto formatted = formatter_->process({{r}});
        result.json_result = formatted.json_output;
        result.markdown_output = formatted.markdown_output;
    }
    return result;
}

PipelineResult Pipeline::process(const std::string& file_path) {
    if (!started_ && !start()) throw std::runtime_error("Failed to start pipeline");
    auto pages = page_loader_->load_pages(file_path);
    if (pages.empty()) return PipelineResult();

    PipelineResult result;
    std::vector<std::vector<Region>> all_page_regions;

    for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
        Image preprocessed = preprocess_image(pages[page_idx]);

        if (layout_detector_->is_loaded()) {
            std::cout << "[Pipeline] Page " << page_idx << ": layout detection..." << std::endl;
            auto regions = layout_detector_->detect_regions(preprocessed);

            if (regions.empty()) {
                auto content = process_page_ocr_only(preprocessed);
                Region r; r.index = 0; r.label = "text"; r.content = content;
                all_page_regions.push_back({r});
                continue;
            }

            std::cout << "[Pipeline] Page " << page_idx << ": " << regions.size() << " regions" << std::endl;

            std::vector<nlohmann::json> ocr_requests;
            for (auto& region : regions) {
                Image cropped = crop_region(preprocessed, region.bbox);
                std::string task_type = region.native_label;
                if (region.label == "image" || region.label == "chart") task_type = "image";
                ocr_requests.push_back(page_loader_->build_request_from_image(
                    cropped.empty() ? preprocessed : cropped, task_type));
            }

            auto ocr_responses = ocr_client_->process_batch(ocr_requests, config_.max_workers);
            for (size_t i = 0; i < regions.size() && i < ocr_responses.size(); ++i) {
                if (ocr_responses[i].success) regions[i].content = ocr_responses[i].content;
                regions[i].is_image = (regions[i].label == "image" || regions[i].label == "chart");
            }
            all_page_regions.push_back(regions);
        } else {
            auto content = process_page_ocr_only(preprocessed);
            Region r; r.index = 0; r.label = "text"; r.content = content;
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
    for (const auto& path : file_paths) results.push_back(process(path));
    return results;
}

GlmOcr::GlmOcr(const GlmOcrConfig& config) : config_(config) {
    pipeline_ = std::make_unique<Pipeline>(config.pipeline);
    pipeline_->start();
}

GlmOcr::~GlmOcr() { pipeline_->stop(); }

PipelineResult GlmOcr::parse(const std::string& file_path) { return pipeline_->process(file_path); }
std::vector<PipelineResult> GlmOcr::parse(const std::vector<std::string>& file_paths) { return pipeline_->process_batch(file_paths); }

} // namespace glmocr
