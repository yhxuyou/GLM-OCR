#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include "glmocr/common.h"
#include "glmocr/config.h"
#include "glmocr/page_loader.h"
#include "glmocr/ocr_client.h"
#include "glmocr/result_formatter.h"
#include "glmocr/layout_detector.h"

namespace glmocr {

class Pipeline {
public:
    Pipeline(const PipelineConfig& config);
    ~Pipeline();

    bool start();
    void stop();

    PipelineResult process(const std::string& file_path);
    std::vector<PipelineResult> process_batch(const std::vector<std::string>& file_paths);
    PipelineResult process_image(const Image& image);
    PipelineResult process_ocr_only(const std::string& file_path);

private:
    PipelineConfig config_;
    std::unique_ptr<PageLoader> page_loader_;
    std::unique_ptr<OCRClient> ocr_client_;
    std::unique_ptr<ResultFormatter> formatter_;
    std::unique_ptr<LayoutDetector> layout_detector_;
    bool started_ = false;

    std::string process_page_ocr_only(const Image& image);
    Image crop_region(const Image& image, const BBox& bbox);
};

class GlmOcr {
public:
    GlmOcr(const GlmOcrConfig& config = GlmOcrConfig());
    ~GlmOcr();

    PipelineResult parse(const std::string& file_path);
    std::vector<PipelineResult> parse(const std::vector<std::string>& file_paths);

private:
    GlmOcrConfig config_;
    std::unique_ptr<Pipeline> pipeline_;
};

} // namespace glmocr
