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

namespace glmocr {

class Pipeline {
public:
    Pipeline(const PipelineConfig& config);
    ~Pipeline();
    
    // Start the pipeline (optional - will start automatically if needed)
    bool start();
    
    // Stop the pipeline
    void stop();
    
    // Process a single image or PDF file
    PipelineResult process(const std::string& file_path);
    
    // Process multiple files
    std::vector<PipelineResult> process_batch(const std::vector<std::string>& file_paths);
    
    // Process raw image data
    PipelineResult process_image(const Image& image);
    
    // Process in OCR-only mode (no layout analysis)
    PipelineResult process_ocr_only(const std::string& file_path);
    
private:
    PipelineConfig config_;
    std::unique_ptr<PageLoader> page_loader_;
    std::unique_ptr<OCRClient> ocr_client_;
    std::unique_ptr<ResultFormatter> formatter_;
    bool started_ = false;
    
    // Page queue item
    struct PageTask {
        int file_idx;
        int page_idx;
        Image image;
    };
    
    // Region task
    struct RegionTask {
        int file_idx;
        int page_idx;
        int region_idx;
        Region region;
        Image cropped_image;
    };
    
    // Result accumulator
    struct FileResult {
        int file_idx;
        std::string file_path;
        std::vector<std::vector<Region>> page_results;
    };
    
    // Process a single page in OCR-only mode
    std::string process_page_ocr_only(const Image& image);
    
    // Simulate layout detection (placeholder - can be replaced with actual implementation)
    std::vector<Region> detect_layout(const Image& image);
    
    // Crop image to region
    Image crop_region(const Image& image, const BBox& bbox);
};

// Convenience class with simpler API
class GlmOcr {
public:
    GlmOcr(const GlmOcrConfig& config = GlmOcrConfig());
    ~GlmOcr();
    
    // Process a file and return result
    PipelineResult parse(const std::string& file_path);
    
    // Process multiple files
    std::vector<PipelineResult> parse(const std::vector<std::string>& file_paths);
    
private:
    GlmOcrConfig config_;
    std::unique_ptr<Pipeline> pipeline_;
};

} // namespace glmocr
