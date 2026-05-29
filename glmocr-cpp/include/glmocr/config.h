#pragma once

#include <string>
#include <map>
#include <vector>
#include <optional>
#include "glmocr/common.h"

namespace glmocr {

// OCR API configuration
struct OCRApiConfig {
    std::string api_host = "localhost";
    int api_port = 8080;
    std::string api_scheme = "http";
    std::string api_path = "/v1/chat/completions";
    std::optional<std::string> api_url;
    std::optional<std::string> api_key;
    std::map<std::string, std::string> headers;
    bool verify_ssl = false;
    int connect_timeout = 30;
    int request_timeout = 120;
    int retry_max_attempts = 2;
    float retry_backoff_base_seconds = 0.5f;
    float retry_backoff_max_seconds = 8.0f;
    float retry_jitter_ratio = 0.2f;
    std::vector<int> retry_status_codes = {429, 500, 502, 503, 504};
    int connection_pool_size = 128;
    std::string api_mode = "openai"; // "openai" or "ollama_generate"
    std::optional<std::string> model;
};

// Page loader configuration
struct PageLoaderConfig {
    int max_tokens = 8192;
    float temperature = 0.0f;
    float top_p = 0.00001f;
    int top_k = 1;
    float repetition_penalty = 1.1f;
    int t_patch_size = 2;
    int patch_expand_factor = 1;
    int image_expect_length = 6144;
    std::string image_format = "JPEG";
    int min_pixels = 112 * 112;
    int max_pixels = 14 * 14 * 4 * 1280;
    std::map<std::string, std::string> task_prompt_mapping;
    int pdf_dpi = 200;
    std::optional<int> pdf_max_pages;
    bool pdf_verbose = false;
};

// Result formatter configuration
struct ResultFormatterConfig {
    bool filter_nested = true;
    float min_overlap_ratio = 0.8f;
    std::string output_format = "both"; // "json", "markdown", "both"
    bool enable_merge_formula_numbers = true;
    bool enable_merge_text_blocks = true;
    bool enable_format_bullet_points = true;
    std::map<std::string, std::vector<std::string>> label_visualization_mapping;
    
    ResultFormatterConfig() {
        // Default label mappings
        label_visualization_mapping["image"] = {"chart", "image"};
        label_visualization_mapping["table"] = {"table"};
        label_visualization_mapping["formula"] = {"display_formula", "inline_formula"};
        label_visualization_mapping["text"] = {
            "abstract", "algorithm", "content", "doc_title",
            "figure_title", "paragraph_title", "reference_content",
            "text", "vertical_text", "vision_footnote", "seal", "formula_number"
        };
    }
};

// Layout detector configuration (placeholder for future implementation)
struct LayoutConfig {
    std::optional<std::string> model_dir;
    float threshold = 0.3f;
    std::optional<std::map<std::string, float>> threshold_by_class;
    int batch_size = 8;
    int workers = 1;
    std::string cuda_visible_devices = "0";
    std::optional<std::string> device;
    std::optional<int> img_size;
    bool layout_nms = true;
    std::optional<std::string> layout_unclip_ratio;
    std::string layout_merge_bboxes_mode = "large";
    std::optional<std::map<std::string, std::vector<std::string>>> label_task_mapping;
    bool use_polygon = false;
    std::optional<std::map<std::string, std::string>> id2label;
    bool filter_nested = true;
    float min_overlap_ratio = 0.8f;
};

// Pipeline configuration
struct PipelineConfig {
    PageLoaderConfig page_loader;
    OCRApiConfig ocr_api;
    ResultFormatterConfig result_formatter;
    LayoutConfig layout;
    int max_workers = 16;
    int page_maxsize = 100;
    int region_maxsize = 800;
};

// Main configuration
struct GlmOcrConfig {
    PipelineConfig pipeline;
    
    // Server settings (optional)
    std::string server_host = "0.0.0.0";
    int server_port = 5002;
    bool server_debug = false;
    
    // Logging
    std::string log_level = "INFO";
    std::optional<std::string> log_format;
    
    GlmOcrConfig() = default;
    
    // Load from JSON string
    static GlmOcrConfig from_json(const std::string& json_str);
    
    // Load from file
    static GlmOcrConfig from_file(const std::string& filepath);
    
    // Save to JSON string
    std::string to_json() const;
};

} // namespace glmocr
