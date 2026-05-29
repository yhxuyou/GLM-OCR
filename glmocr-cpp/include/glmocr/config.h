#pragma once

#include <string>
#include <map>
#include <vector>
#include <optional>
#include "glmocr/common.h"

namespace glmocr {

struct MaasConfig {
    bool enabled = false;
    std::string api_url = "https://open.bigmodel.cn/api/paas/v4/layout_parsing";
    std::string model = "glm-ocr";
    std::optional<std::string> api_key;
    bool verify_ssl = true;
    int connect_timeout = 30;
    int request_timeout = 300;
    int retry_max_attempts = 2;
    float retry_backoff_base_seconds = 0.5f;
    float retry_backoff_max_seconds = 8.0f;
    float retry_jitter_ratio = 0.2f;
    std::vector<int> retry_status_codes = {429, 500, 502, 503, 504};
    int connection_pool_size = 16;
};

struct OCRApiConfig {
    std::string api_host = "127.0.0.1";
    int api_port = 8080;
    std::optional<std::string> api_scheme;
    std::string api_path = "/v1/chat/completions";
    std::optional<std::string> api_url;
    std::string api_mode = "openai";
    std::string model = "glm-ocr";
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
};

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

    PageLoaderConfig() {
        task_prompt_mapping["text"] = "Text Recognition:";
        task_prompt_mapping["table"] = "Table Recognition:";
        task_prompt_mapping["formula"] = "Formula Recognition:";
        task_prompt_mapping["image"] = "Image Description:";
        task_prompt_mapping["chart"] = "Chart Recognition:";
        task_prompt_mapping["display_formula"] = "Formula Recognition:";
        task_prompt_mapping["inline_formula"] = "Formula Recognition:";
    }
};

struct ResultFormatterConfig {
    bool filter_nested = true;
    float min_overlap_ratio = 0.8f;
    std::string output_format = "both";
    bool enable_merge_formula_numbers = true;
    bool enable_merge_text_blocks = true;
    bool enable_format_bullet_points = true;
    std::map<std::string, std::vector<std::string>> label_visualization_mapping;

    ResultFormatterConfig() {
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
    std::optional<std::vector<float>> layout_unclip_ratio;
    std::string layout_merge_bboxes_mode = "large";
    std::optional<std::map<std::string, std::vector<std::string>>> label_task_mapping;
    bool use_polygon = false;
    std::optional<std::map<std::string, std::string>> id2label;
    bool filter_nested = true;
    float min_overlap_ratio = 0.8f;
};

struct DocDetectorConfig {
    std::optional<std::string> model_path;
    int input_size = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_det = 300;
    int workers = 4;
};

struct OrientationConfig {
    std::optional<std::string> model_path;
    int resize_short = 256;
    int crop_size = 224;
    int batch_size = 3;
    int workers = 4;
};

struct UnDistortConfig {
    std::optional<std::string> model_path;
    int img_width = 488;
    int img_height = 712;
    int grid_w = 45;
    int grid_h = 31;
    int workers = 4;
};

struct PreprocessConfig {
    DocDetectorConfig doc_detector;
    OrientationConfig orientation;
    UnDistortConfig undistort;
    bool enable_doc_detect = false;
    bool enable_orientation = false;
    bool enable_undistort = false;
};

struct PipelineConfig {
    MaasConfig maas;
    PageLoaderConfig page_loader;
    OCRApiConfig ocr_api;
    ResultFormatterConfig result_formatter;
    LayoutConfig layout;
    PreprocessConfig preprocess;
    int max_workers = 16;
    int page_maxsize = 100;
    int region_maxsize = 800;
};

struct LoggingConfig {
    std::string level = "INFO";
    std::optional<std::string> format;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 5002;
    bool debug = false;
};

struct GlmOcrConfig {
    PipelineConfig pipeline;
    ServerConfig server;
    LoggingConfig logging;

    GlmOcrConfig() = default;

    static GlmOcrConfig from_json(const std::string& json_str);
    static GlmOcrConfig from_json_file(const std::string& filepath);
    static GlmOcrConfig from_yaml(const std::string& yaml_str);
    static GlmOcrConfig from_yaml_file(const std::string& filepath);
    static GlmOcrConfig from_file(const std::string& filepath);
    std::string to_json() const;
};

} // namespace glmocr
