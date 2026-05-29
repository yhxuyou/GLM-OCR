#include "glmocr/config.h"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace glmocr {

using json = nlohmann::json;

// ============================================================================
// MaasConfig
// ============================================================================

void to_json(json& j, const MaasConfig& c) {
    j = json{
        {"enabled", c.enabled},
        {"api_url", c.api_url},
        {"model", c.model},
        {"verify_ssl", c.verify_ssl},
        {"connect_timeout", c.connect_timeout},
        {"request_timeout", c.request_timeout},
        {"retry_max_attempts", c.retry_max_attempts},
        {"retry_backoff_base_seconds", c.retry_backoff_base_seconds},
        {"retry_backoff_max_seconds", c.retry_backoff_max_seconds},
        {"retry_jitter_ratio", c.retry_jitter_ratio},
        {"retry_status_codes", c.retry_status_codes},
        {"connection_pool_size", c.connection_pool_size}
    };
    if (c.api_key) j["api_key"] = *c.api_key;
}

void from_json(const json& j, MaasConfig& c) {
    if (j.contains("enabled")) j.at("enabled").get_to(c.enabled);
    if (j.contains("api_url")) j.at("api_url").get_to(c.api_url);
    if (j.contains("model")) j.at("model").get_to(c.model);
    if (j.contains("verify_ssl")) j.at("verify_ssl").get_to(c.verify_ssl);
    if (j.contains("connect_timeout")) j.at("connect_timeout").get_to(c.connect_timeout);
    if (j.contains("request_timeout")) j.at("request_timeout").get_to(c.request_timeout);
    if (j.contains("retry_max_attempts")) j.at("retry_max_attempts").get_to(c.retry_max_attempts);
    if (j.contains("retry_backoff_base_seconds")) j.at("retry_backoff_base_seconds").get_to(c.retry_backoff_base_seconds);
    if (j.contains("retry_backoff_max_seconds")) j.at("retry_backoff_max_seconds").get_to(c.retry_backoff_max_seconds);
    if (j.contains("retry_jitter_ratio")) j.at("retry_jitter_ratio").get_to(c.retry_jitter_ratio);
    if (j.contains("retry_status_codes")) j.at("retry_status_codes").get_to(c.retry_status_codes);
    if (j.contains("connection_pool_size")) j.at("connection_pool_size").get_to(c.connection_pool_size);
    if (j.contains("api_key")) c.api_key = j.at("api_key").get<std::string>();
}

// ============================================================================
// OCRApiConfig
// ============================================================================

void to_json(json& j, const OCRApiConfig& c) {
    j = json{
        {"api_host", c.api_host},
        {"api_port", c.api_port},
        {"api_path", c.api_path},
        {"api_mode", c.api_mode},
        {"model", c.model},
        {"headers", c.headers},
        {"verify_ssl", c.verify_ssl},
        {"connect_timeout", c.connect_timeout},
        {"request_timeout", c.request_timeout},
        {"retry_max_attempts", c.retry_max_attempts},
        {"retry_backoff_base_seconds", c.retry_backoff_base_seconds},
        {"retry_backoff_max_seconds", c.retry_backoff_max_seconds},
        {"retry_jitter_ratio", c.retry_jitter_ratio},
        {"retry_status_codes", c.retry_status_codes},
        {"connection_pool_size", c.connection_pool_size}
    };
    if (c.api_scheme) j["api_scheme"] = *c.api_scheme;
    if (c.api_url) j["api_url"] = *c.api_url;
    if (c.api_key) j["api_key"] = *c.api_key;
}

void from_json(const json& j, OCRApiConfig& c) {
    if (j.contains("api_host")) j.at("api_host").get_to(c.api_host);
    if (j.contains("api_port")) j.at("api_port").get_to(c.api_port);
    if (j.contains("api_scheme")) c.api_scheme = j.at("api_scheme").get<std::string>();
    if (j.contains("api_path")) j.at("api_path").get_to(c.api_path);
    if (j.contains("api_url")) c.api_url = j.at("api_url").get<std::string>();
    if (j.contains("api_mode")) j.at("api_mode").get_to(c.api_mode);
    if (j.contains("model")) j.at("model").get_to(c.model);
    if (j.contains("api_key")) c.api_key = j.at("api_key").get<std::string>();
    if (j.contains("headers")) j.at("headers").get_to(c.headers);
    if (j.contains("verify_ssl")) j.at("verify_ssl").get_to(c.verify_ssl);
    if (j.contains("connect_timeout")) j.at("connect_timeout").get_to(c.connect_timeout);
    if (j.contains("request_timeout")) j.at("request_timeout").get_to(c.request_timeout);
    if (j.contains("retry_max_attempts")) j.at("retry_max_attempts").get_to(c.retry_max_attempts);
    if (j.contains("retry_backoff_base_seconds")) j.at("retry_backoff_base_seconds").get_to(c.retry_backoff_base_seconds);
    if (j.contains("retry_backoff_max_seconds")) j.at("retry_backoff_max_seconds").get_to(c.retry_backoff_max_seconds);
    if (j.contains("retry_jitter_ratio")) j.at("retry_jitter_ratio").get_to(c.retry_jitter_ratio);
    if (j.contains("retry_status_codes")) j.at("retry_status_codes").get_to(c.retry_status_codes);
    if (j.contains("connection_pool_size")) j.at("connection_pool_size").get_to(c.connection_pool_size);
}

// ============================================================================
// PageLoaderConfig
// ============================================================================

void to_json(json& j, const PageLoaderConfig& c) {
    j = json{
        {"max_tokens", c.max_tokens},
        {"temperature", c.temperature},
        {"top_p", c.top_p},
        {"top_k", c.top_k},
        {"repetition_penalty", c.repetition_penalty},
        {"t_patch_size", c.t_patch_size},
        {"patch_expand_factor", c.patch_expand_factor},
        {"image_expect_length", c.image_expect_length},
        {"image_format", c.image_format},
        {"min_pixels", c.min_pixels},
        {"max_pixels", c.max_pixels},
        {"task_prompt_mapping", c.task_prompt_mapping},
        {"pdf_dpi", c.pdf_dpi},
        {"pdf_verbose", c.pdf_verbose}
    };
    if (c.pdf_max_pages) j["pdf_max_pages"] = *c.pdf_max_pages;
}

void from_json(const json& j, PageLoaderConfig& c) {
    if (j.contains("max_tokens")) j.at("max_tokens").get_to(c.max_tokens);
    if (j.contains("temperature")) j.at("temperature").get_to(c.temperature);
    if (j.contains("top_p")) j.at("top_p").get_to(c.top_p);
    if (j.contains("top_k")) j.at("top_k").get_to(c.top_k);
    if (j.contains("repetition_penalty")) j.at("repetition_penalty").get_to(c.repetition_penalty);
    if (j.contains("t_patch_size")) j.at("t_patch_size").get_to(c.t_patch_size);
    if (j.contains("patch_expand_factor")) j.at("patch_expand_factor").get_to(c.patch_expand_factor);
    if (j.contains("image_expect_length")) j.at("image_expect_length").get_to(c.image_expect_length);
    if (j.contains("image_format")) j.at("image_format").get_to(c.image_format);
    if (j.contains("min_pixels")) j.at("min_pixels").get_to(c.min_pixels);
    if (j.contains("max_pixels")) j.at("max_pixels").get_to(c.max_pixels);
    if (j.contains("task_prompt_mapping")) j.at("task_prompt_mapping").get_to(c.task_prompt_mapping);
    if (j.contains("pdf_dpi")) j.at("pdf_dpi").get_to(c.pdf_dpi);
    if (j.contains("pdf_max_pages")) c.pdf_max_pages = j.at("pdf_max_pages").get<int>();
    if (j.contains("pdf_verbose")) j.at("pdf_verbose").get_to(c.pdf_verbose);
}

// ============================================================================
// ResultFormatterConfig
// ============================================================================

void to_json(json& j, const ResultFormatterConfig& c) {
    j = json{
        {"filter_nested", c.filter_nested},
        {"min_overlap_ratio", c.min_overlap_ratio},
        {"output_format", c.output_format},
        {"enable_merge_formula_numbers", c.enable_merge_formula_numbers},
        {"enable_merge_text_blocks", c.enable_merge_text_blocks},
        {"enable_format_bullet_points", c.enable_format_bullet_points},
        {"label_visualization_mapping", c.label_visualization_mapping}
    };
}

void from_json(const json& j, ResultFormatterConfig& c) {
    if (j.contains("filter_nested")) j.at("filter_nested").get_to(c.filter_nested);
    if (j.contains("min_overlap_ratio")) j.at("min_overlap_ratio").get_to(c.min_overlap_ratio);
    if (j.contains("output_format")) j.at("output_format").get_to(c.output_format);
    if (j.contains("enable_merge_formula_numbers")) j.at("enable_merge_formula_numbers").get_to(c.enable_merge_formula_numbers);
    if (j.contains("enable_merge_text_blocks")) j.at("enable_merge_text_blocks").get_to(c.enable_merge_text_blocks);
    if (j.contains("enable_format_bullet_points")) j.at("enable_format_bullet_points").get_to(c.enable_format_bullet_points);
    if (j.contains("label_visualization_mapping")) j.at("label_visualization_mapping").get_to(c.label_visualization_mapping);
}

// ============================================================================
// LayoutConfig
// ============================================================================

void to_json(json& j, const LayoutConfig& c) {
    j = json{
        {"threshold", c.threshold},
        {"batch_size", c.batch_size},
        {"workers", c.workers},
        {"cuda_visible_devices", c.cuda_visible_devices},
        {"layout_nms", c.layout_nms},
        {"layout_merge_bboxes_mode", c.layout_merge_bboxes_mode},
        {"use_polygon", c.use_polygon},
        {"filter_nested", c.filter_nested},
        {"min_overlap_ratio", c.min_overlap_ratio}
    };
    if (c.model_dir) j["model_dir"] = *c.model_dir;
    if (c.threshold_by_class) j["threshold_by_class"] = *c.threshold_by_class;
    if (c.device) j["device"] = *c.device;
    if (c.img_size) j["img_size"] = *c.img_size;
    if (c.layout_unclip_ratio) j["layout_unclip_ratio"] = *c.layout_unclip_ratio;
    if (c.label_task_mapping) j["label_task_mapping"] = *c.label_task_mapping;
    if (c.id2label) j["id2label"] = *c.id2label;
}

void from_json(const json& j, LayoutConfig& c) {
    if (j.contains("model_dir")) c.model_dir = j.at("model_dir").get<std::string>();
    if (j.contains("threshold")) j.at("threshold").get_to(c.threshold);
    if (j.contains("threshold_by_class")) c.threshold_by_class = j.at("threshold_by_class").get<std::map<std::string, float>>();
    if (j.contains("batch_size")) j.at("batch_size").get_to(c.batch_size);
    if (j.contains("workers")) j.at("workers").get_to(c.workers);
    if (j.contains("cuda_visible_devices")) j.at("cuda_visible_devices").get_to(c.cuda_visible_devices);
    if (j.contains("device")) c.device = j.at("device").get<std::string>();
    if (j.contains("img_size")) c.img_size = j.at("img_size").get<int>();
    if (j.contains("layout_nms")) j.at("layout_nms").get_to(c.layout_nms);
    if (j.contains("layout_unclip_ratio")) c.layout_unclip_ratio = j.at("layout_unclip_ratio").get<std::vector<float>>();
    if (j.contains("layout_merge_bboxes_mode")) j.at("layout_merge_bboxes_mode").get_to(c.layout_merge_bboxes_mode);
    if (j.contains("label_task_mapping")) c.label_task_mapping = j.at("label_task_mapping").get<std::map<std::string, std::vector<std::string>>>();
    if (j.contains("use_polygon")) j.at("use_polygon").get_to(c.use_polygon);
    if (j.contains("id2label")) c.id2label = j.at("id2label").get<std::map<std::string, std::string>>();
    if (j.contains("filter_nested")) j.at("filter_nested").get_to(c.filter_nested);
    if (j.contains("min_overlap_ratio")) j.at("min_overlap_ratio").get_to(c.min_overlap_ratio);
}

// ============================================================================
// DocDetectorConfig
// ============================================================================

void to_json(json& j, const DocDetectorConfig& c) {
    j = json{
        {"input_size", c.input_size},
        {"conf_threshold", c.conf_threshold},
        {"iou_threshold", c.iou_threshold},
        {"max_det", c.max_det},
        {"workers", c.workers}
    };
    if (c.model_path) j["model_path"] = *c.model_path;
}

void from_json(const json& j, DocDetectorConfig& c) {
    if (j.contains("model_path")) c.model_path = j.at("model_path").get<std::string>();
    if (j.contains("input_size")) j.at("input_size").get_to(c.input_size);
    if (j.contains("conf_threshold")) j.at("conf_threshold").get_to(c.conf_threshold);
    if (j.contains("iou_threshold")) j.at("iou_threshold").get_to(c.iou_threshold);
    if (j.contains("max_det")) j.at("max_det").get_to(c.max_det);
    if (j.contains("workers")) j.at("workers").get_to(c.workers);
}

// ============================================================================
// OrientationConfig
// ============================================================================

void to_json(json& j, const OrientationConfig& c) {
    j = json{
        {"resize_short", c.resize_short},
        {"crop_size", c.crop_size},
        {"batch_size", c.batch_size},
        {"workers", c.workers}
    };
    if (c.model_path) j["model_path"] = *c.model_path;
}

void from_json(const json& j, OrientationConfig& c) {
    if (j.contains("model_path")) c.model_path = j.at("model_path").get<std::string>();
    if (j.contains("resize_short")) j.at("resize_short").get_to(c.resize_short);
    if (j.contains("crop_size")) j.at("crop_size").get_to(c.crop_size);
    if (j.contains("batch_size")) j.at("batch_size").get_to(c.batch_size);
    if (j.contains("workers")) j.at("workers").get_to(c.workers);
}

// ============================================================================
// UnDistortConfig
// ============================================================================

void to_json(json& j, const UnDistortConfig& c) {
    j = json{
        {"img_width", c.img_width},
        {"img_height", c.img_height},
        {"grid_w", c.grid_w},
        {"grid_h", c.grid_h},
        {"workers", c.workers}
    };
    if (c.model_path) j["model_path"] = *c.model_path;
}

void from_json(const json& j, UnDistortConfig& c) {
    if (j.contains("model_path")) c.model_path = j.at("model_path").get<std::string>();
    if (j.contains("img_width")) j.at("img_width").get_to(c.img_width);
    if (j.contains("img_height")) j.at("img_height").get_to(c.img_height);
    if (j.contains("grid_w")) j.at("grid_w").get_to(c.grid_w);
    if (j.contains("grid_h")) j.at("grid_h").get_to(c.grid_h);
    if (j.contains("workers")) j.at("workers").get_to(c.workers);
}

// ============================================================================
// PreprocessConfig
// ============================================================================

void to_json(json& j, const PreprocessConfig& c) {
    j = json{
        {"doc_detector", c.doc_detector},
        {"orientation", c.orientation},
        {"undistort", c.undistort},
        {"enable_doc_detect", c.enable_doc_detect},
        {"enable_orientation", c.enable_orientation},
        {"enable_undistort", c.enable_undistort}
    };
}

void from_json(const json& j, PreprocessConfig& c) {
    if (j.contains("doc_detector")) j.at("doc_detector").get_to(c.doc_detector);
    if (j.contains("orientation")) j.at("orientation").get_to(c.orientation);
    if (j.contains("undistort")) j.at("undistort").get_to(c.undistort);
    if (j.contains("enable_doc_detect")) j.at("enable_doc_detect").get_to(c.enable_doc_detect);
    if (j.contains("enable_orientation")) j.at("enable_orientation").get_to(c.enable_orientation);
    if (j.contains("enable_undistort")) j.at("enable_undistort").get_to(c.enable_undistort);
}

// ============================================================================
// PipelineConfig
// ============================================================================

void to_json(json& j, const PipelineConfig& c) {
    j = json{
        {"maas", c.maas},
        {"page_loader", c.page_loader},
        {"ocr_api", c.ocr_api},
        {"result_formatter", c.result_formatter},
        {"layout", c.layout},
        {"preprocess", c.preprocess},
        {"max_workers", c.max_workers},
        {"page_maxsize", c.page_maxsize},
        {"region_maxsize", c.region_maxsize}
    };
}

void from_json(const json& j, PipelineConfig& c) {
    if (j.contains("maas")) j.at("maas").get_to(c.maas);
    if (j.contains("page_loader")) j.at("page_loader").get_to(c.page_loader);
    if (j.contains("ocr_api")) j.at("ocr_api").get_to(c.ocr_api);
    if (j.contains("result_formatter")) j.at("result_formatter").get_to(c.result_formatter);
    if (j.contains("layout")) j.at("layout").get_to(c.layout);
    if (j.contains("preprocess")) j.at("preprocess").get_to(c.preprocess);
    if (j.contains("max_workers")) j.at("max_workers").get_to(c.max_workers);
    if (j.contains("page_maxsize")) j.at("page_maxsize").get_to(c.page_maxsize);
    if (j.contains("region_maxsize")) j.at("region_maxsize").get_to(c.region_maxsize);
}

// ============================================================================
// LoggingConfig
// ============================================================================

void to_json(json& j, const LoggingConfig& c) {
    j = json{{"level", c.level}};
    if (c.format) j["format"] = *c.format;
}

void from_json(const json& j, LoggingConfig& c) {
    if (j.contains("level")) j.at("level").get_to(c.level);
    if (j.contains("format")) c.format = j.at("format").get<std::string>();
}

// ============================================================================
// ServerConfig
// ============================================================================

void to_json(json& j, const ServerConfig& c) {
    j = json{
        {"host", c.host},
        {"port", c.port},
        {"debug", c.debug}
    };
}

void from_json(const json& j, ServerConfig& c) {
    if (j.contains("host")) j.at("host").get_to(c.host);
    if (j.contains("port")) j.at("port").get_to(c.port);
    if (j.contains("debug")) j.at("debug").get_to(c.debug);
}

// ============================================================================
// GlmOcrConfig
// ============================================================================

void to_json(json& j, const GlmOcrConfig& c) {
    j = json{
        {"pipeline", c.pipeline},
        {"server", c.server},
        {"logging", c.logging}
    };
}

void from_json(const json& j, GlmOcrConfig& c) {
    if (j.contains("pipeline")) j.at("pipeline").get_to(c.pipeline);
    if (j.contains("server")) j.at("server").get_to(c.server);
    if (j.contains("logging")) j.at("logging").get_to(c.logging);
}

std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

json yaml_to_json(const YAML::Node& node) {
    if (node.IsNull()) return json(nullptr);
    if (node.IsScalar()) {
        try {
            return json(node.as<bool>());
        } catch (...) {}
        try {
            return json(node.as<long long>());
        } catch (...) {}
        try {
            return json(node.as<double>());
        } catch (...) {}
        return json(node.as<std::string>());
    }
    if (node.IsSequence()) {
        json j = json::array();
        for (const auto& item : node) j.push_back(yaml_to_json(item));
        return j;
    }
    if (node.IsMap()) {
        json j = json::object();
        for (const auto& it : node) {
            j[it.first.as<std::string>()] = yaml_to_json(it.second);
        }
        return j;
    }
    return json(nullptr);
}

GlmOcrConfig GlmOcrConfig::from_json(const std::string& json_str) {
    GlmOcrConfig config;
    json j = json::parse(json_str);
    ::glmocr::from_json(j, config);
    return config;
}

GlmOcrConfig GlmOcrConfig::from_json_file(const std::string& filepath) {
    std::string content = read_file(filepath);
    return from_json(content);
}

GlmOcrConfig GlmOcrConfig::from_yaml(const std::string& yaml_str) {
    YAML::Node root = YAML::Load(yaml_str);
    json j = yaml_to_json(root);
    GlmOcrConfig config;
    ::glmocr::from_json(j, config);
    return config;
}

GlmOcrConfig GlmOcrConfig::from_yaml_file(const std::string& filepath) {
    std::string content = read_file(filepath);
    return from_yaml(content);
}

GlmOcrConfig GlmOcrConfig::from_file(const std::string& filepath) {
    namespace fs = std::filesystem;
    std::string ext = fs::path(filepath).extension().string();
    if (ext == ".yaml" || ext == ".yml") {
        return from_yaml_file(filepath);
    } else {
        return from_json_file(filepath);
    }
}

std::string GlmOcrConfig::to_json() const {
    json j = *this;
    return j.dump(4);
}

} // namespace glmocr
