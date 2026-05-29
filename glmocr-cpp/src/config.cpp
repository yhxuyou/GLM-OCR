#include "glmocr/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace glmocr {

using json = nlohmann::json;

void to_json(json& j, const OCRApiConfig& c) {
    j = json{
        {"api_host", c.api_host},
        {"api_port", c.api_port},
        {"api_scheme", c.api_scheme},
        {"api_path", c.api_path},
        {"verify_ssl", c.verify_ssl},
        {"connect_timeout", c.connect_timeout},
        {"request_timeout", c.request_timeout},
        {"retry_max_attempts", c.retry_max_attempts},
        {"retry_backoff_base_seconds", c.retry_backoff_base_seconds},
        {"retry_backoff_max_seconds", c.retry_backoff_max_seconds},
        {"retry_jitter_ratio", c.retry_jitter_ratio},
        {"retry_status_codes", c.retry_status_codes},
        {"connection_pool_size", c.connection_pool_size},
        {"api_mode", c.api_mode}
    };
    if (c.api_url) j["api_url"] = *c.api_url;
    if (c.api_key) j["api_key"] = *c.api_key;
    if (c.model) j["model"] = *c.model;
    if (!c.headers.empty()) j["headers"] = c.headers;
}

void from_json(const json& j, OCRApiConfig& c) {
    if (j.contains("api_host")) j.at("api_host").get_to(c.api_host);
    if (j.contains("api_port")) j.at("api_port").get_to(c.api_port);
    if (j.contains("api_scheme")) j.at("api_scheme").get_to(c.api_scheme);
    if (j.contains("api_path")) j.at("api_path").get_to(c.api_path);
    if (j.contains("api_url")) c.api_url = j.at("api_url").get<std::string>();
    if (j.contains("api_key")) c.api_key = j.at("api_key").get<std::string>();
    if (j.contains("model")) c.model = j.at("model").get<std::string>();
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
    if (j.contains("api_mode")) j.at("api_mode").get_to(c.api_mode);
}

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
        {"pdf_dpi", c.pdf_dpi},
        {"pdf_verbose", c.pdf_verbose}
    };
    if (c.pdf_max_pages) j["pdf_max_pages"] = *c.pdf_max_pages;
    if (!c.task_prompt_mapping.empty()) j["task_prompt_mapping"] = c.task_prompt_mapping;
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
    if (j.contains("pdf_dpi")) j.at("pdf_dpi").get_to(c.pdf_dpi);
    if (j.contains("pdf_max_pages")) c.pdf_max_pages = j.at("pdf_max_pages").get<int>();
    if (j.contains("pdf_verbose")) j.at("pdf_verbose").get_to(c.pdf_verbose);
    if (j.contains("task_prompt_mapping")) j.at("task_prompt_mapping").get_to(c.task_prompt_mapping);
}

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
    if (j.contains("layout_unclip_ratio")) c.layout_unclip_ratio = j.at("layout_unclip_ratio").get<std::string>();
    if (j.contains("layout_merge_bboxes_mode")) j.at("layout_merge_bboxes_mode").get_to(c.layout_merge_bboxes_mode);
    if (j.contains("label_task_mapping")) c.label_task_mapping = j.at("label_task_mapping").get<std::map<std::string, std::vector<std::string>>>();
    if (j.contains("use_polygon")) j.at("use_polygon").get_to(c.use_polygon);
    if (j.contains("id2label")) c.id2label = j.at("id2label").get<std::map<std::string, std::string>>();
    if (j.contains("filter_nested")) j.at("filter_nested").get_to(c.filter_nested);
    if (j.contains("min_overlap_ratio")) j.at("min_overlap_ratio").get_to(c.min_overlap_ratio);
}

void to_json(json& j, const PipelineConfig& c) {
    j = json{
        {"page_loader", c.page_loader},
        {"ocr_api", c.ocr_api},
        {"result_formatter", c.result_formatter},
        {"layout", c.layout},
        {"max_workers", c.max_workers},
        {"page_maxsize", c.page_maxsize},
        {"region_maxsize", c.region_maxsize}
    };
}

void from_json(const json& j, PipelineConfig& c) {
    if (j.contains("page_loader")) j.at("page_loader").get_to(c.page_loader);
    if (j.contains("ocr_api")) j.at("ocr_api").get_to(c.ocr_api);
    if (j.contains("result_formatter")) j.at("result_formatter").get_to(c.result_formatter);
    if (j.contains("layout")) j.at("layout").get_to(c.layout);
    if (j.contains("max_workers")) j.at("max_workers").get_to(c.max_workers);
    if (j.contains("page_maxsize")) j.at("page_maxsize").get_to(c.page_maxsize);
    if (j.contains("region_maxsize")) j.at("region_maxsize").get_to(c.region_maxsize);
}

GlmOcrConfig GlmOcrConfig::from_json(const std::string& json_str) {
    GlmOcrConfig config;
    json j = json::parse(json_str);
    if (j.contains("pipeline")) j.at("pipeline").get_to(config.pipeline);
    if (j.contains("server_host")) j.at("server_host").get_to(config.server_host);
    if (j.contains("server_port")) j.at("server_port").get_to(config.server_port);
    if (j.contains("server_debug")) j.at("server_debug").get_to(config.server_debug);
    if (j.contains("log_level")) j.at("log_level").get_to(config.log_level);
    if (j.contains("log_format")) config.log_format = j.at("log_format").get<std::string>();
    return config;
}

GlmOcrConfig GlmOcrConfig::from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return GlmOcrConfig();
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json(buffer.str());
}

std::string GlmOcrConfig::to_json() const {
    json j;
    j["pipeline"] = pipeline;
    j["server_host"] = server_host;
    j["server_port"] = server_port;
    j["server_debug"] = server_debug;
    j["log_level"] = log_level;
    if (log_format) j["log_format"] = *log_format;
    return j.dump(4);
}

} // namespace glmocr
