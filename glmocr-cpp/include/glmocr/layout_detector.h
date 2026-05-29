#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <optional>
#include "glmocr/common.h"
#include "glmocr/config.h"

namespace glmocr {

struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float xmin = 0.0f;
    float ymin = 0.0f;
    float xmax = 0.0f;
    float ymax = 0.0f;
    int read_order = 0;
    std::string label;
    std::string native_label;
};

class LayoutDetector {
public:
    explicit LayoutDetector(const LayoutConfig& config);
    ~LayoutDetector();

    bool load_model(const std::string& model_path);
    void unload_model();
    bool is_loaded() const;

    std::vector<Detection> detect(const Image& image);
    std::vector<Region> detect_regions(const Image& image);

private:
    LayoutConfig config_;
    bool loaded_ = false;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<float> preprocess(const Image& image, float& scale_h, float& scale_w);
    std::vector<Detection> postprocess(const float* output_data, int num_detections,
                                       float scale_h, float scale_w,
                                       int orig_h, int orig_w);
    std::vector<Detection> filter_by_score(const std::vector<Detection>& dets, float threshold);
    std::vector<Detection> filter_nested(const std::vector<Detection>& dets, float min_overlap_ratio);

    static const std::map<int, std::string> DEFAULT_ID2LABEL;
    static std::map<int, std::string> create_default_id2label();
};

} // namespace glmocr
