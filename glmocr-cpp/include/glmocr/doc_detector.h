#pragma once

#include <string>
#include <vector>
#include <memory>
#include "glmocr/common.h"

namespace glmocr {

struct DocDetection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class DocDetector {
public:
    struct Config {
        std::string model_path;
        int input_size = 640;
        float conf_threshold = 0.25f;
        float iou_threshold = 0.45f;
        int max_det = 300;
        int workers = 4;
    };

    explicit DocDetector(const Config& config);
    ~DocDetector();

    bool load_model(const std::string& model_path);
    void unload_model();
    bool is_loaded() const;

    std::vector<DocDetection> detect(const Image& image);

    Image crop_document(const Image& image, const DocDetection& det);

private:
    Config config_;
    bool loaded_ = false;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<float> preprocess(const Image& image, float& scale, int& pad_w, int& pad_h);
    std::vector<DocDetection> postprocess(const float* output, int num_rows,
                                          float scale, int pad_w, int pad_h,
                                          int orig_w, int orig_h);
    std::vector<DocDetection> nms(std::vector<DocDetection>& dets, float iou_thresh);
};

} // namespace glmocr
