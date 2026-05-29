#pragma once

#include <string>
#include <vector>
#include <memory>
#include "glmocr/common.h"

namespace glmocr {

enum class Orientation {
    ORIENTATION_0 = 0,
    ORIENTATION_90 = 1,
    ORIENTATION_180 = 2,
    ORIENTATION_270 = 3,
    ORIENTATION_UNKNOWN = -1
};

class OrientationDetector {
public:
    struct Config {
        std::string model_path;
        int resize_short = 256;
        int crop_size = 224;
        int batch_size = 3;
        int workers = 4;
    };

    explicit OrientationDetector(const Config& config);
    ~OrientationDetector();

    bool load_model(const std::string& model_path);
    void unload_model();
    bool is_loaded() const;

    Orientation detect(const Image& image);

    Image rotate_to_upright(const Image& image);

private:
    Config config_;
    bool loaded_ = false;
    std::vector<std::string> labels_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<float> preprocess(const Image& image);
    Orientation postprocess(const float* output, int batch_size);
};

} // namespace glmocr
