#pragma once

#include <string>
#include <vector>
#include <memory>
#include "glmocr/common.h"

namespace glmocr {

class UnDistorter {
public:
    struct Config {
        std::string model_path;
        int img_width = 488;
        int img_height = 712;
        int grid_w = 45;
        int grid_h = 31;
        int workers = 4;
    };

    explicit UnDistorter(const Config& config);
    ~UnDistorter();

    bool load_model(const std::string& model_path);
    void unload_model();
    bool is_loaded() const;

    Image undistort(const Image& image);

private:
    Config config_;
    bool loaded_ = false;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<float> preprocess(const Image& image);
    Image postprocess(const Image& orig_image, const float* grid_output, int grid_h, int grid_w);

    void bilinear_interpolate(const float* src, int src_h, int src_w,
                              float y, float x, float& out_val);
    std::vector<float> interpolate_grid(const float* input, int batch, int channels,
                                        int in_h, int in_w, int out_h, int out_w);
    std::vector<float> grid_sample(const float* input, int batch, int channels,
                                   int h, int w, const float* grid);
};

} // namespace glmocr
