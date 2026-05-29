#include "glmocr/undistorter.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace glmocr {

struct UnDistorter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "glmocr-undistort"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

UnDistorter::UnDistorter(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>()) {
    if (!config.model_path.empty()) {
        load_model(config.model_path);
    }
}

UnDistorter::~UnDistorter() { unload_model(); }

bool UnDistorter::load_model(const std::string& model_path) {
    if (loaded_) unload_model();
    try {
        impl_->session_options.SetIntraOpNumThreads(config_.workers);
        impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto providers = Ort::GetAvailableProviders();
        for (const auto& p : providers) {
            if (p.find("CUDA") != std::string::npos) {
                OrtCUDAProviderOptions cuda_opts;
                impl_->session_options.AppendExecutionProvider_CUDA(cuda_opts);
                break;
            }
        }

        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);

        size_t n_in = impl_->session->GetInputCount();
        impl_->input_names.clear(); impl_->input_name_ptrs.clear();
        for (size_t i = 0; i < n_in; ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
            impl_->input_names.push_back(name.get());
            impl_->input_name_ptrs.push_back(impl_->input_names.back().c_str());
        }
        size_t n_out = impl_->session->GetOutputCount();
        impl_->output_names.clear(); impl_->output_name_ptrs.clear();
        for (size_t i = 0; i < n_out; ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
            impl_->output_names.push_back(name.get());
            impl_->output_name_ptrs.push_back(impl_->output_names.back().c_str());
        }

        loaded_ = true;
        std::cout << "[UnDistorter] Model loaded: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[UnDistorter] Error: " << e.what() << std::endl;
        return false;
    }
}

void UnDistorter::unload_model() { impl_->session.reset(); loaded_ = false; }
bool UnDistorter::is_loaded() const { return loaded_; }

std::vector<float> UnDistorter::preprocess(const Image& image) {
    int tw = config_.img_width, th = config_.img_height;
    std::vector<float> tensor(1 * 3 * th * tw, 0.0f);

    for (int y = 0; y < th; ++y) {
        int src_y = std::min(static_cast<int>(y * image.height / th), image.height - 1);
        for (int x = 0; x < tw; ++x) {
            int src_x = std::min(static_cast<int>(x * image.width / tw), image.width - 1);
            int si = (src_y * image.width + src_x) * image.channels;
            tensor[0 * th * tw + y * tw + x] = image.data[si] / 255.0f;
            tensor[1 * th * tw + y * tw + x] = image.data[si + 1] / 255.0f;
            tensor[2 * th * tw + y * tw + x] = image.data[si + 2] / 255.0f;
        }
    }
    return tensor;
}

void UnDistorter::bilinear_interpolate(const float* src, int src_h, int src_w,
                                        float y, float x, float& out_val) {
    int y0 = static_cast<int>(std::floor(y));
    int x0 = static_cast<int>(std::floor(x));
    int y1 = y0 + 1, x1 = x0 + 1;
    float dy = y - y0, dx = x - x0;

    y0 = std::max(0, std::min(y0, src_h - 1));
    y1 = std::max(0, std::min(y1, src_h - 1));
    x0 = std::max(0, std::min(x0, src_w - 1));
    x1 = std::max(0, std::min(x1, src_w - 1));

    float v00 = src[y0 * src_w + x0];
    float v01 = src[y0 * src_w + x1];
    float v10 = src[y1 * src_w + x0];
    float v11 = src[y1 * src_w + x1];

    out_val = (1 - dy) * (1 - dx) * v00 + (1 - dy) * dx * v01 +
              dy * (1 - dx) * v10 + dy * dx * v11;
}

std::vector<float> UnDistorter::interpolate_grid(
    const float* input, int batch, int channels,
    int in_h, int in_w, int out_h, int out_w) {

    std::vector<float> output(batch * channels * out_h * out_w);

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_ch = input + (b * channels + c) * in_h * in_w;
            float* out_ch = output.data() + (b * channels + c) * out_h * out_w;

            for (int oy = 0; oy < out_h; ++oy) {
                float sy = (out_h > 1) ? static_cast<float>(oy) * (in_h - 1) / (out_h - 1) : 0.0f;
                for (int ox = 0; ox < out_w; ++ox) {
                    float sx = (out_w > 1) ? static_cast<float>(ox) * (in_w - 1) / (out_w - 1) : 0.0f;
                    bilinear_interpolate(in_ch, in_h, in_w, sy, sx, out_ch[oy * out_w + ox]);
                }
            }
        }
    }
    return output;
}

std::vector<float> UnDistorter::grid_sample(
    const float* input, int batch, int channels,
    int h, int w, const float* grid) {

    std::vector<float> output(batch * channels * h * w);

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_ch = input + (b * channels + c) * h * w;
            float* out_ch = output.data() + (b * channels + c) * h * w;

            for (int oy = 0; oy < h; ++oy) {
                for (int ox = 0; ox < w; ++ox) {
                    int gi = (b * h * w + oy * w + ox) * 2;
                    float gx = grid[gi];
                    float gy = grid[gi + 1];

                    // Convert from [-1,1] to pixel coords (align_corners=True)
                    float px = (gx + 1.0f) * (w - 1) / 2.0f;
                    float py = (gy + 1.0f) * (h - 1) / 2.0f;

                    bilinear_interpolate(in_ch, h, w, py, px, out_ch[oy * w + ox]);
                }
            }
        }
    }
    return output;
}

Image UnDistorter::postprocess(const Image& orig_image, const float* grid_output, int grid_h, int grid_w) {
    int orig_h = orig_image.height, orig_w = orig_image.width;

    // Upsample grid from (grid_h, grid_w) to (orig_h, orig_w)
    auto upsampled = interpolate_grid(grid_output, 1, 2, grid_h, grid_w, orig_h, orig_w);

    // Prepare input in NCHW format, normalized
    std::vector<float> warped(1 * 3 * orig_h * orig_w);
    for (int y = 0; y < orig_h; ++y) {
        for (int x = 0; x < orig_w; ++x) {
            int si = (y * orig_w + x) * orig_image.channels;
            warped[0 * orig_h * orig_w + y * orig_w + x] = orig_image.data[si] / 255.0f;
            warped[1 * orig_h * orig_w + y * orig_w + x] = orig_image.data[si + 1] / 255.0f;
            warped[2 * orig_h * orig_w + y * orig_w + x] = orig_image.data[si + 2] / 255.0f;
        }
    }

    // Grid sample
    auto unwarped = grid_sample(warped.data(), 1, 3, orig_h, orig_w, upsampled.data());

    // Convert back to HWC uint8
    Image result(orig_w, orig_h, 3, {});
    result.data.resize(orig_w * orig_h * 3);
    for (int y = 0; y < orig_h; ++y) {
        for (int x = 0; x < orig_w; ++x) {
            float r = unwarped[0 * orig_h * orig_w + y * orig_w + x] * 255.0f;
            float g = unwarped[1 * orig_h * orig_w + y * orig_w + x] * 255.0f;
            float b = unwarped[2 * orig_h * orig_w + y * orig_w + x] * 255.0f;
            int di = (y * orig_w + x) * 3;
            result.data[di] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r)));
            result.data[di + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g)));
            result.data[di + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b)));
        }
    }
    return result;
}

Image UnDistorter::undistort(const Image& image) {
    if (!loaded_ || image.data.empty()) return image;

    auto tensor = preprocess(image);
    int tw = config_.img_width, th = config_.img_height;
    std::vector<int64_t> dims = {1, 3, th, tw};

    try {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(), dims.data(), dims.size());
        auto outputs = impl_->session->Run(Ort::RunOptions{}, impl_->input_name_ptrs.data(),
            &input_tensor, 1, impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size());

        auto& out = outputs[0];
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        const float* grid_data = out.GetTensorData<float>();

        int out_grid_h = static_cast<int>(shape[2]);
        int out_grid_w = static_cast<int>(shape[3]);

        return postprocess(image, grid_data, out_grid_h, out_grid_w);
    } catch (const std::exception& e) {
        std::cerr << "[UnDistorter] Inference error: " << e.what() << std::endl;
        return image;
    }
}

} // namespace glmocr
