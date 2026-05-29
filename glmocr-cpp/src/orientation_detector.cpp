#include "glmocr/orientation_detector.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

namespace glmocr {

struct OrientationDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "glmocr-orient"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

OrientationDetector::OrientationDetector(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>()) {
    if (!config.model_path.empty()) {
        load_model(config.model_path);
    }
}

OrientationDetector::~OrientationDetector() { unload_model(); }

bool OrientationDetector::load_model(const std::string& model_path) {
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

        // Read labels from model metadata
        auto meta = impl_->session->GetModelMetadata();
        std::string chars;
        try {
            chars = meta.LookupCustomMetadataMapAllocated("character", impl_->allocator).get();
        } catch (...) {}
        if (!chars.empty()) {
            labels_.clear();
            size_t start = 0, end;
            while ((end = chars.find('\n', start)) != std::string::npos) {
                labels_.push_back(chars.substr(start, end - start));
                start = end + 1;
            }
            if (start < chars.size()) labels_.push_back(chars.substr(start));
        } else {
            labels_ = {"0", "90", "180", "270"};
        }

        loaded_ = true;
        std::cout << "[OrientationDetector] Model loaded: " << model_path << std::endl;
        std::cout << "[OrientationDetector] Labels: ";
        for (const auto& l : labels_) std::cout << l << " ";
        std::cout << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OrientationDetector] Error: " << e.what() << std::endl;
        return false;
    }
}

void OrientationDetector::unload_model() { impl_->session.reset(); loaded_ = false; }
bool OrientationDetector::is_loaded() const { return loaded_; }

std::vector<float> OrientationDetector::preprocess(const Image& image) {
    // 1. Resize short side to 256
    float percent = static_cast<float>(config_.resize_short) / std::min(image.width, image.height);
    int new_w = static_cast<int>(std::round(image.width * percent));
    int new_h = static_cast<int>(std::round(image.height * percent));

    // 2. Center crop to 224x224
    int crop = config_.crop_size;
    int w_start = (new_w - crop) / 2;
    int h_start = (new_h - crop) / 2;

    // 3. Normalize (ImageNet) and convert to CHW
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_v[3] = {0.229f, 0.224f, 0.225f};

    int batch = config_.batch_size;
    std::vector<float> tensor(batch * 3 * crop * crop, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int y = 0; y < crop; ++y) {
            int src_y = std::min(static_cast<int>((y + h_start) / percent), image.height - 1);
            for (int x = 0; x < crop; ++x) {
                int src_x = std::min(static_cast<int>((x + w_start) / percent), image.width - 1);
                int si = (src_y * image.width + src_x) * image.channels;
                float r = image.data[si] / 255.0f;
                float g = image.data[si + 1] / 255.0f;
                float b_val = image.data[si + 2] / 255.0f;

                int base = b * 3 * crop * crop;
                tensor[base + 0 * crop * crop + y * crop + x] = (r - mean[0]) / std_v[0];
                tensor[base + 1 * crop * crop + y * crop + x] = (g - mean[1]) / std_v[1];
                tensor[base + 2 * crop * crop + y * crop + x] = (b_val - mean[2]) / std_v[2];
            }
        }
    }
    return tensor;
}

Orientation OrientationDetector::postprocess(const float* output, int batch_size) {
    // Voting: argmax per sample, then majority vote
    std::map<int, int> vote_count;
    int num_classes = 4;

    for (int b = 0; b < batch_size; ++b) {
        int best_idx = 0;
        float best_val = output[b * num_classes];
        for (int c = 1; c < num_classes; ++c) {
            float val = output[b * num_classes + c];
            if (val > best_val) { best_val = val; best_idx = c; }
        }
        vote_count[best_idx]++;
    }

    int final_idx = 0, max_votes = 0;
    for (const auto& [idx, count] : vote_count) {
        if (count > max_votes) { max_votes = count; final_idx = idx; }
    }

    return static_cast<Orientation>(final_idx);
}

Orientation OrientationDetector::detect(const Image& image) {
    if (!loaded_ || image.data.empty()) return Orientation::ORIENTATION_UNKNOWN;

    auto tensor = preprocess(image);
    int crop = config_.crop_size;
    int batch = config_.batch_size;
    std::vector<int64_t> dims = {batch, 3, crop, crop};

    try {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(), dims.data(), dims.size());
        auto outputs = impl_->session->Run(Ort::RunOptions{}, impl_->input_name_ptrs.data(),
            &input_tensor, 1, impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size());

        const float* data = outputs[0].GetTensorData<float>();
        return postprocess(data, batch);
    } catch (const std::exception& e) {
        std::cerr << "[OrientationDetector] Inference error: " << e.what() << std::endl;
        return Orientation::ORIENTATION_UNKNOWN;
    }
}

Image OrientationDetector::rotate_to_upright(const Image& image) {
    Orientation orient = detect(image);
    if (orient == Orientation::ORIENTATION_0 || orient == Orientation::ORIENTATION_UNKNOWN) {
        return image;
    }

    int rotations = static_cast<int>(orient);
    int new_w = image.width, new_h = image.height;
    if (rotations == 1 || rotations == 3) std::swap(new_w, new_h);

    Image result(new_w, new_h, image.channels, {});
    result.data.resize(new_w * new_h * image.channels);

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_x, src_y;
            switch (rotations) {
                case 1: src_x = y; src_y = new_w - 1 - x; break; // 90 CW
                case 2: src_x = image.width - 1 - x; src_y = image.height - 1 - y; break; // 180
                case 3: src_x = image.height - 1 - y; src_y = x; break; // 270 CW
                default: src_x = x; src_y = y; break;
            }
            for (int c = 0; c < image.channels; ++c) {
                result.data[(y * new_w + x) * image.channels + c] =
                    image.data[(src_y * image.width + src_x) * image.channels + c];
            }
        }
    }
    return result;
}

} // namespace glmocr
