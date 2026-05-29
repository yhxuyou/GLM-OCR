#include "glmocr/doc_detector.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace glmocr {

struct DocDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "glmocr-docdet"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

DocDetector::DocDetector(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>()) {
    if (!config.model_path.empty()) {
        load_model(config.model_path);
    }
}

DocDetector::~DocDetector() { unload_model(); }

bool DocDetector::load_model(const std::string& model_path) {
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
        std::cout << "[DocDetector] Model loaded: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DocDetector] Error: " << e.what() << std::endl;
        return false;
    }
}

void DocDetector::unload_model() { impl_->session.reset(); loaded_ = false; }
bool DocDetector::is_loaded() const { return loaded_; }

std::vector<float> DocDetector::preprocess(const Image& image, float& scale, int& pad_w, int& pad_h) {
    int target = config_.input_size;
    scale = std::min(static_cast<float>(target) / image.width, static_cast<float>(target) / image.height);
    int new_w = static_cast<int>(image.width * scale);
    int new_h = static_cast<int>(image.height * scale);
    pad_w = target - new_w;
    pad_h = target - new_h;

    std::vector<float> tensor(1 * 3 * target * target, 0.0f);
    for (int y = 0; y < new_h; ++y) {
        int src_y = std::min(static_cast<int>(y / scale), image.height - 1);
        for (int x = 0; x < new_w; ++x) {
            int src_x = std::min(static_cast<int>(x / scale), image.width - 1);
            int si = (src_y * image.width + src_x) * image.channels;
            tensor[0 * target * target + y * target + x] = image.data[si] / 255.0f;
            tensor[1 * target * target + y * target + x] = image.data[si + 1] / 255.0f;
            tensor[2 * target * target + y * target + x] = image.data[si + 2] / 255.0f;
        }
    }
    return tensor;
}

std::vector<DocDetection> DocDetector::postprocess(
    const float* output, int num_rows, float scale, int pad_w, int pad_h,
    int orig_w, int orig_h) {

    std::vector<DocDetection> results;
    for (int i = 0; i < num_rows; ++i) {
        const float* row = output + i * 6;
        float cx = row[1], cy = row[2], w = row[3], h = row[4], conf = row[5];
        if (conf < config_.conf_threshold) continue;

        float x1 = (cx - w / 2 - pad_w / 2) / scale;
        float y1 = (cy - h / 2 - pad_h / 2) / scale;
        float x2 = (cx + w / 2 - pad_w / 2) / scale;
        float y2 = (cy + h / 2 - pad_h / 2) / scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        DocDetection det;
        det.x1 = x1; det.y1 = y1; det.x2 = x2; det.y2 = y2;
        det.confidence = conf;
        det.class_id = static_cast<int>(row[0]);
        results.push_back(det);
    }
    return results;
}

std::vector<DocDetection> DocDetector::nms(std::vector<DocDetection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const DocDetection& a, const DocDetection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<bool> skip(dets.size(), false);
    std::vector<DocDetection> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (skip[i]) continue;
        result.push_back(dets[i]);
        float area_i = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (skip[j]) continue;
            float ix1 = std::max(dets[i].x1, dets[j].x1);
            float iy1 = std::max(dets[i].y1, dets[j].y1);
            float ix2 = std::min(dets[i].x2, dets[j].x2);
            float iy2 = std::min(dets[i].y2, dets[j].y2);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float area_j = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = inter / (area_i + area_j - inter + 1e-6f);
            if (iou > iou_thresh) skip[j] = true;
        }
    }
    return result;
}

std::vector<DocDetection> DocDetector::detect(const Image& image) {
    if (!loaded_ || image.data.empty()) return {};

    float scale; int pad_w, pad_h;
    auto tensor = preprocess(image, scale, pad_w, pad_h);
    int sz = config_.input_size;
    std::vector<int64_t> dims = {1, 3, sz, sz};

    try {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(), dims.data(), dims.size());
        auto outputs = impl_->session->Run(Ort::RunOptions{}, impl_->input_name_ptrs.data(),
            &input_tensor, 1, impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size());

        auto& out = outputs[0];
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        int num_rows = static_cast<int>(shape[1]);
        int num_cols = static_cast<int>(shape[2]);
        const float* data = out.GetTensorData<float>();

        auto dets = postprocess(data, num_rows, scale, pad_w, pad_h, image.width, image.height);
        return nms(dets, config_.iou_threshold);
    } catch (const std::exception& e) {
        std::cerr << "[DocDetector] Inference error: " << e.what() << std::endl;
        return {};
    }
}

Image DocDetector::crop_document(const Image& image, const DocDetection& det) {
    int x1 = std::max(0, static_cast<int>(det.x1));
    int y1 = std::max(0, static_cast<int>(det.y1));
    int x2 = std::min(image.width - 1, static_cast<int>(det.x2));
    int y2 = std::min(image.height - 1, static_cast<int>(det.y2));
    int w = x2 - x1, h = y2 - y1;
    if (w <= 0 || h <= 0) return Image();

    Image cropped(w, h, image.channels, {});
    cropped.data.resize(w * h * image.channels);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            for (int c = 0; c < image.channels; ++c) {
                cropped.data[(j * w + i) * image.channels + c] =
                    image.data[((y1 + j) * image.width + (x1 + i)) * image.channels + c];
            }
        }
    }
    return cropped;
}

} // namespace glmocr
