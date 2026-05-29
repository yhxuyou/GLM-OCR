#include "glmocr/layout_detector.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>

namespace glmocr {

const std::map<int, std::string> LayoutDetector::DEFAULT_ID2LABEL = LayoutDetector::create_default_id2label();

std::map<int, std::string> LayoutDetector::create_default_id2label() {
    return {
        {0,  "text"},
        {1,  "title"},
        {2,  "figure"},
        {3,  "figure_caption"},
        {4,  "table"},
        {5,  "table_caption"},
        {6,  "header"},
        {7,  "footer"},
        {8,  "reference"},
        {9,  "equation"},
        {10, "abstract"},
        {11, "content"},
        {12, "chart"},
        {13, "list_item"},
        {14, "code"},
        {15, "footnote"},
        {16, "algorithm"},
        {17, "caption"},
        {18, "seal"},
        {19, "figure_title"},
        {20, "doc_title"},
        {21, "paragraph_title"},
        {22, "reference_content"},
        {23, "image"},
        {24, "display_formula"},
        {25, "inline_formula"},
        {26, "formula_number"},
        {27, "vertical_text"},
        {28, "vision_footnote"},
        {29, "number"},
        {30, "aside_text"},
        {31, "footer_image"},
        {32, "header_image"},
    };
}

struct LayoutDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "glmocr-layout"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    int input_size = 800;
};

LayoutDetector::LayoutDetector(const LayoutConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>()) {
}

LayoutDetector::~LayoutDetector() {
    unload_model();
}

bool LayoutDetector::load_model(const std::string& model_path) {
    if (loaded_) {
        unload_model();
    }

    try {
        impl_->session_options.SetIntraOpNumThreads(config_.workers > 0 ? config_.workers : 4);
        impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (config_.img_size) {
            impl_->input_size = *config_.img_size;
        }

        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        bool has_cuda = false;
        for (const auto& p : available_providers) {
            if (p.find("CUDA") != std::string::npos) {
                has_cuda = true;
                break;
            }
        }

        if (has_cuda) {
            OrtCUDAProviderOptions cuda_options;
            impl_->session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[LayoutDetector] Using CUDA execution provider" << std::endl;
        } else {
            std::cout << "[LayoutDetector] Using CPU execution provider" << std::endl;
        }

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_path.c_str(), impl_->session_options);

        size_t num_inputs = impl_->session->GetInputCount();
        impl_->input_names.clear();
        impl_->input_name_ptrs.clear();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
            impl_->input_names.push_back(name.get());
            impl_->input_name_ptrs.push_back(impl_->input_names.back().c_str());
        }

        size_t num_outputs = impl_->session->GetOutputCount();
        impl_->output_names.clear();
        impl_->output_name_ptrs.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
            impl_->output_names.push_back(name.get());
            impl_->output_name_ptrs.push_back(impl_->output_names.back().c_str());
        }

        std::cout << "[LayoutDetector] Model loaded: " << model_path << std::endl;
        std::cout << "[LayoutDetector] Inputs: ";
        for (const auto& n : impl_->input_names) std::cout << n << " ";
        std::cout << std::endl;
        std::cout << "[LayoutDetector] Outputs: ";
        for (const auto& n : impl_->output_names) std::cout << n << " ";
        std::cout << std::endl;

        loaded_ = true;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[LayoutDetector] ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[LayoutDetector] Error loading model: " << e.what() << std::endl;
        return false;
    }
}

void LayoutDetector::unload_model() {
    if (impl_->session) {
        impl_->session.reset();
    }
    loaded_ = false;
}

bool LayoutDetector::is_loaded() const {
    return loaded_;
}

std::vector<float> LayoutDetector::preprocess(
    const Image& image, float& scale_h, float& scale_w) {

    int target_size = impl_->input_size;
    int orig_h = image.height;
    int orig_w = image.width;

    scale_h = static_cast<float>(target_size) / orig_h;
    scale_w = static_cast<float>(target_size) / orig_w;

    int new_h = static_cast<int>(orig_h * scale_h);
    int new_w = static_cast<int>(orig_w * scale_w);

    std::vector<float> input_tensor(1 * 3 * target_size * target_size, 0.0f);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};

    for (int y = 0; y < new_h && y < target_size; ++y) {
        int src_y = std::min(static_cast<int>(y / scale_h), orig_h - 1);
        for (int x = 0; x < new_w && x < target_size; ++x) {
            int src_x = std::min(static_cast<int>(x / scale_w), orig_w - 1);

            int src_idx = (src_y * orig_w + src_x) * image.channels;
            float r = image.data[src_idx + 0] / 255.0f;
            float g = image.data[src_idx + 1] / 255.0f;
            float b = image.data[src_idx + 2] / 255.0f;

            int dst_c0 = 0 * target_size * target_size + y * target_size + x;
            int dst_c1 = 1 * target_size * target_size + y * target_size + x;
            int dst_c2 = 2 * target_size * target_size + y * target_size + x;

            input_tensor[dst_c0] = (r - mean[0]) / std_val[0];
            input_tensor[dst_c1] = (g - mean[1]) / std_val[1];
            input_tensor[dst_c2] = (b - mean[2]) / std_val[2];
        }
    }

    return input_tensor;
}

std::vector<Detection> LayoutDetector::postprocess(
    const float* output_data, int num_detections,
    float scale_h, float scale_w,
    int orig_h, int orig_w) {

    std::vector<Detection> results;
    results.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i) {
        const float* row = output_data + i * 7;

        Detection det;
        det.class_id = static_cast<int>(row[0]);
        det.score = row[1];
        det.xmin = row[2] / scale_w;
        det.ymin = row[3] / scale_h;
        det.xmax = row[4] / scale_w;
        det.ymax = row[5] / scale_h;
        det.read_order = static_cast<int>(row[6]);

        det.xmin = std::max(0.0f, std::min(det.xmin, static_cast<float>(orig_w)));
        det.ymin = std::max(0.0f, std::min(det.ymin, static_cast<float>(orig_h)));
        det.xmax = std::max(0.0f, std::min(det.xmax, static_cast<float>(orig_w)));
        det.ymax = std::max(0.0f, std::min(det.ymax, static_cast<float>(orig_h)));

        auto it = DEFAULT_ID2LABEL.find(det.class_id);
        if (it != DEFAULT_ID2LABEL.end()) {
            det.label = it->second;
            det.native_label = it->second;
        } else {
            det.label = "text";
            det.native_label = "text";
        }

        if (config_.id2label) {
            auto custom_it = config_.id2label->find(std::to_string(det.class_id));
            if (custom_it != config_.id2label->end()) {
                det.native_label = custom_it->second;
            }
        }

        results.push_back(det);
    }

    return results;
}

std::vector<Detection> LayoutDetector::filter_by_score(
    const std::vector<Detection>& dets, float threshold) {
    std::vector<Detection> filtered;
    for (const auto& det : dets) {
        float thresh = threshold;
        if (config_.threshold_by_class) {
            auto it = config_.threshold_by_class->find(det.native_label);
            if (it != config_.threshold_by_class->end()) {
                thresh = it->second;
            }
        }
        if (det.score >= thresh) {
            filtered.push_back(det);
        }
    }
    return filtered;
}

std::vector<Detection> LayoutDetector::filter_nested(
    const std::vector<Detection>& dets, float min_overlap_ratio) {
    if (dets.size() <= 1) return dets;

    std::vector<Detection> result;
    std::vector<bool> skip(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (skip[i]) continue;

        float area_i = (dets[i].xmax - dets[i].xmin) * (dets[i].ymax - dets[i].ymin);
        if (area_i <= 0) continue;

        bool is_nested = false;
        for (size_t j = 0; j < dets.size(); ++j) {
            if (i == j || skip[j]) continue;

            float area_j = (dets[j].xmax - dets[j].xmin) * (dets[j].ymax - dets[j].ymin);
            if (area_j <= 0) continue;

            float ix1 = std::max(dets[i].xmin, dets[j].xmin);
            float iy1 = std::max(dets[i].ymin, dets[j].ymin);
            float ix2 = std::min(dets[i].xmax, dets[j].xmax);
            float iy2 = std::min(dets[i].ymax, dets[j].ymax);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);

            float smaller_area = std::min(area_i, area_j);
            float overlap_ratio = inter / smaller_area;

            if (overlap_ratio > min_overlap_ratio) {
                if (area_i < area_j) {
                    is_nested = true;
                    break;
                }
            }
        }

        if (!is_nested) {
            result.push_back(dets[i]);
        } else {
            skip[i] = true;
        }
    }

    return result;
}

std::vector<Detection> LayoutDetector::detect(const Image& image) {
    if (!loaded_ || image.data.empty()) {
        return {};
    }

    float scale_h, scale_w;
    std::vector<float> input_tensor = preprocess(image, scale_h, scale_w);

    int target_size = impl_->input_size;

    std::vector<int64_t> im_shape_dims = {2};
    std::vector<float> im_shape_data = {
        static_cast<float>(target_size),
        static_cast<float>(target_size)
    };

    std::vector<int64_t> image_dims = {1, 3, target_size, target_size};

    std::vector<int64_t> scale_factor_dims = {1, 2};
    std::vector<float> scale_factor_data = {scale_h, scale_w};

    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, im_shape_data.data(), im_shape_data.size(),
            im_shape_dims.data(), im_shape_dims.size()));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(),
            image_dims.data(), image_dims.size()));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, scale_factor_data.data(), scale_factor_data.size(),
            scale_factor_dims.data(), scale_factor_dims.size()));

        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size());

        auto& output_tensor = output_tensors[0];
        auto output_type = output_tensor.GetTensorTypeAndShapeInfo();
        auto output_shape = output_type.GetShape();

        int num_detections = 1;
        for (size_t i = 0; i < output_shape.size() - 1; ++i) {
            num_detections *= static_cast<int>(output_shape[i]);
        }
        if (output_shape.size() >= 2) {
            num_detections = static_cast<int>(output_shape[0]);
        }

        const float* output_data = output_tensor.GetTensorData<float>();

        auto results = postprocess(output_data, num_detections,
                                   scale_h, scale_w,
                                   image.height, image.width);

        results = filter_by_score(results, config_.threshold);

        if (config_.layout_nms && config_.filter_nested) {
            results = filter_nested(results, config_.min_overlap_ratio > 0 ? config_.min_overlap_ratio : 0.8f);
        }

        std::sort(results.begin(), results.end(),
            [](const Detection& a, const Detection& b) {
                return a.read_order < b.read_order;
            });

        return results;

    } catch (const Ort::Exception& e) {
        std::cerr << "[LayoutDetector] ONNX Runtime inference error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "[LayoutDetector] Inference error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<Region> LayoutDetector::detect_regions(const Image& image) {
    auto detections = detect(image);

    std::vector<Region> regions;
    regions.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];

        Region region;
        region.index = static_cast<int>(i);
        region.label = det.label;
        region.native_label = det.native_label;
        region.score = det.score;
        region.task_type = det.native_label;

        region.bbox = BBox(
            static_cast<int>(det.xmin * 1000 / image.width),
            static_cast<int>(det.ymin * 1000 / image.height),
            static_cast<int>(det.xmax * 1000 / image.width),
            static_cast<int>(det.ymax * 1000 / image.height)
        );

        regions.push_back(region);
    }

    return regions;
}

} // namespace glmocr
