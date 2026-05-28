#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <map>
#include <functional>

namespace glmocr {

// Bounding box structure (normalized coordinates 0-1000)
struct BBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    
    BBox() = default;
    BBox(int x1_, int y1_, int x2_, int y2_) 
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

// Polygon structure (normalized coordinates)
struct Polygon {
    std::vector<std::pair<int, int>> points;
    
    Polygon() = default;
    explicit Polygon(const std::vector<std::pair<int, int>>& pts) : points(pts) {}
};

// Region detection result from layout analyzer
struct Region {
    int index = 0;
    std::string label;
    std::string native_label;
    float score = 0.0f;
    BBox bbox;
    Polygon polygon;
    std::string task_type;
    std::string content;
    bool is_image = false;
    std::string image_path;
};

// Page result (contains multiple regions)
struct PageResult {
    int page_index = 0;
    std::vector<Region> regions;
    std::shared_ptr<void> image_data; // Placeholder for image data
};

// Pipeline result (contains multiple pages)
struct PipelineResult {
    std::vector<PageResult> pages;
    std::string json_result;
    std::string markdown_result;
    std::map<std::string, std::shared_ptr<void>> image_files; // filename -> image data
    std::map<int, std::shared_ptr<void>> layout_vis_images; // page idx -> vis image
};

// Image wrapper
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
    
    Image() = default;
    Image(int w, int h, int c, const std::vector<uint8_t>& d) 
        : width(w), height(h), channels(c), data(d) {}
    
    bool empty() const { return data.empty(); }
};

} // namespace glmocr
