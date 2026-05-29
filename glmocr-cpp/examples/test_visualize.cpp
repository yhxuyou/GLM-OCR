#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <stb_image_write.h>

#include "glmocr/config.h"
#include "glmocr/utils/image_utils.h"
#include "glmocr/doc_detector.h"
#include "glmocr/orientation_detector.h"
#include "glmocr/undistorter.h"
#include "glmocr/layout_detector.h"

namespace fs = std::filesystem;

struct Color {
    uint8_t r, g, b;
};

static const Color COLOR_RED    = {255, 60, 60};
static const Color COLOR_GREEN  = {60, 200, 60};
static const Color COLOR_BLUE   = {60, 100, 255};
static const Color COLOR_YELLOW = {255, 200, 40};
static const Color COLOR_CYAN   = {40, 220, 220};
static const Color COLOR_MAGENTA = {220, 60, 220};
static const Color COLOR_ORANGE = {255, 140, 30};
static const Color COLOR_WHITE  = {255, 255, 255};
static const Color COLOR_BLACK  = {0, 0, 0};

static const std::vector<Color> PALETTE = {
    COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE,
    {180, 80, 80}, {80, 180, 80}, {80, 80, 180},
    {200, 150, 50}, {50, 200, 150}, {150, 50, 200},
    {120, 200, 120}, {200, 120, 120}, {120, 120, 200},
    {220, 180, 100}, {100, 220, 180}, {180, 100, 220},
    {160, 160, 60}, {60, 160, 160}, {160, 60, 160},
    {200, 200, 100}, {100, 200, 200}, {200, 100, 200},
};

static Color get_color(int idx) {
    return PALETTE[idx % PALETTE.size()];
}

static void draw_rect(glmocr::Image& img, int x1, int y1, int x2, int y2, Color c, int thickness = 2) {
    for (int t = 0; t < thickness; ++t) {
        for (int x = x1 + t; x <= x2 - t; ++x) {
            if (y1 + t >= 0 && y1 + t < img.height && x >= 0 && x < img.width) {
                int idx = ((y1 + t) * img.width + x) * img.channels;
                img.data[idx] = c.r; img.data[idx+1] = c.g; img.data[idx+2] = c.b;
            }
            if (y2 - t >= 0 && y2 - t < img.height && x >= 0 && x < img.width) {
                int idx = ((y2 - t) * img.width + x) * img.channels;
                img.data[idx] = c.r; img.data[idx+1] = c.g; img.data[idx+2] = c.b;
            }
        }
        for (int y = y1 + t; y <= y2 - t; ++y) {
            if (y >= 0 && y < img.height && x1 + t >= 0 && x1 + t < img.width) {
                int idx = (y * img.width + x1 + t) * img.channels;
                img.data[idx] = c.r; img.data[idx+1] = c.g; img.data[idx+2] = c.b;
            }
            if (y >= 0 && y < img.height && x2 - t >= 0 && x2 - t < img.width) {
                int idx = (y * img.width + x2 - t) * img.channels;
                img.data[idx] = c.r; img.data[idx+1] = c.g; img.data[idx+2] = c.b;
            }
        }
    }
}

static void draw_filled_rect(glmocr::Image& img, int x1, int y1, int x2, int y2, Color c, uint8_t alpha = 80) {
    for (int y = std::max(0, y1); y <= std::min(img.height - 1, y2); ++y) {
        for (int x = std::max(0, x1); x <= std::min(img.width - 1, x2); ++x) {
            int idx = (y * img.width + x) * img.channels;
            img.data[idx]   = static_cast<uint8_t>((img.data[idx]   * (255 - alpha) + c.r * alpha) / 255);
            img.data[idx+1] = static_cast<uint8_t>((img.data[idx+1] * (255 - alpha) + c.g * alpha) / 255);
            img.data[idx+2] = static_cast<uint8_t>((img.data[idx+2] * (255 - alpha) + c.b * alpha) / 255);
        }
    }
}

struct BitmapChar {
    int width;
    int height;
    std::vector<uint8_t> data;
};

static BitmapChar make_char_bitmap(char c) {
    BitmapChar bm;
    const int S = 8;
    bm.width = S;
    bm.height = S;
    bm.data.resize(S * S, 0);

    static const std::map<char, std::vector<uint8_t>> font5x7 = {
        {'0',{0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}},
        {'1',{0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
        {'2',{0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}},
        {'3',{0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}},
        {'4',{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}},
        {'5',{0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
        {'6',{0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}},
        {'7',{0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
        {'8',{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}},
        {'9',{0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
        {'A',{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}},
        {'B',{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
        {'C',{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},
        {'D',{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
        {'E',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}},
        {'F',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
        {'G',{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}},
        {'H',{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
        {'I',{0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}},
        {'J',{0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
        {'K',{0x11,0x12,0x14,0x18,0x14,0x12,0x11}},
        {'L',{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
        {'M',{0x11,0x1B,0x15,0x15,0x11,0x11,0x11}},
        {'N',{0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
        {'O',{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}},
        {'P',{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
        {'Q',{0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}},
        {'R',{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
        {'S',{0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E}},
        {'T',{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
        {'U',{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}},
        {'V',{0x11,0x11,0x11,0x11,0x0A,0x0A,0x04}},
        {'W',{0x11,0x11,0x11,0x15,0x15,0x1B,0x11}},
        {'X',{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
        {'Y',{0x11,0x11,0x0A,0x04,0x04,0x04,0x04}},
        {'Z',{0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
        {'a',{0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}},
        {'b',{0x10,0x10,0x1E,0x11,0x11,0x11,0x1E}},
        {'c',{0x00,0x00,0x0E,0x11,0x10,0x11,0x0E}},
        {'d',{0x01,0x01,0x0F,0x11,0x11,0x11,0x0F}},
        {'e',{0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}},
        {'f',{0x06,0x08,0x08,0x1E,0x08,0x08,0x08}},
        {'g',{0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E}},
        {'h',{0x10,0x10,0x1E,0x11,0x11,0x11,0x11}},
        {'i',{0x04,0x00,0x0C,0x04,0x04,0x04,0x0E}},
        {'j',{0x02,0x00,0x06,0x02,0x02,0x12,0x0C}},
        {'k',{0x10,0x10,0x12,0x14,0x18,0x14,0x12}},
        {'l',{0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}},
        {'m',{0x00,0x00,0x1A,0x15,0x15,0x15,0x15}},
        {'n',{0x00,0x00,0x1E,0x11,0x11,0x11,0x11}},
        {'o',{0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}},
        {'p',{0x00,0x00,0x1E,0x11,0x1E,0x10,0x10}},
        {'q',{0x00,0x00,0x0F,0x11,0x0F,0x01,0x01}},
        {'r',{0x00,0x00,0x16,0x19,0x10,0x10,0x10}},
        {'s',{0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E}},
        {'t',{0x08,0x08,0x1E,0x08,0x08,0x09,0x06}},
        {'u',{0x00,0x00,0x11,0x11,0x11,0x11,0x0F}},
        {'v',{0x00,0x00,0x11,0x11,0x11,0x0A,0x04}},
        {'w',{0x00,0x00,0x11,0x11,0x15,0x15,0x0A}},
        {'x',{0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}},
        {'y',{0x00,0x00,0x11,0x11,0x0F,0x01,0x0E}},
        {'z',{0x00,0x00,0x1F,0x02,0x04,0x08,0x1F}},
        {' ',{0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
        {':',{0x00,0x04,0x04,0x00,0x04,0x04,0x00}},
        {'-',{0x00,0x00,0x00,0x1F,0x00,0x00,0x00}},
        {'.',{0x00,0x00,0x00,0x00,0x00,0x04,0x00}},
        {',',{0x00,0x00,0x00,0x00,0x04,0x04,0x08}},
        {'(',{0x02,0x04,0x08,0x08,0x08,0x04,0x02}},
        {')',{0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
        {'/',{0x01,0x01,0x02,0x04,0x08,0x10,0x10}},
        {'_',{0x00,0x00,0x00,0x00,0x00,0x00,0x1F}},
        {'=',{0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}},
        {'+',{0x00,0x04,0x04,0x1F,0x04,0x04,0x00}},
        {'%',{0x18,0x19,0x02,0x04,0x08,0x13,0x03}},
    };

    auto it = font5x7.find(c);
    if (it == font5x7.end()) {
        for (auto& p : bm.data) p = 0;
        return bm;
    }
    const auto& glyph = it->second;

    for (int row = 0; row < 7; ++row) {
        uint8_t line = glyph[row];
        for (int col = 0; col < 5; ++col) {
            bool on = (line >> (4 - col)) & 1;
            for (int dy = 0; dy < S/7; ++dy) {
                for (int dx = 0; dx < S/5; ++dx) {
                    int py = row * (S/7) + dy;
                    int px = col * (S/5) + dx;
                    if (py < S && px < S) {
                        bm.data[py * S + px] = on ? 255 : 0;
                    }
                }
            }
        }
    }
    return bm;
}

static void draw_text(glmocr::Image& img, const std::string& text, int x, int y, Color fg, Color bg, int scale = 2) {
    int cursor_x = x;
    for (char c : text) {
        auto bm = make_char_bitmap(c);
        for (int row = 0; row < bm.height; ++row) {
            for (int col = 0; col < bm.width; ++col) {
                bool on = bm.data[row * bm.width + col] > 128;
                for (int sy = 0; sy < scale; ++sy) {
                    for (int sx = 0; sx < scale; ++sx) {
                        int px = cursor_x + col * scale + sx;
                        int py = y + row * scale + sy;
                        if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                            int idx = (py * img.width + px) * img.channels;
                            if (on) {
                                img.data[idx] = fg.r; img.data[idx+1] = fg.g; img.data[idx+2] = fg.b;
                            } else {
                                img.data[idx] = bg.r; img.data[idx+1] = bg.g; img.data[idx+2] = bg.b;
                            }
                        }
                    }
                }
            }
        }
        cursor_x += bm.width * scale;
    }
}

static int text_width(const std::string& text, int scale = 2) {
    return static_cast<int>(text.size()) * 8 * scale;
}

static int text_height(int scale = 2) {
    return 8 * scale;
}

static bool save_image(const glmocr::Image& img, const std::string& filepath) {
    int result = stbi_write_png(filepath.c_str(), img.width, img.height, img.channels,
                                 img.data.data(), img.width * img.channels);
    return result != 0;
}

static std::string orientation_to_string(glmocr::Orientation o) {
    switch (o) {
        case glmocr::Orientation::ORIENTATION_0:   return "0";
        case glmocr::Orientation::ORIENTATION_90:  return "90";
        case glmocr::Orientation::ORIENTATION_180: return "180";
        case glmocr::Orientation::ORIENTATION_270: return "270";
        default: return "unknown";
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS] <image_path>\n\n";
    std::cout << "Test preprocessing and layout detection, draw results on image.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help                  Show this help\n";
    std::cout << "  -o, --output DIR            Output directory (default: ./vis_output)\n";
    std::cout << "  --doc-model PATH            YOLOv11 doc detection ONNX model\n";
    std::cout << "  --orientation-model PATH    RapidOrientation ONNX model\n";
    std::cout << "  --undistort-model PATH      RapidUnDistort ONNX model\n";
    std::cout << "  --layout-model PATH         PP-DocLayoutV3 ONNX model\n";
    std::cout << "  --layout-threshold F        Layout detection threshold (default: 0.3)\n";
    std::cout << "  --config PATH               YAML/JSON config file\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --layout-model layout.onnx document.png\n";
    std::cout << "  " << prog << " --config config.yaml document.png\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string image_path;
    std::string output_dir = "./vis_output";
    glmocr::GlmOcrConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "-o" || arg == "--output") { if (++i >= argc) return 1; output_dir = argv[i]; }
        else if (arg == "--config") { if (++i >= argc) return 1; config = glmocr::GlmOcrConfig::from_file(argv[i]); }
        else if (arg == "--doc-model") {
            if (++i >= argc) return 1;
            config.pipeline.preprocess.doc_detector.model_path = argv[i];
            config.pipeline.preprocess.enable_doc_detect = true;
        }
        else if (arg == "--orientation-model") {
            if (++i >= argc) return 1;
            config.pipeline.preprocess.orientation.model_path = argv[i];
            config.pipeline.preprocess.enable_orientation = true;
        }
        else if (arg == "--undistort-model") {
            if (++i >= argc) return 1;
            config.pipeline.preprocess.undistort.model_path = argv[i];
            config.pipeline.preprocess.enable_undistort = true;
        }
        else if (arg == "--layout-model") { if (++i >= argc) return 1; config.pipeline.layout.model_dir = argv[i]; }
        else if (arg == "--layout-threshold") { if (++i >= argc) return 1; config.pipeline.layout.threshold = std::stof(argv[i]); }
        else if (arg[0] == '-') { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
        else { image_path = arg; }
    }

    if (image_path.empty()) { std::cerr << "No image path provided\n"; return 1; }
    if (!fs::exists(image_path)) { std::cerr << "Image not found: " << image_path << "\n"; return 1; }
    fs::create_directories(output_dir);

    std::string stem = fs::path(image_path).stem().string();

    std::cout << "========================================\n";
    std::cout << " GLM-OCR Visualize Test\n";
    std::cout << "========================================\n";
    std::cout << "Input:  " << image_path << "\n";
    std::cout << "Output: " << output_dir << "\n\n";

    // Load original image
    std::cout << "[1/5] Loading image...\n";
    auto t0 = std::chrono::steady_clock::now();
    glmocr::Image orig_image = glmocr::utils::load_image(image_path);
    if (orig_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  Size: " << orig_image.width << "x" << orig_image.height
              << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "ms)\n";

    // Save original
    save_image(orig_image, (fs::path(output_dir) / (stem + "_00_original.png")).string());

    glmocr::Image current = orig_image;

    // ---- Step 2: Document Detection ----
    std::cout << "\n[2/5] Document Detection (YOLOv11)...\n";
    bool doc_detected = false;
    if (config.pipeline.preprocess.enable_doc_detect && config.pipeline.preprocess.doc_detector.model_path) {
        glmocr::DocDetector::Config det_config;
        det_config.model_path = *config.pipeline.preprocess.doc_detector.model_path;
        det_config.conf_threshold = config.pipeline.preprocess.doc_detector.conf_threshold;
        det_config.iou_threshold = config.pipeline.preprocess.doc_detector.iou_threshold;

        glmocr::DocDetector detector(det_config);
        if (detector.load_model(det_config.model_path)) {
            auto t_start = std::chrono::steady_clock::now();
            auto dets = detector.detect(current);
            auto t_end = std::chrono::steady_clock::now();
            std::cout << "  Detections: " << dets.size()
                      << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << "ms)\n";

            glmocr::Image vis = current;
            for (size_t i = 0; i < dets.size(); ++i) {
                const auto& d = dets[i];
                int ix1 = static_cast<int>(d.x1);
                int iy1 = static_cast<int>(d.y1);
                int ix2 = static_cast<int>(d.x2);
                int iy2 = static_cast<int>(d.y2);
                Color c = get_color(static_cast<int>(i));
                draw_filled_rect(vis, ix1, iy1, ix2, iy2, c, 40);
                draw_rect(vis, ix1, iy1, ix2, iy2, c, 3);
                std::stringstream ss;
                ss << "doc " << std::fixed << std::setprecision(2) << d.confidence;
                int label_y = std::max(0, iy1 - text_height(2) - 4);
                draw_text(vis, ss.str(), ix1 + 4, label_y + 2, COLOR_WHITE, c, 2);
                std::cout << "  [" << i << "] conf=" << d.confidence
                          << " bbox=[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]\n";
            }
            save_image(vis, (fs::path(output_dir) / (stem + "_01_doc_detect.png")).string());

            if (!dets.empty()) {
                current = detector.crop_document(current, dets[0]);
                save_image(current, (fs::path(output_dir) / (stem + "_01_doc_cropped.png")).string());
                doc_detected = true;
                std::cout << "  Cropped to: " << current.width << "x" << current.height << "\n";
            }
        } else {
            std::cerr << "  Failed to load doc detection model\n";
        }
    } else {
        std::cout << "  Skipped (no model specified)\n";
    }

    // ---- Step 3: Orientation Detection ----
    std::cout << "\n[3/5] Orientation Detection...\n";
    bool orientation_corrected = false;
    if (config.pipeline.preprocess.enable_orientation && config.pipeline.preprocess.orientation.model_path) {
        glmocr::OrientationDetector::Config ori_config;
        ori_config.model_path = *config.pipeline.preprocess.orientation.model_path;

        glmocr::OrientationDetector ori_detector(ori_config);
        if (ori_detector.load_model(ori_config.model_path)) {
            auto t_start = std::chrono::steady_clock::now();
            glmocr::Orientation ori = ori_detector.detect(current);
            auto t_end = std::chrono::steady_clock::now();
            std::cout << "  Orientation: " << orientation_to_string(ori)
                      << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << "ms)\n";

            glmocr::Image vis = current;
            std::string ori_text = "Orientation: " + orientation_to_string(ori) + " deg";
            draw_text(vis, ori_text, 10, 10, COLOR_YELLOW, COLOR_BLACK, 2);
            save_image(vis, (fs::path(output_dir) / (stem + "_02_orientation.png")).string());

            if (ori != glmocr::Orientation::ORIENTATION_0) {
                current = ori_detector.rotate_to_upright(current);
                save_image(current, (fs::path(output_dir) / (stem + "_02_rotated.png")).string());
                orientation_corrected = true;
                std::cout << "  Rotated to upright: " << current.width << "x" << current.height << "\n";
            }
        } else {
            std::cerr << "  Failed to load orientation model\n";
        }
    } else {
        std::cout << "  Skipped (no model specified)\n";
    }

    // ---- Step 4: UnDistort ----
    std::cout << "\n[4/5] UnDistort (UVDoc)...\n";
    bool undistorted = false;
    if (config.pipeline.preprocess.enable_undistort && config.pipeline.preprocess.undistort.model_path) {
        glmocr::UnDistorter::Config ud_config;
        ud_config.model_path = *config.pipeline.preprocess.undistort.model_path;

        glmocr::UnDistorter undistorter(ud_config);
        if (undistorter.load_model(ud_config.model_path)) {
            auto t_start = std::chrono::steady_clock::now();
            glmocr::Image undist_img = undistorter.undistort(current);
            auto t_end = std::chrono::steady_clock::now();
            std::cout << "  Undistorted: " << undist_img.width << "x" << undist_img.height
                      << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << "ms)\n";

            save_image(undist_img, (fs::path(output_dir) / (stem + "_03_undistorted.png")).string());
            current = undist_img;
            undistorted = true;
        } else {
            std::cerr << "  Failed to load undistort model\n";
        }
    } else {
        std::cout << "  Skipped (no model specified)\n";
    }

    // ---- Step 5: Layout Detection ----
    std::cout << "\n[5/5] Layout Detection (PP-DocLayoutV3)...\n";
    if (config.pipeline.layout.model_dir) {
        glmocr::LayoutDetector layout_detector(config.pipeline.layout);
        if (layout_detector.load_model(*config.pipeline.layout.model_dir)) {
            auto t_start = std::chrono::steady_clock::now();
            auto regions = layout_detector.detect_regions(current);
            auto t_end = std::chrono::steady_clock::now();
            std::cout << "  Regions: " << regions.size()
                      << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << "ms)\n";

            glmocr::Image vis = current;

            std::map<std::string, int> label_color_map;
            int color_idx = 0;

            for (size_t i = 0; i < regions.size(); ++i) {
                const auto& r = regions[i];
                std::string label = r.label.empty() ? r.native_label : r.label;

                if (label_color_map.find(label) == label_color_map.end()) {
                    label_color_map[label] = color_idx++;
                }
                Color c = get_color(label_color_map[label]);

                int ix1 = r.bbox.x1;
                int iy1 = r.bbox.y1;
                int ix2 = r.bbox.x2;
                int iy2 = r.bbox.y2;

                draw_filled_rect(vis, ix1, iy1, ix2, iy2, c, 30);
                draw_rect(vis, ix1, iy1, ix2, iy2, c, 2);

                std::stringstream ss;
                ss << label << " " << std::fixed << std::setprecision(2) << r.score;
                int label_y = std::max(0, iy1 - text_height(2) - 4);
                draw_text(vis, ss.str(), ix1 + 2, label_y + 2, COLOR_WHITE, c, 2);

                std::cout << "  [" << i << "] label=" << label
                          << " score=" << r.score
                          << " bbox=[" << r.bbox.x1 << "," << r.bbox.y1
                          << "," << r.bbox.x2 << "," << r.bbox.y2 << "]";

                if (!r.polygon.points.empty()) {
                    std::cout << " polygon=[";
                    for (size_t p = 0; p < r.polygon.points.size(); ++p) {
                        if (p > 0) std::cout << ",";
                        std::cout << "(" << r.polygon.points[p].first
                                  << "," << r.polygon.points[p].second << ")";
                    }
                    std::cout << "]";
                }
                std::cout << "\n";
            }

            // Draw legend
            int legend_y = current.height - (color_idx + 1) * (text_height(2) + 6) - 10;
            for (const auto& [label, idx] : label_color_map) {
                Color c = get_color(idx);
                int lx = 10;
                draw_filled_rect(vis, lx, legend_y, lx + 20, legend_y + text_height(2), c, 200);
                draw_rect(vis, lx, legend_y, lx + 20, legend_y + text_height(2), c, 1);
                draw_text(vis, label, lx + 26, legend_y + 2, COLOR_WHITE, COLOR_BLACK, 2);
                legend_y += text_height(2) + 6;
            }

            save_image(vis, (fs::path(output_dir) / (stem + "_04_layout.png")).string());

            // Also draw on original image if preprocessing was applied
            if (doc_detected || orientation_corrected || undistorted) {
                glmocr::Image full_vis = orig_image;
                draw_text(full_vis, "Preprocessing applied - see layout result on cropped image",
                          10, 10, COLOR_YELLOW, COLOR_BLACK, 2);
                save_image(full_vis, (fs::path(output_dir) / (stem + "_05_full_overview.png")).string());
            }

        } else {
            std::cerr << "  Failed to load layout model\n";
        }
    } else {
        std::cout << "  Skipped (no model specified)\n";
    }

    auto t_total = std::chrono::steady_clock::now();
    std::cout << "\n========================================\n";
    std::cout << " Done! Results saved to: " << output_dir << "\n";
    std::cout << "========================================\n";

    return 0;
}
