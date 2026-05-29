#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "glmocr/pipeline.h"
#include "glmocr/config.h"

namespace fs = std::filesystem;

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS] <FILES...>\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help                Show this help message\n";
    std::cout << "  -o, --output DIR          Output directory (default: current directory)\n";
    std::cout << "  --host HOST               OCR API host (default: localhost)\n";
    std::cout << "  --port PORT               OCR API port (default: 8080)\n";
    std::cout << "  --url URL                 Full OCR API URL\n";
    std::cout << "  --model NAME              OCR model name (default: glm-ocr)\n";
    std::cout << "  --api-key KEY             API key\n";
    std::cout << "  --layout-model PATH       PP-DocLayoutV3 ONNX model path\n";
    std::cout << "  --layout-threshold F      Layout detection threshold (default: 0.3)\n";
    std::cout << "  --doc-model PATH          YOLOv11 doc detection ONNX model path\n";
    std::cout << "  --orientation-model PATH  RapidOrientation ONNX model path\n";
    std::cout << "  --undistort-model PATH    RapidUnDistort ONNX model path\n";
    std::cout << "  --json                    Only output JSON\n";
    std::cout << "  --markdown                Only output Markdown\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog_name << " document.png\n";
    std::cout << "  " << prog_name << " --layout-model layout.onnx --doc-model yolo.onnx doc.pdf\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::vector<std::string> files;
    std::string output_dir = ".";
    bool json_only = false, markdown_only = false;
    glmocr::GlmOcrConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "-o" || arg == "--output") { if (++i >= argc) return 1; output_dir = argv[i]; }
        else if (arg == "--host") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_host = argv[i]; }
        else if (arg == "--port") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_port = std::stoi(argv[i]); }
        else if (arg == "--url") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_url = argv[i]; }
        else if (arg == "--model") { if (++i >= argc) return 1; config.pipeline.ocr_api.model = argv[i]; }
        else if (arg == "--api-key") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_key = argv[i]; }
        else if (arg == "--layout-model") { if (++i >= argc) return 1; config.pipeline.layout.model_dir = argv[i]; }
        else if (arg == "--layout-threshold") { if (++i >= argc) return 1; config.pipeline.layout.threshold = std::stof(argv[i]); }
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
        else if (arg == "--json") { json_only = true; }
        else if (arg == "--markdown") { markdown_only = true; }
        else if (arg[0] == '-') { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
        else { files.push_back(arg); }
    }

    if (files.empty()) { std::cerr << "No input files\n"; return 1; }
    if (!fs::exists(output_dir)) fs::create_directories(output_dir);

    try {
        glmocr::GlmOcr ocr(config);
        int ok = 0;
        for (const auto& fp : files) {
            if (!fs::exists(fp)) { std::cerr << "Not found: " << fp << "\n"; continue; }
            std::cout << "Processing: " << fp << "\n";
            auto result = ocr.parse(fp);
            std::string base = fs::path(fp).stem().string();
            if (!markdown_only) {
                std::ofstream((fs::path(output_dir) / (base + ".json")).string()) << result.json_result;
            }
            if (!json_only) {
                std::ofstream((fs::path(output_dir) / (base + ".md")).string()) << result.markdown_output;
            }
            ok++;
        }
        std::cout << "Done! " << ok << " file(s)\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
    return 0;
}
