#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include "glmocr/server.h"
#include "glmocr/config.h"

static glmocr::OcrServer* g_server = nullptr;

void signal_handler(int signum) {
    std::cout << "\n[Signal] Received signal " << signum << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
    exit(signum);
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help                Show this help\n";
    std::cout << "  --config PATH             Config JSON file path\n";
    std::cout << "  --host HOST               Server host (default: 0.0.0.0)\n";
    std::cout << "  --port PORT               Server port (default: 5002)\n";
    std::cout << "  --workers N               Number of worker threads (default: 16)\n";
    std::cout << "  --ocr-host HOST           OCR API host\n";
    std::cout << "  --ocr-port PORT           OCR API port\n";
    std::cout << "  --layout-model PATH       Layout detection ONNX model\n";
    std::cout << "  --doc-model PATH          Doc detection YOLO ONNX model\n";
    std::cout << "  --orientation-model PATH  Orientation detection ONNX model\n";
    std::cout << "  --undistort-model PATH    UnDistort ONNX model\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --config config.json\n";
    std::cout << "  " << prog << " --port 8080 --ocr-host localhost --ocr-port 8000\n";
}

int main(int argc, char* argv[]) {
    glmocr::GlmOcrConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--config") {
            if (++i >= argc) { std::cerr << "--config requires argument\n"; return 1; }
            config = glmocr::GlmOcrConfig::from_file(argv[i]);
        }
        else if (arg == "--host") { if (++i >= argc) return 1; config.server.host = argv[i]; }
        else if (arg == "--port") { if (++i >= argc) return 1; config.server.port = std::stoi(argv[i]); }
        else if (arg == "--workers") { if (++i >= argc) return 1; config.pipeline.max_workers = std::stoi(argv[i]); }
        else if (arg == "--ocr-host") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_host = argv[i]; }
        else if (arg == "--ocr-port") { if (++i >= argc) return 1; config.pipeline.ocr_api.api_port = std::stoi(argv[i]); }
        else if (arg == "--layout-model") { if (++i >= argc) return 1; config.pipeline.layout.model_dir = argv[i]; }
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
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        glmocr::OcrServer server(config);
        g_server = &server;

        if (!server.start()) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }

        server.wait();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
