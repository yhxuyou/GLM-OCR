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
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -o, --output DIR        Output directory (default: current directory)\n";
    std::cout << "  --host HOST             OCR API host (default: localhost)\n";
    std::cout << "  --port PORT             OCR API port (default: 8080)\n";
    std::cout << "  --url URL               Full OCR API URL (overrides host/port)\n";
    std::cout << "  --model NAME            Model name (default: glm-ocr)\n";
    std::cout << "  --api-key KEY           API key (if required)\n";
    std::cout << "  --json                  Only output JSON\n";
    std::cout << "  --markdown              Only output Markdown\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog_name << " document.png\n";
    std::cout << "  " << prog_name << " --host localhost --port 8080 document.pdf\n";
    std::cout << "  " << prog_name << " --output ./results *.png\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::vector<std::string> files;
    std::string output_dir = ".";
    bool json_only = false;
    bool markdown_only = false;
    
    glmocr::GlmOcrConfig config;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                std::cerr << "Error: --output requires an argument\n";
                return 1;
            }
            output_dir = argv[i];
        } else if (arg == "--host") {
            if (++i >= argc) {
                std::cerr << "Error: --host requires an argument\n";
                return 1;
            }
            config.pipeline.ocr_api.api_host = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) {
                std::cerr << "Error: --port requires an argument\n";
                return 1;
            }
            config.pipeline.ocr_api.api_port = std::stoi(argv[i]);
        } else if (arg == "--url") {
            if (++i >= argc) {
                std::cerr << "Error: --url requires an argument\n";
                return 1;
            }
            config.pipeline.ocr_api.api_url = argv[i];
        } else if (arg == "--model") {
            if (++i >= argc) {
                std::cerr << "Error: --model requires an argument\n";
                return 1;
            }
            config.pipeline.ocr_api.model = argv[i];
        } else if (arg == "--api-key") {
            if (++i >= argc) {
                std::cerr << "Error: --api-key requires an argument\n";
                return 1;
            }
            config.pipeline.ocr_api.api_key = argv[i];
        } else if (arg == "--json") {
            json_only = true;
        } else if (arg == "--markdown") {
            markdown_only = true;
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        } else {
            files.push_back(arg);
        }
    }
    
    if (files.empty()) {
        std::cerr << "Error: No input files specified\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    try {
        glmocr::GlmOcr ocr(config);
        
        int success_count = 0;
        
        for (const auto& file_path : files) {
            if (!fs::exists(file_path)) {
                std::cerr << "Warning: File not found: " << file_path << "\n";
                continue;
            }
            
            std::cout << "Processing: " << file_path << "\n";
            
            auto result = ocr.parse(file_path);
            
            // Get base filename
            fs::path p(file_path);
            std::string base_name = p.stem().string();
            
            // Save outputs
            if (!markdown_only) {
                std::string json_path = (fs::path(output_dir) / (base_name + ".json")).string();
                std::ofstream json_out(json_path);
                json_out << result.json_result;
                std::cout << "  JSON: " << json_path << "\n";
            }
            
            if (!json_only) {
                std::string md_path = (fs::path(output_dir) / (base_name + ".md")).string();
                std::ofstream md_out(md_path);
                md_out << result.markdown_output;
                std::cout << "  Markdown: " << md_path << "\n";
            }
            
            success_count++;
        }
        
        std::cout << "\nDone! Processed " << success_count << " file(s)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
