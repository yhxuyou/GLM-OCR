#include <iostream>
#include <fstream>
#include "glmocr/pipeline.h"
#include "glmocr/config.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_file> [onnx_model_path]\n";
        return 1;
    }
    
    std::string file_path = argv[1];
    
    glmocr::GlmOcrConfig config;
    
    // OCR API 配置
    config.pipeline.ocr_api.api_host = "localhost";
    config.pipeline.ocr_api.api_port = 8080;
    config.pipeline.ocr_api.model = "glm-ocr";
    
    // 布局检测模型配置（如果提供了 ONNX 模型路径）
    if (argc >= 3) {
        std::string model_path = argv[2];
        config.pipeline.layout.model_dir = model_path;
        config.pipeline.layout.threshold = 0.3f;
        std::cout << "Layout model: " << model_path << std::endl;
    } else {
        std::cout << "No layout model specified, using OCR-only mode" << std::endl;
    }
    
    try {
        glmocr::GlmOcr ocr(config);
        
        std::cout << "Processing: " << file_path << "\n";
        
        auto result = ocr.parse(file_path);
        
        std::cout << "\n=== Markdown Output ===\n";
        std::cout << result.markdown_output << "\n";
        
        std::cout << "\n=== JSON Output ===\n";
        std::cout << result.json_result << "\n";
        
        {
            std::ofstream out("output.md");
            out << result.markdown_output;
        }
        {
            std::ofstream out("output.json");
            out << result.json_result;
        }
        
        std::cout << "\nResults saved to output.md and output.json\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
