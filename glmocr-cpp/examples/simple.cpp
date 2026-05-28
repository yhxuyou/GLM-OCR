#include <iostream>
#include <fstream>
#include "glmocr/pipeline.h"
#include "glmocr/config.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_file>\n";
        return 1;
    }
    
    std::string file_path = argv[1];
    
    // Configure
    glmocr::GlmOcrConfig config;
    
    // Set up OCR API
    config.pipeline.ocr_api.api_host = "localhost";
    config.pipeline.ocr_api.api_port = 8080;
    // Or use api_url for full URL:
    // config.pipeline.ocr_api.api_url = "http://localhost:8080/v1/chat/completions";
    
    // Optionally set model name
    config.pipeline.ocr_api.model = "glm-ocr";
    
    try {
        // Create OCR instance
        glmocr::GlmOcr ocr(config);
        
        std::cout << "Processing: " << file_path << "\n";
        
        // Process the file
        auto result = ocr.parse(file_path);
        
        // Output results
        std::cout << "\n=== Markdown Output ===\n";
        std::cout << result.markdown_output << "\n";
        
        std::cout << "\n=== JSON Output ===\n";
        std::cout << result.json_result << "\n";
        
        // Save outputs to files
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
