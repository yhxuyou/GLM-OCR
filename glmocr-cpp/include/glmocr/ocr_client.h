#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "glmocr/config.h"

namespace glmocr {

struct OCRResponse {
    std::string content;
    int status_code;
    bool success;
    std::string error_message;
};

class OCRClient {
public:
    OCRClient(const OCRApiConfig& config);
    ~OCRClient();
    
    // Initialize the client (call before use)
    bool start();
    
    // Clean up resources
    void stop();
    
    // Check if server is alive
    bool is_alive(int timeout_seconds = 5);
    
    // Process a single request
    OCRResponse process(const nlohmann::json& request_data);
    
    // Process multiple requests in parallel (thread-safe)
    std::vector<OCRResponse> process_batch(
        const std::vector<nlohmann::json>& requests,
        int max_workers = 8);
    
private:
    OCRApiConfig config_;
    CURL* curl_;
    bool initialized_;
    
    // Helper to build the full API URL
    std::string build_api_url();
    
    // Helper to convert request format for Ollama
    nlohmann::json convert_to_ollama_format(const nlohmann::json& request);
    
    // Helper to parse Ollama response
    std::string parse_ollama_response(const std::string& response_body);
    
    // Helper to parse OpenAI response
    std::string parse_openai_response(const std::string& response_body);
    
    // Calculate retry sleep time with jitter
    int calculate_sleep_ms(int attempt);
};

} // namespace glmocr
