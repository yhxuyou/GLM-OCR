#include "glmocr/ocr_client.h"
#include "glmocr/utils/string_utils.h"
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>

namespace glmocr {

using json = nlohmann::json;

// Callback for writing response data
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    auto* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

OCRClient::OCRClient(const OCRApiConfig& config)
    : config_(config)
    , curl_(nullptr)
    , initialized_(false) {
}

OCRClient::~OCRClient() {
    stop();
}

std::string OCRClient::build_api_url() {
    if (config_.api_url) {
        return *config_.api_url;
    }
    std::string scheme = config_.api_scheme.has_value() ? *config_.api_scheme : 
                         (config_.api_port == 443 ? "https" : "http");
    return scheme + "://" + config_.api_host + ":" + 
           std::to_string(config_.api_port) + config_.api_path;
}

bool OCRClient::start() {
    if (initialized_) return true;
    
    curl_global_init(CURL_GLOBAL_ALL);
    curl_ = curl_easy_init();
    if (!curl_) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

void OCRClient::stop() {
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    if (initialized_) {
        curl_global_cleanup();
        initialized_ = false;
    }
}

bool OCRClient::is_alive(int timeout_seconds) {
    if (!initialized_ && !start()) {
        return false;
    }
    
    // Simple TCP connection check
    CURL* easy = curl_easy_init();
    if (!easy) return false;
    
    std::string scheme = config_.api_scheme.has_value() ? *config_.api_scheme : 
                         (config_.api_port == 443 ? "https" : "http");
    std::string url = scheme + "://" + config_.api_host + ":" + 
                      std::to_string(config_.api_port);
    
    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT, timeout_seconds);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(easy, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, config_.verify_ssl ? 1L : 0L);
    
    CURLcode res = curl_easy_perform(easy);
    bool alive = (res == CURLE_OK);
    
    // Also check HTTP code if we got that far
    if (alive) {
        long http_code = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);
        // Accept 200, 404, 405 as signs the server is up
        alive = (http_code == 200 || http_code == 404 || http_code == 405 || http_code == 0);
    }
    
    curl_easy_cleanup(easy);
    return alive;
}

int OCRClient::calculate_sleep_ms(int attempt) {
    // Exponential backoff with jitter
    double base = config_.retry_backoff_base_seconds * 1000.0;
    double max = config_.retry_backoff_max_seconds * 1000.0;
    double sleep = std::min(base * (1 << attempt), max);
    
    // Add jitter (±20%)
    std::random_device rd;
    std::mt19937 gen(rd());
    double jitter_factor = config_.retry_jitter_ratio;
    std::uniform_real_distribution<> dis(1.0 - jitter_factor, 1.0 + jitter_factor);
    sleep *= dis(gen);
    
    return static_cast<int>(sleep);
}

json OCRClient::convert_to_ollama_format(const json& request) {
    json ollama_req;
    
    // Extract prompt and image from the last user message
    std::string prompt = "Text Recognition:";
    std::string image_b64;
    
    if (request.contains("messages") && request["messages"].is_array()) {
        // Find the last user message
        for (auto it = request["messages"].rbegin(); it != request["messages"].rend(); ++it) {
            if ((*it)["role"] == "user") {
                const auto& content = (*it)["content"];
                if (content.is_string()) {
                    prompt = content;
                } else if (content.is_array()) {
                    for (const auto& part : content) {
                        if (part["type"] == "text") {
                            prompt = part["text"];
                        } else if (part["type"] == "image_url") {
                            std::string url = part["image_url"]["url"];
                            if (utils::starts_with(url, "data:image")) {
                                // Extract base64 part
                                size_t comma_pos = url.find(',');
                                if (comma_pos != std::string::npos) {
                                    image_b64 = url.substr(comma_pos + 1);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }
    
    ollama_req["model"] = config_.model;
    ollama_req["prompt"] = prompt;
    ollama_req["stream"] = false;
    
    if (!image_b64.empty()) {
        ollama_req["images"] = json::array({image_b64});
    }
    
    // Copy parameters
    json options;
    if (request.contains("max_tokens")) {
        options["num_predict"] = request["max_tokens"];
    }
    if (request.contains("temperature")) {
        options["temperature"] = request["temperature"];
    }
    if (request.contains("top_p")) {
        options["top_p"] = request["top_p"];
    }
    if (request.contains("top_k")) {
        options["top_k"] = request["top_k"];
    }
    if (request.contains("repetition_penalty")) {
        options["repeat_penalty"] = request["repetition_penalty"];
    }
    
    if (!options.empty()) {
        ollama_req["options"] = options;
    }
    
    return ollama_req;
}

std::string OCRClient::parse_openai_response(const std::string& response_body) {
    try {
        json j = json::parse(response_body);
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
            auto& choice = j["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content")) {
                return choice["message"]["content"];
            }
        }
        return "";
    } catch (...) {
        return "";
    }
}

std::string OCRClient::parse_ollama_response(const std::string& response_body) {
    try {
        json j = json::parse(response_body);
        if (j.contains("response")) {
            return j["response"];
        }
        return "";
    } catch (...) {
        return "";
    }
}

OCRResponse OCRClient::process(const json& request_data) {
    OCRResponse response;
    response.success = false;
    response.status_code = 0;
    
    if (!initialized_ && !start()) {
        response.error_message = "Failed to initialize OCR client";
        return response;
    }
    
    std::string api_url = build_api_url();
    
    // Convert request if needed
    json request = request_data;
    bool is_ollama = (config_.api_mode == "ollama_generate");
    if (is_ollama) {
        request = convert_to_ollama_format(request);
    }
    
    // Add model if configured
    if (!is_ollama) {
        request["model"] = config_.model;
    }
    
    std::string json_str = request.dump();
    
    int max_attempts = config_.retry_max_attempts + 1;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        CURL* easy = curl_easy_init();
        if (!easy) {
            response.error_message = "Failed to create curl handle";
            continue;
        }
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        // Add API key if present
        if (config_.api_key) {
            std::string auth_header = "Authorization: Bearer " + *config_.api_key;
            headers = curl_slist_append(headers, auth_header.c_str());
        }
        
        // Add custom headers
        for (const auto& [key, value] : config_.headers) {
            std::string header = key + ": " + value;
            headers = curl_slist_append(headers, header.c_str());
        }
        
        std::string response_body;
        
        curl_easy_setopt(easy, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(easy, CURLOPT_POSTFIELDS, json_str.c_str());
        curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE, json_str.size());
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, config_.verify_ssl ? 1L : 0L);
        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout);
        curl_easy_setopt(easy, CURLOPT_TIMEOUT, config_.request_timeout);
        curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
        
        CURLcode res = curl_easy_perform(easy);
        
        if (res == CURLE_OK) {
            long http_code = 0;
            curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);
            response.status_code = static_cast<int>(http_code);
            
            if (http_code == 200) {
                // Success! Parse response
                response.success = true;
                if (is_ollama) {
                    response.content = parse_ollama_response(response_body);
                } else {
                    response.content = parse_openai_response(response_body);
                }
                curl_slist_free_all(headers);
                curl_easy_cleanup(easy);
                return response;
            } else {
                // Check if we should retry
                bool should_retry = false;
                for (int code : config_.retry_status_codes) {
                    if (http_code == code) {
                        should_retry = true;
                        break;
                    }
                }
                
                if (should_retry && attempt < max_attempts - 1) {
                    // Retry after sleep
                    curl_slist_free_all(headers);
                    curl_easy_cleanup(easy);
                    std::this_thread::sleep_for(std::chrono::milliseconds(calculate_sleep_ms(attempt)));
                    continue;
                }
                
                // No more retries
                response.error_message = "HTTP error: " + std::to_string(http_code) + 
                                         " - " + response_body;
                curl_slist_free_all(headers);
                curl_easy_cleanup(easy);
                return response;
            }
        } else {
            response.error_message = "curl error: " + std::string(curl_easy_strerror(res));
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(easy);
        
        // Retry on curl errors except last attempt
        if (attempt < max_attempts - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(calculate_sleep_ms(attempt)));
        }
    }
    
    return response;
}

std::vector<OCRResponse> OCRClient::process_batch(
    const std::vector<json>& requests,
    int max_workers) {
    
    std::vector<OCRResponse> responses(requests.size());
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::atomic<int> next_idx(0);
    
    int num_workers = std::min(max_workers, static_cast<int>(requests.size()));
    
    // Worker function
    auto worker_func = [&]() {
        while (true) {
            int idx = next_idx.fetch_add(1);
            if (idx >= static_cast<int>(requests.size())) {
                break;
            }
            
            // Create a new client for each thread for thread safety
            OCRClient thread_client(config_);
            thread_client.start();
            responses[idx] = thread_client.process(requests[idx]);
            thread_client.stop();
        }
    };
    
    // Start workers
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(worker_func);
    }
    
    // Wait for all workers
    for (auto& worker : workers) {
        worker.join();
    }
    
    return responses;
}

} // namespace glmocr
