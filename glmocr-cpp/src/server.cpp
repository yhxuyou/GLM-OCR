#include "glmocr/server.h"
#include "glmocr/utils/image_utils.h"
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>
#include <cstring>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif

namespace glmocr {

using json = nlohmann::json;

// --- TaskQueue ---

TaskQueue::TaskQueue(size_t max_size) : max_size_(max_size) {}
TaskQueue::~TaskQueue() { shutdown(); }

bool TaskQueue::push(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdown_) return false;
    not_full_.wait(lock, [this] { return queue_.size() < max_size_ || shutdown_; });
    if (shutdown_) return false;
    queue_.push(std::move(task));
    not_empty_.notify_one();
    return true;
}

bool TaskQueue::pop(Task& task) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (shutdown_ && queue_.empty()) return false;
    task = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
}

size_t TaskQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool TaskQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void TaskQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
}

// --- WorkerPool ---

WorkerPool::WorkerPool(int num_workers, ProcessFunc func)
    : num_workers_(num_workers), process_func_(std::move(func)) {}

WorkerPool::~WorkerPool() { stop(); }

void WorkerPool::start() {
    if (running_) return;
    running_ = true;
    for (int i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&WorkerPool::worker_loop, this);
    }
    std::cout << "[WorkerPool] Started " << num_workers_ << " workers" << std::endl;
}

void WorkerPool::stop() {
    if (!running_) return;
    running_ = false;
    queue_.shutdown();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
    std::cout << "[WorkerPool] Stopped" << std::endl;
}

void WorkerPool::submit(Task task) { queue_.push(std::move(task)); }

size_t WorkerPool::pending_count() const { return queue_.size(); }

void WorkerPool::worker_loop() {
    while (running_) {
        Task task;
        if (!queue_.pop(task)) break;
        try {
            auto result = process_func_(task);
            // result is handled internally by OcrServer
        } catch (const std::exception& e) {
            std::cerr << "[Worker] Task " << task.task_id << " error: " << e.what() << std::endl;
        }
    }
}

// --- ResultStore ---

void ResultStore::put(const std::string& task_id, TaskResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    store_[task_id] = {std::move(result), std::chrono::steady_clock::now()};
}

std::optional<TaskResult> ResultStore::get(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = store_.find(task_id);
    if (it == store_.end()) return std::nullopt;
    return it->second.result;
}

bool ResultStore::exists(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return store_.find(task_id) != store_.end();
}

void ResultStore::cleanup(std::chrono::seconds max_age) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    for (auto it = store_.begin(); it != store_.end();) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.timestamp);
        if (age > max_age) {
            it = store_.erase(it);
        } else {
            ++it;
        }
    }
}

// --- OcrServer ---

OcrServer::OcrServer(const GlmOcrConfig& config)
    : config_(config)
    , host_(config.server.host)
    , port_(config.server.port) {

    pipeline_ = std::make_unique<Pipeline>(config.pipeline);
    result_store_ = std::make_unique<ResultStore>();

    int num_workers = config.pipeline.max_workers > 0 ? config.pipeline.max_workers : 4;
    worker_pool_ = std::make_unique<WorkerPool>(num_workers,
        [this](const Task& task) -> TaskResult {
            return process_task(task);
        });
}

OcrServer::~OcrServer() { stop(); }

std::string OcrServer::generate_task_id() {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    std::stringstream ss;
    ss << "task-" << ms << "-" << dis(gen) << "-" << counter.fetch_add(1);
    return ss.str();
}

TaskResult OcrServer::process_task(const Task& task) {
    TaskResult result;
    result.task_id = task.task_id;

    try {
        if (!task.file_paths.empty()) {
            if (task.file_paths.size() == 1) {
                auto pipe_result = pipeline_->process(task.file_paths[0]);
                result.json_result = pipe_result.json_result;
                result.markdown_result = pipe_result.markdown_output;
            } else {
                auto pipe_results = pipeline_->process_batch(task.file_paths);
                json json_list = json::array();
                std::vector<std::string> md_parts;
                for (auto& r : pipe_results) {
                    json_list.push_back(json::parse(r.json_result.empty() ? "null" : r.json_result));
                    md_parts.push_back(r.markdown_output);
                }
                result.json_result = json_list.dump(2);
                result.markdown_result = "";
                for (size_t i = 0; i < md_parts.size(); ++i) {
                    if (i > 0) result.markdown_result += "\n\n---\n\n";
                    result.markdown_result += md_parts[i];
                }
            }
            result.success = true;
        } else {
            result.success = false;
            result.error_message = "No file paths provided";
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Parse error: ") + e.what();
    }

    result_store_->put(task.task_id, result);
    return result;
}

std::string OcrServer::build_response_json(const TaskResult& result) {
    json j;
    j["json_result"] = json::parse(result.json_result.empty() ? "null" : result.json_result);
    j["markdown_result"] = result.markdown_result;
    j["layout_details"] = j["json_result"];
    j["md_results"] = result.markdown_result;
    j["data_info"] = {{"pages", json::array()}};
    j["usage"] = json::object();
    j["model"] = "glm-ocr";
    j["id"] = "chatcmpl-" + result.task_id;
    j["created"] = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    if (!result.success) {
        j["error"] = result.error_message;
    }
    return j.dump(2);
}

std::string OcrServer::build_response_json(const std::vector<TaskResult>& results) {
    if (results.size() == 1) {
        return build_response_json(results[0]);
    }

    json json_list = json::array();
    std::string md_merged;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        json_list.push_back(json::parse(r.json_result.empty() ? "null" : r.json_result));
        if (i > 0) md_merged += "\n\n---\n\n";
        md_merged += r.markdown_result;
    }

    json j;
    j["json_result"] = json_list;
    j["markdown_result"] = md_merged;
    j["layout_details"] = json_list;
    j["md_results"] = md_merged;
    j["data_info"] = {{"pages", json::array()}};
    j["usage"] = json::object();
    j["model"] = "glm-ocr";
    j["id"] = "chatcmpl-batch-" + generate_task_id();
    j["created"] = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return j.dump(2);
}

bool OcrServer::start() {
    if (running_) return true;

    if (!pipeline_->start()) {
        std::cerr << "[Server] Failed to start pipeline" << std::endl;
        return false;
    }

    worker_pool_->start();
    running_ = true;

    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    // Parse endpoint - synchronous mode (blocks until done)
    svr.Post("/glmocr/parse", [this](const httplib::Request& req, httplib::Response& res) {
        if (req.get_header_value("Content-Type") != "application/json") {
            json err = {{"error", "Invalid Content-Type. Expected 'application/json'."}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        json data;
        try {
            data = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON payload"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::vector<std::string> images;
        if (data.contains("images")) {
            if (data["images"].is_string()) {
                images.push_back(data["images"].get<std::string>());
            } else if (data["images"].is_array()) {
                for (const auto& img : data["images"]) {
                    images.push_back(img.get<std::string>());
                }
            }
        }
        if (images.empty() && data.contains("file")) {
            if (data["file"].is_string() && !data["file"].get<std::string>().empty()) {
                images.push_back(data["file"].get<std::string>());
            }
        }
        if (images.empty()) {
            json err = {{"error", "No images provided"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Submit task and wait for result
        Task task;
        task.task_id = generate_task_id();
        task.file_paths = images;
        task.submit_time = std::chrono::steady_clock::now();

        // Process synchronously in this request context
        // (the worker pool handles concurrency)
        auto result = process_task(task);

        res.set_content(build_response_json(result), "application/json");
    });

    // Async parse endpoint - returns task_id immediately
    svr.Post("/glmocr/parse_async", [this](const httplib::Request& req, httplib::Response& res) {
        if (req.get_header_value("Content-Type") != "application/json") {
            json err = {{"error", "Invalid Content-Type. Expected 'application/json'."}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        json data;
        try {
            data = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON payload"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::vector<std::string> images;
        if (data.contains("images")) {
            if (data["images"].is_string()) {
                images.push_back(data["images"].get<std::string>());
            } else if (data["images"].is_array()) {
                for (const auto& img : data["images"]) {
                    images.push_back(img.get<std::string>());
                }
            }
        }
        if (images.empty() && data.contains("file")) {
            if (data["file"].is_string() && !data["file"].get<std::string>().empty()) {
                images.push_back(data["file"].get<std::string>());
            }
        }
        if (images.empty()) {
            json err = {{"error", "No images provided"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        Task task;
        task.task_id = generate_task_id();
        task.file_paths = images;
        task.submit_time = std::chrono::steady_clock::now();

        worker_pool_->submit(std::move(task));

        json resp = {
            {"task_id", task.task_id},
            {"status", "queued"},
            {"pending", worker_pool_->pending_count()}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // Query task result
    svr.Get(R"(/glmocr/result/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = req.matches[1];
        auto result = result_store_->get(task_id);

        if (!result) {
            json resp = {{"task_id", task_id}, {"status", "pending"}};
            res.set_content(resp.dump(), "application/json");
            return;
        }

        if (result->success) {
            res.set_content(build_response_json(*result), "application/json");
        } else {
            json resp = {
                {"task_id", task_id},
                {"status", "failed"},
                {"error", result->error_message}
            };
            res.status = 500;
            res.set_content(resp.dump(), "application/json");
        }
    });

    // Server status
    svr.Get("/glmocr/status", [this](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"status", "running"},
            {"pending_tasks", worker_pool_->pending_count()},
            {"host", host_},
            {"port", port_}
        };
        res.set_content(j.dump(), "application/json");
    });

    // Cleanup old results periodically
    svr.set_pre_routing_handler([this](const httplib::Request&, httplib::Response&) -> httplib::Server::HandlerResponse {
        static std::atomic<int> request_count{0};
        if (request_count.fetch_add(1) % 100 == 0) {
            result_store_->cleanup(std::chrono::seconds(3600));
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GlmOcr C++ Server starting on " << host_ << ":" << port_ << "\n";
    std::cout << "API endpoints:\n";
    std::cout << "  POST /glmocr/parse        - Synchronous parse\n";
    std::cout << "  POST /glmocr/parse_async  - Async parse (returns task_id)\n";
    std::cout << "  GET  /glmocr/result/:id   - Query async result\n";
    std::cout << "  GET  /glmocr/status       - Server status\n";
    std::cout << "  GET  /health              - Health check\n";
    std::cout << std::string(60, '=') << "\n\n";

    if (!svr.listen(host_, port_)) {
        std::cerr << "[Server] Failed to listen on " << host_ << ":" << port_ << std::endl;
        running_ = false;
        return false;
    }

    return true;
}

void OcrServer::stop() {
    if (!running_) return;
    running_ = false;
    worker_pool_->stop();
    pipeline_->stop();
    std::cout << "[Server] Stopped" << std::endl;
}

void OcrServer::wait() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace glmocr
