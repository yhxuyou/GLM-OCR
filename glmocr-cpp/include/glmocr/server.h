#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include "glmocr/config.h"
#include "glmocr/pipeline.h"

namespace glmocr {

struct TaskResult {
    std::string task_id;
    bool success = false;
    std::string json_result;
    std::string markdown_result;
    std::string error_message;
    int status_code = 0;
};

struct Task {
    std::string task_id;
    std::vector<std::string> file_paths;
    std::vector<std::string> image_urls;
    nlohmann::json request_data;
    std::chrono::steady_clock::time_point submit_time;
};

class TaskQueue {
public:
    explicit TaskQueue(size_t max_size = 1024);
    ~TaskQueue();

    bool push(Task task);
    bool pop(Task& task);
    size_t size() const;
    bool empty() const;
    void shutdown();

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<Task> queue_;
    size_t max_size_;
    bool shutdown_ = false;
};

class WorkerPool {
public:
    using ProcessFunc = std::function<TaskResult(const Task&)>;

    WorkerPool(int num_workers, ProcessFunc func);
    ~WorkerPool();

    void start();
    void stop();
    void submit(Task task);
    size_t pending_count() const;

private:
    int num_workers_;
    ProcessFunc process_func_;
    TaskQueue queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    void worker_loop();
};

class ResultStore {
public:
    void put(const std::string& task_id, TaskResult result);
    std::optional<TaskResult> get(const std::string& task_id);
    bool exists(const std::string& task_id);
    void cleanup(std::chrono::seconds max_age);

private:
    mutable std::mutex mutex_;
    struct Entry {
        TaskResult result;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<std::string, Entry> store_;
};

class OcrServer {
public:
    explicit OcrServer(const GlmOcrConfig& config);
    ~OcrServer();

    bool start();
    void stop();
    void wait();

private:
    GlmOcrConfig config_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<WorkerPool> worker_pool_;
    std::unique_ptr<ResultStore> result_store_;

    std::string host_;
    int port_;
    std::atomic<bool> running_{false};

    TaskResult process_task(const Task& task);
    std::string build_response_json(const TaskResult& result);
    std::string build_response_json(const std::vector<TaskResult>& results);

    static std::string generate_task_id();
};

} // namespace glmocr
