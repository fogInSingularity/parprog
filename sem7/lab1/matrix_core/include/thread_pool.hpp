#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace matmul {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const { return workers_.size() + 1; }

    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& task);

private:
    std::future<void> submit(std::function<void()> task);
    void worker(std::stop_token stop);

    std::mutex mutex_;
    std::condition_variable_any ready_;
    std::queue<std::function<void()>> queue_;
    std::vector<std::jthread> workers_;
};

}

#endif
