#include "thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>

namespace matmul {
namespace {

void consume_tasks(std::atomic<std::size_t>& next,
                   std::size_t count,
                   const std::function<void(std::size_t)>& task) {
    for (;;) {
        const auto index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) {
            return;
        }
        task(index);
    }
}

}

ThreadPool::ThreadPool(std::size_t threads) {
    if (threads == 0) {
        throw std::invalid_argument("Thread count must be positive");
    }
    workers_.reserve(threads - 1);
    for (std::size_t i = 1; i < threads; ++i) {
        workers_.emplace_back([this](std::stop_token stop) { worker(stop); });
    }
}

void ThreadPool::worker(std::stop_token stop) {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, stop, [this] { return !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop();
        }
        task();
    }
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
    auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
    auto future = packaged->get_future();
    {
        std::lock_guard lock(mutex_);
        queue_.emplace([packaged] { (*packaged)(); });
    }
    ready_.notify_one();
    return future;
}

void ThreadPool::parallel_for(std::size_t count, const std::function<void(std::size_t)>& task) {
    if (count == 0) {
        return;
    }
    if (size() == 1 || count == 1) {
        for (std::size_t i = 0; i < count; ++i) {
            task(i);
        }
        return;
    }
    std::atomic<std::size_t> next{0};
    const auto work = [&next, count, &task] { consume_tasks(next, count, task); };
    const auto participants = std::min(count, size());
    std::vector<std::future<void>> futures;
    futures.reserve(participants - 1);
    std::exception_ptr failure;
    try {
        for (std::size_t i = 1; i < participants; ++i) {
            futures.push_back(submit(work));
        }
        work();
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}
}
