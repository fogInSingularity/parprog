#include "thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>

namespace matmul {

ThreadPool::ThreadPool(std::size_t threads) {
    if (threads == 0) throw std::invalid_argument("Thread count must be positive");
    workers_.reserve(threads - 1);
    for (std::size_t i = 1; i < threads; ++i)
        workers_.emplace_back([this](std::stop_token stop) { worker(stop); });
}

void ThreadPool::worker(std::stop_token stop) {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, stop, [this] { return !queue_.empty(); });
            if (queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
        }
        task(); // packaged_task forwards exceptions to the owning caller.
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
    if (count == 0) return;
    if (size() == 1 || count == 1) {
        for (std::size_t i = 0; i < count; ++i) task(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (;;) {
            const auto i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) return;
            task(i);
        }
    };
    std::vector<std::future<void>> futures;
    futures.reserve(std::min(count, size()) - 1);
    std::exception_ptr failure;
    try {
        for (std::size_t i = 1; i < std::min(count, size()); ++i)
            futures.push_back(submit(work));
        work(); // The calling thread is one of the p participants.
    } catch (...) {
        failure = std::current_exception();
    }
    // Always join every submitted job before referenced stack variables expire.
    for (auto& future : futures) {
        try { future.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}
}  // namespace matmul
