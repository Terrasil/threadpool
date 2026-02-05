#pragma once

#include <atomic>
#include <vector>
#include <cstddef>
#include <thread>
#include <bit>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>

namespace te {
    // =========================================================
    // Worker
    // =========================================================
    template <typename TaskType, typename ThreadType>
    struct worker {
        std::deque<TaskType> queue;
        ThreadType thread;
        std::atomic<bool> running{true};
        std::mutex mtx;
        std::condition_variable cv;
    };

    // =========================================================
    // Task group
    // =========================================================
    class task_group {
    public:
        void add() noexcept {
            counter.fetch_add(1, std::memory_order_relaxed);
        }

        void done() noexcept {
            if (counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lock(mtx);
                cv.notify_all();
            }
        }

        void wait() noexcept {
            if (counter.load(std::memory_order_acquire) == 0)
                return;

            std::unique_lock lock(mtx);
            cv.wait(lock, [&] -> bool {
                return counter.load(std::memory_order_acquire) == 0;
            });
        }

    private:
        std::atomic<std::size_t> counter{0};
        std::mutex mtx;
        std::condition_variable cv;
    };

    // =========================================================
    // Thread pool
    // =========================================================
    template <
        typename TaskType = std::move_only_function<void()>,
        typename ThreadType = std::jthread
    >
    class thread_pool {
    public:
        explicit thread_pool(std::size_t thread_count = std::thread::hardware_concurrency())
            : workers(thread_count)
        {
            for (std::size_t i = 0; i < workers.size(); ++i) {
                workers[i].thread = ThreadType(
                    [this, i](const std::stop_token &st) -> void {
                        worker_loop(i, st);
                    });
            }
        }

        ~thread_pool() {
            stop();
            wait();
        }

        // -----------------------------------------------------
        // enqueue basic task
        // -----------------------------------------------------
        void enqueue(TaskType&& task) {
            auto i = rr.fetch_add(1, std::memory_order_relaxed);
            auto& w = workers[i % workers.size()];
            tg.add();
            {
                std::lock_guard lk(w.mtx);
                w.queue.push_front(std::move(task));
            }
            w.cv.notify_one();
        }

        // -----------------------------------------------------
        // enqueue task group
        // -----------------------------------------------------
        void enqueue(task_group& group, TaskType&& task) {
            group.add();
            enqueue(TaskType{
                [t = std::move(task), &group]() mutable -> void {
                    t();
                    group.done();
                }
            });
        }

        // -----------------------------------------------------
        // enqueue future
        // -----------------------------------------------------
        template <typename F>
        auto enqueue_future(F&& f) -> std::future<std::invoke_result_t<F>>
        {
            std::promise<std::invoke_result_t<F>> prom;
            auto fut = prom.get_future();

            enqueue(TaskType{
                [p = std::move(prom),
                 fn = std::forward<F>(f)]() mutable
                {
                    try {
                        if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
                            fn();
                            p.set_value();
                        } else {
                            p.set_value(fn());
                        }
                    } catch (...) {
                        p.set_exception(std::current_exception());
                    }
                }
            });

            return fut;
        }

        // -----------------------------------------------------
        // wait until pool empty
        // -----------------------------------------------------
        void wait() {
            tg.wait();
        }

        void stop() {
            for (auto& w : workers) {
                w.running.store(false, std::memory_order_relaxed);
                w.cv.notify_one();
            }
        }

        [[nodiscard]] std::size_t thread_count() const noexcept {
            return workers.size();
        }

    private:
        std::vector<worker<TaskType, ThreadType>> workers;
        std::atomic<std::size_t> rr{0};
        task_group tg;

        // -----------------------------------------------------
        // worker loop
        // -----------------------------------------------------
        void worker_loop(std::size_t index, const std::stop_token& st) {
            TaskType task;
            bool has_task = false;
            auto& self = workers[index];
            while (!st.stop_requested() && self.running.load(std::memory_order_relaxed)) {
                has_task = false;
                {
                    std::lock_guard lk(self.mtx);
                    if (!self.queue.empty()) {
                        task = std::move(self.queue.back());
                        self.queue.pop_back();
                        has_task = true;
                    }
                }
                if (has_task) {
                    execute_task(task);
                    continue;
                }
                // Stealing
                for (auto& other : workers) {
                    if (&other == &self) continue;
                    std::lock_guard lk(other.mtx);
                    if (!other.queue.empty()) {
                        task = std::move(other.queue.front());
                        other.queue.pop_front();
                        has_task = true;
                        break;
                    }
                }
                if (has_task) {
                    execute_task(task);
                    continue;
                }
                std::unique_lock lk(self.mtx);
                self.cv.wait(lk, [&] {
                     return !self.running.load()
                         || st.stop_requested()
                         || !self.queue.empty();
                });
            }
        }

        // -----------------------------------------------------
        // common execution path
        // -----------------------------------------------------
        void execute_task(TaskType& task) {
            task();
            tg.done();
        }
    };

} // namespace te
