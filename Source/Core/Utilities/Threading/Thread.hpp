#pragma once

#include "Core\System\FuturePromiseTaskCoarse.hpp"
#include "Core\System\concurrentqueue.h"

namespace IntelliDesign_NS::Core {

using Type_Executor = MemoryPool::Type_UniquePtr<TaskExecutor_Basic>;

class Thread {
public:
    Thread(::std::pmr::memory_resource* pMemPool);

    template <class Func, class... Args>
    auto Submit(bool bNotify, bool bExternalDep, Func&& func, Args&&... args)
        -> MemoryPool::Type_SharedPtr<
            TaskRequestHandleCoarse<decltype(func(args...))>>;

    ~Thread();

private:
    ::std::pmr::memory_resource* pMemPool;

    ::std::condition_variable mCondVar;
    ::std::mutex mMutex;

    bool mShutdown;

    moodycamel::ConcurrentQueue<Type_Executor> mQueue;

    ::std::thread mThread;
};

inline Thread::Thread(::std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool), mShutdown(false) {
    mThread = ::std::thread([this]() {
        while (true) {
            Type_Executor func {nullptr};

            if (mQueue.try_dequeue(func))
                func->Execute();
            else {
                ::std::unique_lock lock {mMutex};

                mCondVar.wait(lock, [this]() {
                    return mShutdown || mQueue.size_approx();
                });

                if (mShutdown)
                    break;
            }
        }
    });
}

inline Thread::~Thread() {
    {
        ::std::unique_lock lock {mMutex};
        mShutdown = true;
    }

    mCondVar.notify_all();

    if (mThread.joinable()) {
        mThread.join();
    }
}

template <class Func, class... Args>
auto Thread::Submit(bool bNotify, bool bExternalDep, Func&& func,
                    Args&&... args)
    -> MemoryPool::Type_SharedPtr<
        TaskRequestHandleCoarse<decltype(func(args...))>> {
    auto [pTask, pExecutor] = Create_TaskRequestCoarse(
        pMemPool, bExternalDep, ::std::forward<Func>(func),
        ::std::forward<Args>(args)...);

    mQueue.enqueue(::std::move(pExecutor));

    if (bNotify)
        mCondVar.notify_one();

    return pTask;
}

class ResourceManagerThread : public Thread {
public:
    ResourceManagerThread(::std::pmr::memory_resource* pMemPool);

private:

};

}  // namespace IntelliDesign_NS::Core