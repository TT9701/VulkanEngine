#include "Timer.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

namespace IntelliDesign_NS::Core::Utils {

FrameTimer::FrameTimer() {
    Reset();
}

void FrameTimer::Reset() {
    start = GetTime();
    last = start;
    lastPeriod = 0;
}

double FrameTimer::Frame() {
    auto newTime = GetTime() - idleTime;
    lastPeriod = newTime - last;
    last = newTime;
    return static_cast<double>(lastPeriod) * 1e-9;
}

double FrameTimer::Frame(double frameTime) {
    lastPeriod = static_cast<uint64_t>(frameTime * 1e9);
    last += lastPeriod;
    return frameTime;
}

double FrameTimer::GetElapsed() const {
    return static_cast<double>(last - start) * 1e-9;
}

double FrameTimer::GetFrameTime() const {
    return static_cast<double>(lastPeriod) * 1e-9;
}

void FrameTimer::EnterIdle() {
    idleStart = GetTime();
}

void FrameTimer::LeaveIdle() {
    auto idleEnd = GetTime();
    idleTime += idleEnd - idleStart;
}

uint64_t FrameTimer::GetTime() {
    return GetCurrentTimeNanoSecs();
}

Timer::Timer() {
    Start();
}

void Timer::Start() {
    t = GetCurrentTimeNanoSecs();
}

double Timer::End() {
    auto nt = GetCurrentTimeNanoSecs();
    auto duration = nt - t;
    t = nt;
    return static_cast<double>(duration) * 1e-9;
}

#ifdef _WIN32
struct QPCFreq {
    QPCFreq() {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        invFreq = 1e9 / static_cast<double>(freq.QuadPart);
    }

    double invFreq;
} sQPCFreq;
#endif

uint64_t GetCurrentTimeNanoSecs() {
#ifdef _WIN32
    LARGE_INTEGER li;
    if (!QueryPerformanceCounter(&li))
        return 0;
    return static_cast<uint64_t>(static_cast<double>(li.QuadPart)
                                 * sQPCFreq.invFreq);
#else
    struct timespec ts = {};
#if defined(ANDROID) || defined(__FreeBSD__)
    constexpr auto timebase = CLOCK_MONOTONIC;
#else
    constexpr auto timebase = CLOCK_MONOTONIC_RAW;
#endif
    if (clock_gettime(timebase, &ts) < 0)
        return 0;
    return ts.tv_sec * 1000000000ll + ts.tv_nsec;
#endif
}

}  // namespace IntelliDesign_NS::Core::Utils