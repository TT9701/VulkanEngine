#pragma once

#include <stdint.h>

namespace IntelliDesign_NS::Core::Utils {

class FrameTimer {
public:
    FrameTimer();

    void Reset();
    double Frame();
    double Frame(double frameTime);
    double GetElapsed() const;
    double GetFrameTime() const;

    void EnterIdle();
    void LeaveIdle();

private:
    uint64_t start;
    uint64_t last;
    uint64_t lastPeriod;
    uint64_t idleStart;
    uint64_t idleTime {0};

    uint64_t GetTime();
};

class Timer {
public:
    void Start();
    double End();

private:
    uint64_t t {0};
};

uint64_t GetCurrentTimeNanoSecs();

}  // namespace IntelliDesign_NS::Core::Utils
