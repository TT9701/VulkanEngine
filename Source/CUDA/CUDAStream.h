#pragma once

#include "CUDAHelper.h"

namespace CUDA {

class CUDAStream {
public:
    CUDAStream() { HANDLE_ERROR(cudaStreamCreate(&mStream)); }

    ~CUDAStream() { HANDLE_ERROR(cudaStreamDestroy(mStream)); }

public:
    cudaStream_t GetHandle() const { return mStream; }

    void Synchronize() const;

    void EventSynchronize(cudaEvent_t cuEvent) const;

    void WaitExternalSemaphoresAsync(
        cudaExternalSemaphore_t const*         semsArray,
        cudaExternalSemaphoreWaitParams const* paramsArray,
        unsigned int                           numExtSems);

    void SignalExternalSemaphoresAsyn(
        cudaExternalSemaphore_t const*           semsArray,
        cudaExternalSemaphoreSignalParams const* paramsArray,
        unsigned int                             numExtSems);

private:
    cudaStream_t mStream {};
};

}  // namespace CUDA
