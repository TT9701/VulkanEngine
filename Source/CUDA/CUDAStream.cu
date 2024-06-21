#include "CUDAStream.h"

namespace CUDA {

void CUDAStream::Synchronize() const {
    HANDLE_ERROR(cudaStreamSynchronize(mStream));
}

void CUDAStream::EventSynchronize(cudaEvent_t cuEvent) const {
    HANDLE_ERROR(cudaEventSynchronize(cuEvent));
}

void CUDAStream::WaitExternalSemaphoresAsync(
    cudaExternalSemaphore_t const*         semsArray,
    cudaExternalSemaphoreWaitParams const* paramsArray,
    unsigned int                           numExtSems) {
    HANDLE_ERROR(cudaWaitExternalSemaphoresAsync(semsArray, paramsArray,
                                                 numExtSems, mStream));
}

void CUDAStream::SignalExternalSemaphoresAsyn(
    cudaExternalSemaphore_t const*           semsArray,
    cudaExternalSemaphoreSignalParams const* paramsArray,
    unsigned int                             numExtSems) {
    HANDLE_ERROR(cudaSignalExternalSemaphoresAsync(semsArray, paramsArray,
                                                   numExtSems, mStream));
}

}  // namespace CUDA