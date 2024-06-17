#include "simpleCUDA.h"

float* MatAdd(float* a, float* b, int length) {
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);
    int    threadMaxSize = devProp.maxThreadsPerBlock;
    int    blockSize     = (length + threadMaxSize - 1) / threadMaxSize;
    dim3   thread(threadMaxSize);
    dim3   block(blockSize);
    int    size = length * sizeof(float);
    float* sum  = (float*)malloc(size);
    float *sumGPU, *aGPU, *bGPU;
    cudaMalloc((void**)&sumGPU, size);
    cudaMalloc((void**)&aGPU, size);
    cudaMalloc((void**)&bGPU, size);
    cudaMemcpy((void*)aGPU, (void*)a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bGPU, (void*)b, size, cudaMemcpyHostToDevice);
    CudaAdd<float><<<block, thread>>>(aGPU, bGPU, sumGPU);
    //cudaThreadSynchronize();
    cudaMemcpy(sum, sumGPU, size, cudaMemcpyDeviceToHost);
    cudaFree(sumGPU);
    cudaFree(aGPU);
    cudaFree(bGPU);
    return sum;
}