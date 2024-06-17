#include <Win32/DbgMemLeak.h>
#include "Core/Engine.hpp"
#include "../CUDA/simpleCUDA.h"


int main() {
    INTELLI_DS_SetDbgFlag();

    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {10.0f, 11.0f, 12.0f};
    float* res  = MatAdd(a, b, 3);

    printf("cuda res: %f, %f, %f \n", res[0], res[1], res[2]);

    free(res);

    VulkanEngine engine {};

    engine.Init();

    engine.Run();

    engine.Cleanup();

    return 0;
}