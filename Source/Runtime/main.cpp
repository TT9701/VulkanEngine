#include <Win32/DbgMemLeak.h>
#include "Core/Engine.hpp"

int main() {
    INTELLI_DS_SetDbgFlag();

    VulkanEngine engine {};

    engine.Init();

    engine.Run();

    engine.Cleanup();

    return 0;
}