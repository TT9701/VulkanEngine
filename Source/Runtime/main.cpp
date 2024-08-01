#include <Win32/DbgMemLeak.h>

#include "Core/VulkanCore/VulkanEngine.hpp"

int main() {
    INTELLI_DS_SetDbgFlag();

    VulkanEngine engine {};

    engine.Run();

    return 0;
}