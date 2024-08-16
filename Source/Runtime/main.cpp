#include <Win32/DbgMemLeak.h>

#include "Core/Vulkan/EngineCore.hpp"

int main() {
    INTELLI_DS_SetDbgFlag();

    IntelliDesign_NS::Vulkan::Core::EngineCore engine {};

    engine.Run();

    return 0;
}