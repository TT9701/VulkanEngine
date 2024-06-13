#include "Core/Engine.hpp"
#include "Core/DbgMemLeak.h"

int main() {
    INTELLI_DS_SetDbgFlag();
    
    VulkanEngine engine {};
    
    engine.Init();
    
    engine.Run();
    
    engine.Cleanup();

    return 0;
}