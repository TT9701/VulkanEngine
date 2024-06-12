#include "Core/Engine.hpp"

int main() {
    VulkanEngine engine {};

    engine.Init();

    engine.Run();

    engine.Cleanup();

    return 0;
}