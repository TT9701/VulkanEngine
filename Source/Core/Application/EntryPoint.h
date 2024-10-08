#pragma once

#include <Win32/DbgMemLeak.h>

#include "Application.hpp"

extern IntelliDesign_NS::Vulkan::Core::Application*
IntelliDesign_NS::Vulkan::Core::CreateApplication(
    ApplicationCommandLineArgs args);

int main(int argc, char** argv) {
    INTELLI_DS_SetDbgFlag();

    auto app = IntelliDesign_NS::Vulkan::Core::CreateApplication({argc, argv});

    app->Run();

    delete app;
}