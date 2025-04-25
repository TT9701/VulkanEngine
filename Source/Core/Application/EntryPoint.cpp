#include "EntryPoint.h"

int main(int argc, char** argv) {
    INTELLI_DS_SetDbgFlag();

    setlocale(LC_ALL, ".utf8");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    auto app = IntelliDesign_NS::Vulkan::Core::CreateApplication({argc, argv});

    app->Run();

    delete app;
    
    return 0;
}