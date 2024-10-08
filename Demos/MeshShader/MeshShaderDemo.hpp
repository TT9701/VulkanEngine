#pragma once

#include "Core/Application/Application.hpp"
#include "Core/Application/EntryPoint.h"

namespace IDNSVC = IntelliDesign_NS::Vulkan::Core;

class Demo : public IDNSVC::Application {
public:
    Demo(IDNSVC::ApplicationSpecification const& spec) : Application(spec) {}
};

VE_CREATE_APPLICATION(Demo, 1600, 900);