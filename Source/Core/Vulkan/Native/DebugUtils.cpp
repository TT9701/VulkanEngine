#include "DebugUtils.h"

#include "Core/Utilities/Logger.h"
#include "Instance.h"

namespace {

#if defined(VK_EXT_debug_utils)
vk::Bool32 VKAPI_PTR debugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("[ERROR]: MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
#if defined(_WIN32)
        __debugbreak();
#else
        raise(SIGTRAP);
#endif
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        printf("[WARNING]: MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        printf("[INFO]: MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        printf("[VERBOSE]: MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    }

    return vk::False;
}
#endif

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

DebugUtils::DebugUtils(Instance& instance) : mInstance(instance) {
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo {};
    messengerInfo
        .setMessageSeverity(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            /*| vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose*/)
        .setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
#if defined(VK_EXT_device_address_binding_report)
            vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
#endif
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(&debugMessengerCallback);

    mHandle = instance.GetHandle().createDebugUtilsMessengerEXT(messengerInfo);

    DBG_LOG_INFO("Vulkan Debug Messenger Created");
}

DebugUtils::~DebugUtils() {
    mInstance.GetHandle().destroy(mHandle);
}

}  // namespace IntelliDesign_NS::Vulkan::Core