#include "DebugUtils.hpp"

#include "Core/Utilities/Logger.hpp"
#include "Instance.hpp"

namespace {

#if defined(VK_EXT_debug_utils)
vk::Bool32 VKAPI_PTR debugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT             messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*                                       pUserData) {
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
#if defined(_WIN32)
        __debugbreak();
#else
        raise(SIGTRAP);
#endif
    } else if (messageSeverity
               & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else {
        printf("MessageCode is %s & Message is %s \n",
               pCallbackData->pMessageIdName, pCallbackData->pMessage);
    }

    return vk::False;
}
#endif

}  // namespace

namespace IntelliDesign_NS::Vulkan::Core {

DebugUtils::DebugUtils(Instance* instance)
    : pInstance(instance) {
    DBG_LOG_INFO("Vulkan Debug Messenger Created");
}

DebugUtils::~DebugUtils() {
    pInstance->GetHandle().destroy(mDebugMessenger);
}

vk::DebugUtilsMessengerEXT DebugUtils::CreateDebugMessenger() {
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo {};
    messengerInfo
        .setMessageSeverity(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
        .setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
#if defined(VK_EXT_device_address_binding_report)
            vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
#endif
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(&debugMessengerCallback);

    return pInstance->GetHandle().createDebugUtilsMessengerEXT(messengerInfo);
}

}  // namespace IntelliDesign_NS::Vulkan::Core