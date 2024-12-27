#include "Swapchain.h"

#include "Core/Utilities/VulkanUtilities.h"
#include "Core/Vulkan/Manager/RenderFrame.h"
#include "Core/Vulkan/Manager/VulkanContext.h"
#include "RenderResource.h"

namespace IntelliDesign_NS::Vulkan::Core {

Swapchain::Swapchain(VulkanContext& ctx, vk::Format format,
                     vk::Extent2D extent2D)
    : mContex(ctx), mFormat(format), mSwapchain(RecreateSwapchain(extent2D)) {
    SetSwapchainImages();

    DBG_LOG_INFO(
        "Vulkan Swapchain Created. PresentMode: %s. \n\t\t\t    "
        "Swapchain Image Count: %d",
        vk::to_string(mCreateInfo.presentMode).c_str(), mImages.size());
}

Swapchain::~Swapchain() {
    mContex.GetDevice()->destroy(mSwapchain);
}

uint32_t Swapchain::AcquireNextImageIndex(RenderFrame& frame) {
    frame.Reset();

    VK_CHECK(mContex.GetDevice()->acquireNextImageKHR(
        mSwapchain, WAIT_NEXT_IMAGE_TIME_OUT,
        frame.GetPresentFinishedSemaphore().GetHandle(), VK_NULL_HANDLE,
        &mCurrentImageIndex));

    return mCurrentImageIndex;
}

void Swapchain::Present(RenderFrame& frame, vk::Queue queue) {
    auto sem = frame.GetSwapchainPresentSemaphore().GetHandle();

    vk::PresentInfoKHR presentInfo {};
    presentInfo.setSwapchains(mSwapchain)
        .setWaitSemaphores(sem)
        .setImageIndices(mCurrentImageIndex);

    VK_CHECK(queue.presentKHR(presentInfo));
}

vk::SwapchainKHR Swapchain::RecreateSwapchain(vk::Extent2D extent,
                                              vk::SwapchainKHR old) {
    mExtent2D = extent;
    mCreateInfo.setSurface(mContex.GetSurface().GetHandle())
        .setMinImageCount(3u)
        .setImageFormat(mFormat)
        .setImageExtent(mExtent2D)
        .setImageArrayLayers(1u)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment
                       | vk::ImageUsageFlagBits::eTransferDst)
        .setPresentMode(vk::PresentModeKHR::eMailbox)
        .setClipped(vk::True)
        .setOldSwapchain(old);

    auto handle = mContex.GetDevice()->createSwapchainKHR(mCreateInfo);
    mContex.SetName(handle, "Default Swapchain");
    return handle;
}

vk::SwapchainKHR Swapchain::GetHandle() const {
    return mSwapchain;
}

vk::Image Swapchain::GetImageHandle(uint32_t index) const {
    return mImages[index].GetTexHandle();
}

vk::ImageView Swapchain::GetImageViewHandle(uint32_t index) const {
    return mImages[index].GetTexViewHandle("Color-Whole");
}

vk::RenderingAttachmentInfo Swapchain::GetColorAttachmentInfo(
    uint32_t index) const {
    vk::RenderingAttachmentInfo info {};
    info.setImageView(mImages[index].GetTexViewHandle("Color-Whole"))
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    return info;
}

vk::Format Swapchain::GetFormat() const {
    return mFormat;
}

vk::Extent2D Swapchain::GetExtent2D() const {
    return mExtent2D;
}

uint32_t Swapchain::GetImageCount() const {
    return mImages.size();
}

uint32_t Swapchain::GetCurrentImageIndex() const {
    return mCurrentImageIndex;
}

uint32_t Swapchain::GetPrevImageIndex() const {
    return (mCurrentImageIndex - 1) % 3;
}

RenderResource const& Swapchain::GetCurrentImage() const {
    return mImages[mCurrentImageIndex];
}

RenderResource const& Swapchain::GetImage(uint32_t index) const {
    return mImages[index];
}

Type_STLVector<RenderResource> const& Swapchain::GetImages() const {
    return mImages;
}

void Swapchain::Resize(vk::Extent2D extent) {
    mContex.GetDevice()->waitIdle();
    auto newSP = RecreateSwapchain(extent, mSwapchain);
    mContex.GetDevice()->destroy(mSwapchain);
    mSwapchain = newSP;
    SetSwapchainImages();
}

void Swapchain::SetSwapchainImages() {
    mImages.clear();
    auto images = mContex.GetDevice()->getSwapchainImagesKHR(mSwapchain);
    mImages.reserve(images.size());

    ::std::string nameBase {"_Swapchain_"};
    for (uint32_t i = 0; i < images.size(); ++i) {
        auto& image = mImages.emplace_back(
            mContex, images[i], RenderResource::Type::Texture2D, mFormat,
            vk::Extent3D {mExtent2D.width, mExtent2D.height, 1}, 1, 1);

        mContex.SetName(image.GetTexHandle(), "Swapchain Images");

        image.CreateTexView("Color-Whole", vk::ImageAspectFlagBits::eColor);

        image.SetName((nameBase + ::std::to_string(i)).c_str());
    }
}

namespace {
template <class T>
constexpr const T& Clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

vk::Extent2D ChooseExtent(vk::Extent2D requestExtent,
                          vk::Extent2D const& minImageExtent,
                          vk::Extent2D const& maxImageExtent,
                          vk::Extent2D const& currentExtent) {
    if (currentExtent.width == 0xFFFFFFFF) {
        return requestExtent;
    }

    if (requestExtent.width < 1 || requestExtent.height < 1) {
        DBG_LOG_INFO(
            "(HPPSwapchain) Image extent (%d, %d) not supported. Selecting "
            "(%d, %d).",
            requestExtent.width, requestExtent.height, currentExtent.width,
            currentExtent.height);
        return currentExtent;
    }

    requestExtent.width =
        Clamp(requestExtent.width, minImageExtent.width, maxImageExtent.width);
    requestExtent.height = Clamp(requestExtent.height, minImageExtent.height,
                                 maxImageExtent.height);

    return requestExtent;
}

vk::PresentModeKHR ChoosePresentMode(
    vk::PresentModeKHR requestPresentMode,
    std::vector<vk::PresentModeKHR> const& availablePresentModes,
    std::vector<vk::PresentModeKHR> const& presentModePriorityList) {
    auto const presentModeIt =
        std::find(availablePresentModes.begin(), availablePresentModes.end(),
                  requestPresentMode);
    if (presentModeIt == availablePresentModes.end()) {
        auto const chosenPresentModeIt = std::find_if(
            presentModePriorityList.begin(), presentModePriorityList.end(),
            [&availablePresentModes](vk::PresentModeKHR presentMode) {
                return std::find(availablePresentModes.begin(),
                                 availablePresentModes.end(), presentMode)
                    != availablePresentModes.end();
            });

        // If nothing found, always default to FIFO
        vk::PresentModeKHR const chosenPresentMode =
            (chosenPresentModeIt != presentModePriorityList.end())
                ? *chosenPresentModeIt
                : vk::PresentModeKHR::eFifo;

        DBG_LOG_INFO(
            "(HPPSwapchain) Present mode '%s' not supported. Selecting '%s'.",
            vk::to_string(requestPresentMode).c_str(),
            vk::to_string(chosenPresentMode).c_str());
        return chosenPresentMode;
    } else {
        DBG_LOG_INFO("(HPPSwapchain) Present mode selected: %s",
                     to_string(requestPresentMode).c_str());
        return requestPresentMode;
    }
}

vk::SurfaceFormatKHR ChooseSurfaceFormat(
    const vk::SurfaceFormatKHR requestedSurfaceFormat,
    const std::vector<vk::SurfaceFormatKHR>& availableSurfaceFormats,
    const std::vector<vk::SurfaceFormatKHR>& surfaceFormatPriorityList) {
    auto const surfaceFormatIt =
        std::find(availableSurfaceFormats.begin(),
                  availableSurfaceFormats.end(), requestedSurfaceFormat);

    if (surfaceFormatIt == availableSurfaceFormats.end()) {
        auto const chosenSurfaceFormatIt = std::find_if(
            surfaceFormatPriorityList.begin(), surfaceFormatPriorityList.end(),
            [&availableSurfaceFormats](vk::SurfaceFormatKHR surfaceFormat) {
                return std::find(availableSurfaceFormats.begin(),
                                 availableSurfaceFormats.end(), surfaceFormat)
                    != availableSurfaceFormats.end();
            });

        // If nothing found, default to the first available format
        vk::SurfaceFormatKHR const& chosenSurfaceFormat =
            (chosenSurfaceFormatIt != surfaceFormatPriorityList.end())
                ? *chosenSurfaceFormatIt
                : availableSurfaceFormats[0];

        DBG_LOG_INFO(
            "(HPPSwapchain) Surface format (%s) not supported. Selecting (%s).",
            (vk::to_string(requestedSurfaceFormat.format) + ", "
             + vk::to_string(requestedSurfaceFormat.colorSpace))
                .c_str(),
            (vk::to_string(chosenSurfaceFormat.format) + ", "
             + vk::to_string(chosenSurfaceFormat.colorSpace))
                .c_str());
        return chosenSurfaceFormat;
    } else {
        DBG_LOG_INFO("(HPPSwapchain) Surface format selected: %s",
                     (vk::to_string(requestedSurfaceFormat.format) + ", "
                      + vk::to_string(requestedSurfaceFormat.colorSpace))
                         .c_str());
        return requestedSurfaceFormat;
    }
}

vk::SurfaceTransformFlagBitsKHR ChooseTransform(
    vk::SurfaceTransformFlagBitsKHR requestTransform,
    vk::SurfaceTransformFlagsKHR supportedTransform,
    vk::SurfaceTransformFlagBitsKHR currentTransform) {
    if (requestTransform & supportedTransform) {
        return requestTransform;
    }

    DBG_LOG_INFO(
        "(HPPSwapchain) Surface transform '%s' not supported. Selecting '%s'.",
        vk::to_string(requestTransform).c_str(),
        vk::to_string(currentTransform).c_str());
    return currentTransform;
}

vk::CompositeAlphaFlagBitsKHR ChooseCompositeAlpha(
    vk::CompositeAlphaFlagBitsKHR requestCompositeAlpha,
    vk::CompositeAlphaFlagsKHR supportedCompositeAlpha) {
    if (requestCompositeAlpha & supportedCompositeAlpha) {
        return requestCompositeAlpha;
    }

    static const std::vector<vk::CompositeAlphaFlagBitsKHR>
        CompositeAlphaPriorityList = {
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
            vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
            vk::CompositeAlphaFlagBitsKHR::eInherit};

    auto const chosenCompositeAlphaIt = std::find_if(
        CompositeAlphaPriorityList.begin(), CompositeAlphaPriorityList.end(),
        [&supportedCompositeAlpha](
            vk::CompositeAlphaFlagBitsKHR compositeAlpha) {
            return compositeAlpha & supportedCompositeAlpha;
        });
    if (chosenCompositeAlphaIt == CompositeAlphaPriorityList.end()) {
        throw std::runtime_error("No compatible composite alpha found.");
    } else {
        DBG_LOG_INFO(
            "(HPPSwapchain) Composite alpha '%s' not supported. Selecting '%s.",
            vk::to_string(requestCompositeAlpha).c_str(),
            vk::to_string(*chosenCompositeAlphaIt).c_str());
        return *chosenCompositeAlphaIt;
    }
}

bool ValidateFormatFeature(vk::ImageUsageFlagBits imageUsage,
                           vk::FormatFeatureFlags supportedFeatures) {
    return (imageUsage != vk::ImageUsageFlagBits::eStorage)
        || (supportedFeatures & vk::FormatFeatureFlagBits::eStorageImage);
}

std::set<vk::ImageUsageFlagBits> ChooseImageUsage(
    const std::set<vk::ImageUsageFlagBits>& requestedImageUsageFlags,
    vk::ImageUsageFlags supportedImageUsage,
    vk::FormatFeatureFlags supportedFeatures) {
    std::set<vk::ImageUsageFlagBits> validatedImageUsageFlags;
    for (auto flag : requestedImageUsageFlags) {
        if ((flag & supportedImageUsage)
            && ValidateFormatFeature(flag, supportedFeatures)) {
            validatedImageUsageFlags.insert(flag);
        } else {
            DBG_LOG_INFO(
                "(HPPSwapchain) Image usage (%s) requested but not supported.",
                vk::to_string(flag).c_str());
        }
    }

    if (validatedImageUsageFlags.empty()) {
        // Pick the first format from list of defaults, if supported
        static const std::vector<vk::ImageUsageFlagBits>
            imageUsagePriorityList = {vk::ImageUsageFlagBits::eColorAttachment,
                                      vk::ImageUsageFlagBits::eStorage,
                                      vk::ImageUsageFlagBits::eSampled,
                                      vk::ImageUsageFlagBits::eTransferDst};

        auto const priorityListIt = std::find_if(
            imageUsagePriorityList.begin(), imageUsagePriorityList.end(),
            [&supportedImageUsage, &supportedFeatures](auto const imageUsage) {
                return (
                    (imageUsage & supportedImageUsage)
                    && ValidateFormatFeature(imageUsage, supportedFeatures));
            });
        if (priorityListIt != imageUsagePriorityList.end()) {
            validatedImageUsageFlags.insert(*priorityListIt);
        }
    }

    if (validatedImageUsageFlags.empty()) {
        throw std::runtime_error("No compatible image usage found.");
    } else {
        // Log image usage flags used
        std::string usageList;
        for (vk::ImageUsageFlagBits imageUsage : validatedImageUsageFlags) {
            usageList += to_string(imageUsage) + " ";
        }
        DBG_LOG_INFO("(HPPSwapchain) Image usage flags: %s", usageList.c_str());
    }

    return validatedImageUsageFlags;
}

vk::ImageUsageFlags CompositeImageFlags(
    std::set<vk::ImageUsageFlagBits>& imageUsageFlags) {
    vk::ImageUsageFlags imageUsage;
    for (auto flag : imageUsageFlags) {
        imageUsage |= flag;
    }
    return imageUsage;
}

}  // namespace

HPPSwapchain::HPPSwapchain(HPPSwapchain& old, vk::Extent2D const& extent)
    : HPPSwapchain {old.mContext,
                    old.mProperties.presentMode,
                    old.mPresentModePriorityList,
                    old.mSurfaceFormatPriorityList,
                    extent,
                    old.mProperties.imageCount,
                    old.mProperties.preTransform,
                    old.mImageUsageFlag,
                    old.GetHandle()} {}

HPPSwapchain::HPPSwapchain(
    VulkanContext& context, vk::PresentModeKHR presentMode,
    std::vector<vk::PresentModeKHR> const& presentModePriorityList,
    std::vector<vk::SurfaceFormatKHR> const& surfaceFormatPriorityList,
    vk::Extent2D const& extent, uint32_t imageCount,
    vk::SurfaceTransformFlagBitsKHR transform,
    std::set<vk::ImageUsageFlagBits> const& imageUsageFlags,
    vk::SwapchainKHR old)
    : mContext(context) {
    this->mPresentModePriorityList = presentModePriorityList;
    this->mSurfaceFormatPriorityList = surfaceFormatPriorityList;

    std::vector<vk::SurfaceFormatKHR> surfaceFormats =
        context.GetPhysicalDevice()->getSurfaceFormatsKHR(
            context.GetSurface().GetHandle());
    DBG_LOG_INFO("Surface supports the following surface formats:");
    for (auto& surfaceFormat : surfaceFormats) {
        DBG_LOG_INFO("  \t%s", (vk::to_string(surfaceFormat.format) + ", "
                                + vk::to_string(surfaceFormat.colorSpace))
                                   .c_str());
    }

    std::vector<vk::PresentModeKHR> presentModes =
        context.GetPhysicalDevice()->getSurfacePresentModesKHR(
            context.GetSurface().GetHandle());
    DBG_LOG_INFO("Surface supports the following present modes:");
    for (auto& mode : presentModes) {
        DBG_LOG_INFO("  \t{}", to_string(mode).c_str());
    }

    vk::SurfaceCapabilitiesKHR const surfaceCapabilities =
        context.GetPhysicalDevice()->getSurfaceCapabilitiesKHR(
            context.GetSurface().GetHandle());

    mProperties.oldSwapchain = old;
    mProperties.imageCount =
        Clamp(imageCount, surfaceCapabilities.minImageCount,
              surfaceCapabilities.maxImageCount
                  ? surfaceCapabilities.maxImageCount
                  : std::numeric_limits<uint32_t>::max());
    mProperties.extent = ChooseExtent(
        extent, surfaceCapabilities.minImageExtent,
        surfaceCapabilities.maxImageExtent, surfaceCapabilities.currentExtent);
    mProperties.surfaceFormat = ChooseSurfaceFormat(
        mProperties.surfaceFormat, surfaceFormats, surfaceFormatPriorityList);
    mProperties.arrayLayers = 1;

    vk::FormatProperties const formatProperties =
        context.GetPhysicalDevice()->getFormatProperties(
            mProperties.surfaceFormat.format);
    this->mImageUsageFlag = ChooseImageUsage(
        imageUsageFlags, surfaceCapabilities.supportedUsageFlags,
        formatProperties.optimalTilingFeatures);

    mProperties.imageUsage = CompositeImageFlags(this->mImageUsageFlag);
    mProperties.preTransform =
        ChooseTransform(transform, surfaceCapabilities.supportedTransforms,
                        surfaceCapabilities.currentTransform);
    mProperties.compositeAlpha =
        ChooseCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eInherit,
                             surfaceCapabilities.supportedCompositeAlpha);
    mProperties.presentMode =
        ChoosePresentMode(presentMode, presentModes, presentModePriorityList);

    vk::SwapchainCreateInfoKHR const createInfo(
        {}, context.GetSurface().GetHandle(), mProperties.imageCount,
        mProperties.surfaceFormat.format, mProperties.surfaceFormat.colorSpace,
        mProperties.extent, mProperties.arrayLayers, mProperties.imageUsage, {},
        {}, mProperties.preTransform, mProperties.compositeAlpha,
        mProperties.presentMode, {}, mProperties.oldSwapchain);

    mHandle = context.GetDevice()->createSwapchainKHR(createInfo);

    mImages = context.GetDevice()->getSwapchainImagesKHR(mHandle);
}

HPPSwapchain::HPPSwapchain(HPPSwapchain&& other) noexcept
    : mContext {other.mContext},
      mHandle {std::exchange(other.mHandle, nullptr)},
      mImages {std::exchange(other.mImages, {})},
      mProperties {std::exchange(other.mProperties, {})},
      mPresentModePriorityList {
          std::exchange(other.mPresentModePriorityList, {})},
      mSurfaceFormatPriorityList {
          std::exchange(other.mSurfaceFormatPriorityList, {})},
      mImageUsageFlag {std::move(other.mImageUsageFlag)} {}

HPPSwapchain::~HPPSwapchain() {
    if (mHandle) {
        mContext.GetDevice()->destroy(mHandle);
    }
}

bool HPPSwapchain::IsValid() const {
    return !!mHandle;
}

vk::SwapchainKHR HPPSwapchain::GetHandle() const {
    return mHandle;
}

std::pair<vk::Result, uint32_t> HPPSwapchain::AcquireNextImage(
    vk::Semaphore imageAcquiredSemaphore, vk::Fence fence) const {
    vk::ResultValue<uint32_t> rv = mContext.GetDevice()->acquireNextImageKHR(
        mHandle, WAIT_NEXT_IMAGE_TIME_OUT, imageAcquiredSemaphore, fence);
    return std::make_pair(rv.result, rv.value);
}

const vk::Extent2D& HPPSwapchain::GetExtent() const {
    return mProperties.extent;
}

vk::Format HPPSwapchain::GetFormat() const {
    return mProperties.surfaceFormat.format;
}

const std::vector<vk::Image>& HPPSwapchain::GetImages() const {
    return mImages;
}

vk::ImageUsageFlags HPPSwapchain::GetUsage() const {
    return mProperties.imageUsage;
}

vk::PresentModeKHR HPPSwapchain::GetPresentMode() const {
    return mProperties.presentMode;
}

}  // namespace IntelliDesign_NS::Vulkan::Core