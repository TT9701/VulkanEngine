#pragma once

#include "Core/Utilities/Defines.h"
#include "DGCSequenceLayout.h"

#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/ShaderManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Buffer;

class DGCSequenceBase {
    struct PreprocessBuffer {
        uint32_t size;
        vk::Buffer handle;
        vk::DeviceMemory memory;
        vk::DeviceAddress address;
    };

public:
    DGCSequenceBase(VulkanContext& context, uint32_t sequenceCount,
                    uint32_t maxDrawCount, uint32_t maxESObjectCount);

    virtual ~DGCSequenceBase();

    uint32_t GetSequenceCount() const;

    virtual void Finalize();

    virtual void Execute(vk::CommandBuffer cmd, Buffer const& buffer) = 0;

protected:
    vk::PipelineBindPoint GetPipelineBindPoint();

    Type_STLVector<vk::ShaderStageFlagBits> GetStageFlagBitArray();

    vk::ShaderStageFlags GetStageFlags();

    void InsertInitialObjectToMap(const char* name, uint32_t idx);

    void Preprocess(::std::variant<vk::Pipeline, Type_STLVector<vk::ShaderEXT>>
                        initialObjs);

protected:
    VulkanContext& mContext;

    uint32_t mMaxSequenceCount;
    uint32_t mMaxESObjectCount;
    uint32_t mMaxDrawCount;

    bool mIsCompute;
    bool mUseExecutionSet;

    Type_STLUnorderedMap_String<uint32_t> mObjectNamesIdxMap;

    vk::IndirectExecutionSetEXT mExecutionSet;

    Type_UniquePtr<SequenceLayout> mLayout;

    PreprocessBuffer mPreprocessBuffer {};

    bool mExplicitPreprocess {false};
};

class DGCSequence_ESPipeline : public DGCSequenceBase {
public:
    DGCSequence_ESPipeline(VulkanContext& context, PipelineManager& pipelineMgr,
                           uint32_t maxSequenceCount, uint32_t maxDrawCount,
                           uint32_t maxPipelineCount);

    DGCSequence_ESPipeline& AddPipeline(const char* name);

    template <class TDGCSequenceTemplate>
    void MakeSequenceLayout(const char* pipelineLayoutName,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    void MakeExecutionSet(const char* initialPipelineName);

    void Finalize() override;

    void Execute(vk::CommandBuffer cmd, Buffer const& buffer) override;

private:
    PipelineManager& mPipelineMgr;

    vk::Pipeline mInitialPipeline;
};

class DGCSequence_ESShader : public DGCSequenceBase {
public:
    DGCSequence_ESShader(VulkanContext& context, ShaderManager& shaderMgr,
                         uint32_t maxSequenceCount, uint32_t maxDrawCount,
                         uint32_t maxPipelineCount);

    DGCSequence_ESShader& AddShader(ShaderIDInfo const& idInfo);

    template <class TDGCSequenceTemplate>
    void MakeSequenceLayout(Type_STLVector<ShaderIDInfo> const& idInfos,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    void MakeExecutionSet(
        Type_STLVector<ShaderIDInfo> const& initialShaderIdInfos);

    void Finalize() override;

    void Execute(vk::CommandBuffer cmd, Buffer const& buffer) override;

private:
    enum { TaskShaderIdx = 0, MeshShaderIdx = 1, FragmentShaderIdx = 2 };

    uint32_t ShaderStageToIdx(vk::ShaderStageFlagBits stage) const;

    template <class TDGCSequenceTemplate>
    void MakeSequenceLayout(ShaderIDInfo const& idInfo,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    void MakeExecutionSet(ShaderIDInfo const& initialShaderIdInfo);

private:
    ShaderManager& mShaderMgr;

    uint32_t mShaderCount;

    Type_STLVector<vk::ShaderEXT> mInitialShaders {};
};

template <class TDGCSequenceTemplate>
void DGCSequence_ESPipeline::MakeSequenceLayout(const char* pipelineLayoutName,
                                                bool unorderedSequence,
                                                bool explicitPreprocess) {
    auto pipelineLayout = mPipelineMgr.GetLayoutHandle(pipelineLayoutName);

    mLayout = CreateLayout<TDGCSequenceTemplate>(
        mContext, pipelineLayout, unorderedSequence, explicitPreprocess);

    mExplicitPreprocess = explicitPreprocess;
    mIsCompute = TDGCSequenceTemplate::_IsCompute_;
    mUseExecutionSet = TDGCSequenceTemplate::_UseExecutionSet_;
}

template <class TDGCSequenceTemplate>
void DGCSequence_ESShader::MakeSequenceLayout(
    Type_STLVector<ShaderIDInfo> const& idInfos, bool unorderedSequence,
    bool explicitPreprocess) {
    mUseExecutionSet = TDGCSequenceTemplate::_UseExecutionSet_ ? true : false;
    mIsCompute = TDGCSequenceTemplate::_IsCompute_ ? true : false;
    mShaderCount = TDGCSequenceTemplate::_IsCompute_ ? 1 : 3;
    mInitialShaders.resize(mShaderCount);

    VE_ASSERT(mShaderCount == idInfos.size(),
              "Invalid number of shaders for the sequence.");

    if (idInfos.size() == 1) {
        MakeSequenceLayout<TDGCSequenceTemplate>(idInfos[0], unorderedSequence,
                                                 explicitPreprocess);
        return;
    }

    Type_STLVector<ShaderObject*> shaderObjects(3, nullptr);
    for (uint32_t i = 0; i < idInfos.size(); ++i) {
        auto idx = ShaderStageToIdx(idInfos[i].stage);
        if (shaderObjects[idx] != nullptr) {
            throw std::runtime_error(
                "Shader stage already assigned to another shader.");
        } else {
            shaderObjects[idx] =
                mShaderMgr.GetShaderObject(idInfos[i].name, idInfos[i].stage,
                                           idInfos[i].macros, idInfos[i].entry);
        }
    }

    auto program = ShaderManager::MakeTempProgram(
        shaderObjects[TaskShaderIdx], shaderObjects[MeshShaderIdx],
        shaderObjects[FragmentShaderIdx]);

    auto descLayout = program.GetCombinedDescLayoutHandles();
    auto pcRange = program.GetPCRanges()[0];

    mLayout = CreateLayout<TDGCSequenceTemplate>(
        mContext, descLayout, pcRange, unorderedSequence, explicitPreprocess);
}

template <class TDGCSequenceTemplate>
void DGCSequence_ESShader::MakeSequenceLayout(ShaderIDInfo const& idInfo,
                                              bool unorderedSequence,
                                              bool explicitPreprocess) {
    auto shader = mShaderMgr.GetShaderObject(idInfo.name, idInfo.stage,
                                             idInfo.macros, idInfo.entry);
    const auto descLayout = shader->GetDescLayoutHandles();
    const auto pcRange = shader->GetPushContantData();

    mLayout = CreateLayout<TDGCSequenceTemplate>(
        mContext, descLayout, pcRange, unorderedSequence, explicitPreprocess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core
