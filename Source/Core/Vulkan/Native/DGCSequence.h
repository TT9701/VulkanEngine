#pragma once

#include "Core/Utilities/Defines.h"
#include "DGCSeqRenderLayout.h"

#include "Core/Vulkan/Manager/PipelineManager.h"
#include "Core/Vulkan/Manager/ShaderManager.h"

namespace IntelliDesign_NS::Vulkan::Core {

class Buffer;

class DGCSeqBase {
    struct PreprocessBuffer {
        uint32_t size;
        vk::Buffer handle;
        vk::DeviceMemory memory;
        vk::DeviceAddress address;
    };

public:
    DGCSeqBase(VulkanContext& context, PipelineManager& pipelineMgr,
               uint32_t sequenceCount, uint32_t maxDrawCount,
               uint32_t maxESObjectCount);

    virtual ~DGCSeqBase();

    uint32_t GetSequenceCount() const;

    PipelineLayout* GetPipelineLayout() const;

    bool IsCompute() const;

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
    PipelineManager& mPipelineMgr;

    uint32_t mMaxSequenceCount;
    uint32_t mMaxESObjectCount;
    uint32_t mMaxDrawCount;

    bool mIsCompute;
    bool mUseExecutionSet;

    Type_STLUnorderedMap_String<uint32_t> mObjectNamesIdxMap;

    vk::IndirectExecutionSetEXT mExecutionSet;

    DGCSeqRenderLayout mLayout {};

    PreprocessBuffer mPreprocessBuffer {};

    bool mExplicitPreprocess {false};
};

class DGCSeq_ESPipeline : public DGCSeqBase {
public:
    DGCSeq_ESPipeline(VulkanContext& context, PipelineManager& pipelineMgr,
                      uint32_t maxSequenceCount, uint32_t maxDrawCount,
                      uint32_t maxPipelineCount);

    DGCSeq_ESPipeline& AddPipeline(const char* name);

    template <class TDGCSeqTemplate>
    void MakeSequenceLayout(const char* pipelineLayoutName,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    void MakeExecutionSet(const char* initialPipelineName);

    void Finalize() override;

    void Execute(vk::CommandBuffer cmd, Buffer const& buffer) override;

private:
    vk::Pipeline mInitialPipeline;
};

class DGCSeq_ESShader : public DGCSeqBase {
public:
    DGCSeq_ESShader(VulkanContext& context, PipelineManager& pipelineMgr,
                    ShaderManager& shaderMgr, uint32_t maxSequenceCount,
                    uint32_t maxDrawCount, uint32_t maxPipelineCount);

    DGCSeq_ESShader& AddShader(ShaderIDInfo const& idInfo);

    template <class TDGCSeqTemplate>
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

    template <class TDGCSeqTemplate>
    void MakeSequenceLayout(ShaderIDInfo const& idInfo,
                            bool unorderedSequence = false,
                            bool explicitPreprocess = false);

    void MakeExecutionSet(ShaderIDInfo const& initialShaderIdInfo);

private:
    ShaderManager& mShaderMgr;

    uint32_t mShaderCount;

    Type_STLVector<vk::ShaderEXT> mInitialShaders {};
};

template <class TDGCSeqTemplate>
void DGCSeq_ESPipeline::MakeSequenceLayout(const char* pipelineLayoutName,
                                           bool unorderedSequence,
                                           bool explicitPreprocess) {
    mLayout.CreateLayout<TDGCSeqTemplate>(
        mContext, mPipelineMgr.GetLayout(pipelineLayoutName), unorderedSequence,
        explicitPreprocess);

    mExplicitPreprocess = explicitPreprocess;
    mIsCompute = TDGCSeqTemplate::_IsCompute_;
    mUseExecutionSet = TDGCSeqTemplate::_UseExecutionSet_;
}

template <class TDGCSeqTemplate>
void DGCSeq_ESShader::MakeSequenceLayout(
    Type_STLVector<ShaderIDInfo> const& idInfos, bool unorderedSequence,
    bool explicitPreprocess) {
    mUseExecutionSet = TDGCSeqTemplate::_UseExecutionSet_ ? true : false;
    mIsCompute = TDGCSeqTemplate::_IsCompute_ ? true : false;
    mShaderCount = TDGCSeqTemplate::_IsCompute_ ? 1 : 3;
    mInitialShaders.resize(mShaderCount);

    VE_ASSERT(mShaderCount == idInfos.size(),
              "Invalid number of shaders for the sequence.");

    if (idInfos.size() == 1) {
        MakeSequenceLayout<TDGCSeqTemplate>(idInfos[0], unorderedSequence,
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

    auto program = mShaderMgr.CreateProgram(
        typeid(TDGCSeqTemplate).name(), shaderObjects[TaskShaderIdx],
        shaderObjects[MeshShaderIdx], shaderObjects[FragmentShaderIdx]);

    auto pipelineLayout =
        mPipelineMgr.CreateLayout(typeid(TDGCSeqTemplate).name(), program);

    mLayout.CreateLayout<TDGCSeqTemplate>(
        mContext, pipelineLayout, unorderedSequence, explicitPreprocess);
}

template <class TDGCSeqTemplate>
void DGCSeq_ESShader::MakeSequenceLayout(ShaderIDInfo const& idInfo,
                                         bool unorderedSequence,
                                         bool explicitPreprocess) {
    VE_ASSERT(idInfo.stage == vk::ShaderStageFlagBits::eCompute,
              "Invalid shader stage for the sequence.");

    auto shader = mShaderMgr.GetShaderObject(idInfo.name, idInfo.stage,
                                             idInfo.macros, idInfo.entry);

    auto program =
        mShaderMgr.CreateProgram(typeid(TDGCSeqTemplate).name(), shader);

    auto pipelineLayout =
        mPipelineMgr.CreateLayout(typeid(TDGCSeqTemplate).name(), program);

    mLayout.CreateLayout<TDGCSeqTemplate>(
        mContext, pipelineLayout, unorderedSequence, explicitPreprocess);
}

}  // namespace IntelliDesign_NS::Vulkan::Core
