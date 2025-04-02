#include "DGCSequence.h"

#include "Buffer.h"

namespace IntelliDesign_NS::Vulkan::Core {

DGCSeqBase::DGCSeqBase(VulkanContext& context, PipelineManager& pipelineMgr,
                       uint32_t sequenceCount, uint32_t maxDrawCount,
                       uint32_t maxESObjectCount)
    : mContext(context),
      mPipelineMgr(pipelineMgr),
      mMaxESObjectCount(maxESObjectCount),
      mMaxSequenceCount(sequenceCount),
      mMaxDrawCount(maxDrawCount) {}

DGCSeqBase::~DGCSeqBase() {
    if (mExecutionSet != VK_NULL_HANDLE)
        mContext.GetDevice()->destroy(mExecutionSet);

    if (mPreprocessBuffer.handle != VK_NULL_HANDLE)
        mContext.GetDevice()->destroy(mPreprocessBuffer.handle);

    if (mPreprocessBuffer.memory != VK_NULL_HANDLE)
        mContext.GetDevice()->freeMemory(mPreprocessBuffer.memory);
}

uint32_t DGCSeqBase::GetSequenceCount() const {
    return mMaxSequenceCount;
}

PipelineLayout const* DGCSeqBase::GetPipelineLayout() const {
    return mLayout.GetPipelineLayout();
}

DGCSeqBase::SequenceDataBufferPool* DGCSeqBase::GetBufferPool() const {
    return mSeqDataBufPool.get();
}

bool DGCSeqBase::IsCompute() const {
    return mIsCompute;
}

void DGCSeqBase::InsertInitialObjectToMap(const char* name, uint32_t idx) {
    if (mObjectNamesIdxMap.contains(name)) {
        if (mObjectNamesIdxMap.at(name) != idx) {
            auto it = ::std::find_if(
                mObjectNamesIdxMap.begin(), mObjectNamesIdxMap.end(),
                [&](const auto& pair) { return pair.second == idx; });

            ::std::swap(it->second, mObjectNamesIdxMap.at(name));
        }
    } else {
        for (auto& [name, i] : mObjectNamesIdxMap) {
            if (i >= idx)
                i += 1;
        }
        mObjectNamesIdxMap.emplace(name, idx);
    }
}

void DGCSeqBase::Preprocess(
    ::std::variant<vk::Pipeline, Type_STLVector<vk::ShaderEXT>> initialObjs) {
    vk::GeneratedCommandsMemoryRequirementsInfoEXT memInfo {};
    memInfo.setMaxSequenceCount(mMaxSequenceCount)
        .setMaxDrawCount(mMaxDrawCount)
        .setIndirectCommandsLayout(mLayout.GetHandle());

    vk::GeneratedCommandsPipelineInfoEXT pipelineInfo {};
    vk::GeneratedCommandsShaderInfoEXT shaderInfo {};

    if (!mUseExecutionSet) {
        if (const auto pPipeline = ::std::get_if<vk::Pipeline>(&initialObjs)) {
            pipelineInfo.setPipeline(*pPipeline);
            memInfo.setPNext(&pipelineInfo);

        } else {
            const auto pShaders =
                ::std::get_if<Type_STLVector<vk::ShaderEXT>>(&initialObjs);
            shaderInfo.setShaders(*pShaders);
            memInfo.setPNext(&shaderInfo);
        }
    }

    if (mExecutionSet != VK_NULL_HANDLE) {
        memInfo.setIndirectExecutionSet(mExecutionSet);
    }

    auto memReqs =
        mContext.GetDevice()->getGeneratedCommandsMemoryRequirementsEXT(
            memInfo);

    mPreprocessBuffer.size = memReqs.memoryRequirements.size;

    vk::BufferCreateInfo bufferCreateInfo {};
    vk::BufferUsageFlags2CreateInfoKHR bufferFlags2 {};
    bufferCreateInfo.size = mPreprocessBuffer.size;
    bufferFlags2.usage = vk::BufferUsageFlagBits2::ePreprocessBufferEXT
                       | vk::BufferUsageFlagBits2::eIndirectBuffer
                       | vk::BufferUsageFlagBits2::eShaderDeviceAddress;
    bufferCreateInfo.pNext = &bufferFlags2;

    mPreprocessBuffer.handle =
        mContext.GetDevice()->createBuffer(bufferCreateInfo);

    auto const& memProps = mContext.GetPhysicalDevice().GetMemoryProperties();

    uint32_t memTypeIndex {~0ui32};

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (((memReqs.memoryRequirements.memoryTypeBits & (1 << i)) > 0)
            && (memProps.memoryTypes[i].propertyFlags
                & vk::MemoryPropertyFlagBits::eDeviceLocal)
                   == vk::MemoryPropertyFlagBits::eDeviceLocal) {
            memTypeIndex = i;
            break;
        }
    }

    vk::MemoryAllocateFlagsInfo flagsInfo {
        vk::MemoryAllocateFlagBits::eDeviceAddress};

    auto req = mContext.GetDevice()->getBufferMemoryRequirements(
        mPreprocessBuffer.handle);

    vk::MemoryAllocateInfo allocInfo {req.size, memTypeIndex, &flagsInfo};

    mPreprocessBuffer.memory = mContext.GetDevice()->allocateMemory(allocInfo);

    mContext.GetDevice()->bindBufferMemory(mPreprocessBuffer.handle,
                                           mPreprocessBuffer.memory, 0);

    vk::BufferDeviceAddressInfo deviceAdressInfo {mPreprocessBuffer.handle};
    mPreprocessBuffer.address =
        mContext.GetDevice()->getBufferAddress(deviceAdressInfo);
}

DGCSeq_ESPipeline::DGCSeq_ESPipeline(VulkanContext& context,
                                     PipelineManager& pipelineMgr,
                                     uint32_t maxSequenceCount,
                                     uint32_t maxDrawCount,
                                     uint32_t maxPipelineCount)
    : DGCSeqBase(context, pipelineMgr, maxSequenceCount, maxDrawCount,
                 maxPipelineCount) {}

DGCSeq_ESPipeline& DGCSeq_ESPipeline::AddPipeline(const char* name) {
    uint32_t size = mObjectNamesIdxMap.size();
    VE_ASSERT(size < mMaxESObjectCount,
              "The number of pipelines exceeds the maximum limit.");

    mObjectNamesIdxMap.emplace(name, size);
    return *this;
}

void DGCSeq_ESPipeline::MakeExecutionSet(const char* initialPipelineName) {
    mInitialPipeline = mPipelineMgr.GetPipelineHandle(initialPipelineName);

    if (!mUseExecutionSet)
        return;

    InsertInitialObjectToMap(initialPipelineName, 0);

    vk::IndirectExecutionSetPipelineInfoEXT esPipelineInfo {};
    esPipelineInfo.setInitialPipeline(mInitialPipeline)
        .setMaxPipelineCount(mMaxESObjectCount);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::ePipelines)
        .setInfo(&esPipelineInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetPipelineEXT> writeIES {};
    writeIES.reserve(mObjectNamesIdxMap.size());

    for (const auto& [name, idx] : mObjectNamesIdxMap) {
        vk::WriteIndirectExecutionSetPipelineEXT write {};
        write.setPipeline(mPipelineMgr.GetPipelineHandle(name.c_str()))
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetPipelineEXT(mExecutionSet,
                                                                writeIES);
}

void DGCSeq_ESPipeline::Finalize() {
    Preprocess(mInitialPipeline);

    vk::GeneratedCommandsInfoEXT info {};
    mInfo.setIndirectCommandsLayout(mLayout.GetHandle())
        .setMaxSequenceCount(mMaxSequenceCount)
        .setMaxDrawCount(mMaxDrawCount)
        .setPreprocessAddress(mPreprocessBuffer.address)
        .setPreprocessSize(mPreprocessBuffer.size)
        .setShaderStages(GetStageFlags());

    if (!mUseExecutionSet) {
        mPipelineInfo.setPipeline(mInitialPipeline);
        info.setPNext(&mPipelineInfo);
    }

    if (mExecutionSet != VK_NULL_HANDLE) {
        info.setIndirectExecutionSet(mExecutionSet);
    }
}

void DGCSeq_ESPipeline::Execute(vk::CommandBuffer cmd, Buffer const& buffer) {
    cmd.bindPipeline(GetPipelineBindPoint(), mInitialPipeline);

    mInfo.setIndirectAddress(buffer.GetDeviceAddress())
        .setIndirectAddressSize(buffer.GetStride());

    cmd.executeGeneratedCommandsEXT(mExplicitPreprocess, mInfo);
}

void DGCSeq_ESPipeline::ExecutePrerocess(vk::CommandBuffer cmd,
                                         Buffer const& buffer,
                                         vk::CommandBuffer preprocessCmd) {
    assert(mExplicitPreprocess);

    mInfo.setIndirectAddress(buffer.GetDeviceAddress())
        .setIndirectAddressSize(buffer.GetStride());

    preprocessCmd.preprocessGeneratedCommandsEXT(mInfo, cmd);


}

DGCSeq_ESShader::DGCSeq_ESShader(VulkanContext& context,
                                 PipelineManager& pipelineMgr,
                                 ShaderManager& shaderMgr,
                                 uint32_t maxSequenceCount,
                                 uint32_t maxDrawCount,
                                 uint32_t maxPipelineCount)
    : DGCSeqBase(context, pipelineMgr, maxSequenceCount, maxDrawCount,
                 maxPipelineCount),
      mShaderMgr(shaderMgr) {}

DGCSeq_ESShader& DGCSeq_ESShader::AddShader(ShaderIDInfo const& idInfo) {
    uint32_t size = mObjectNamesIdxMap.size();
    VE_ASSERT(size < mMaxESObjectCount,
              "The number of pipelines exceeds the maximum limit.");

    auto name = ShaderManager::ParseShaderName(idInfo.name, idInfo.stage,
                                               idInfo.macros, idInfo.entry);

    mObjectNamesIdxMap.emplace(name, size);
    return *this;
}

void DGCSeq_ESShader::MakeExecutionSet(
    Type_STLVector<ShaderIDInfo> const& initialShaderIdInfos) {
    VE_ASSERT(mShaderCount == initialShaderIdInfos.size(),
              "Invalid number of shaders for the sequence.");

    if (initialShaderIdInfos.size() == 1) {
        MakeExecutionSet(initialShaderIdInfos[0]);
        return;
    }

    Type_STLVector<ShaderObject*> shaderObjects(3, nullptr);
    for (uint32_t i = 0; i < initialShaderIdInfos.size(); ++i) {
        auto idx = ShaderStageToIdx(initialShaderIdInfos[i].stage);
        if (shaderObjects[idx] != nullptr) {
            throw std::runtime_error(
                "Shader stage already assigned to another shader.");
        } else {
            shaderObjects[idx] = mShaderMgr.GetShaderObject(
                initialShaderIdInfos[i].name, initialShaderIdInfos[i].stage,
                initialShaderIdInfos[i].macros, initialShaderIdInfos[i].entry);
        }
    }

    auto program = ShaderManager::MakeTempProgram(
        shaderObjects[TaskShaderIdx], shaderObjects[MeshShaderIdx],
        shaderObjects[FragmentShaderIdx]);

    Type_STLVector<Type_STLVector<vk::DescriptorSetLayout>> descLayouts(3);
    for (auto const& shaderObjcet : shaderObjects) {
        if (shaderObjcet) {
            descLayouts.push_back(shaderObjcet->GetDescLayoutHandles());
        } else {
            throw std::runtime_error(
                "Shader is not assigned to the shader stage.");
        }
    }

    Type_STLVector<vk::IndirectExecutionSetShaderLayoutInfoEXT> layoutInfos(3);
    for (uint32_t i = 0; i < layoutInfos.size(); ++i) {
        layoutInfos[i].setSetLayouts(descLayouts[i]);
    }

    auto pcRange = program.GetPCRanges()[0];

    for (uint32_t i = 0; i < shaderObjects.size(); ++i) {
        if (shaderObjects[i]) {
            mInitialShaders[i] = shaderObjects[i]->GetHandle();
        } else {
            throw std::runtime_error(
                "Shader is not assigned to the shader stage.");
        }
    }

    for (uint32_t i = 0; i < initialShaderIdInfos.size(); ++i) {
        const auto name = ShaderManager::ParseShaderName(
            initialShaderIdInfos[i].name, initialShaderIdInfos[i].stage,
            initialShaderIdInfos[i].macros, initialShaderIdInfos[i].entry);
        const auto idx = ShaderStageToIdx(initialShaderIdInfos[i].stage);
        InsertInitialObjectToMap(name.c_str(), idx);
    }

    if (!mUseExecutionSet)
        return;

    vk::IndirectExecutionSetShaderInfoEXT esShaderInfo {};
    esShaderInfo.setPushConstantRanges(pcRange)
        .setMaxShaderCount(mMaxESObjectCount)
        .setShaderCount(3)
        .setInitialShaders(mInitialShaders)
        .setSetLayoutInfos(layoutInfos);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::eShaderObjects)
        .setInfo(&esShaderInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetShaderEXT> writeIES {};
    writeIES.reserve(mObjectNamesIdxMap.size());

    for (const auto& [name, idx] : mObjectNamesIdxMap) {
        vk::WriteIndirectExecutionSetShaderEXT write {};
        write.setShader(mShaderMgr.GetShaderObject(name.c_str())->GetHandle())
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetShaderEXT(mExecutionSet,
                                                              writeIES);
}

void DGCSeq_ESShader::Finalize() {
    Preprocess(mInitialShaders);

    mInfo.setIndirectCommandsLayout(mLayout.GetHandle())
        .setMaxSequenceCount(mMaxSequenceCount)
        .setMaxDrawCount(mMaxDrawCount)
        .setPreprocessAddress(mPreprocessBuffer.address)
        .setPreprocessSize(mPreprocessBuffer.size)
        .setShaderStages(GetStageFlags());

    if (!mUseExecutionSet) {
        mShaderInfo.setShaders(mInitialShaders);
        mInfo.setPNext(&mShaderInfo);
    }

    if (mExecutionSet != VK_NULL_HANDLE) {
        mInfo.setIndirectExecutionSet(mExecutionSet);
    }
}

void DGCSeq_ESShader::Execute(vk::CommandBuffer cmd, Buffer const& buffer) {
    cmd.bindShadersEXT(GetStageFlagBitArray(), mInitialShaders);

    mInfo.setIndirectAddress(buffer.GetDeviceAddress())
        .setIndirectAddressSize(buffer.GetStride());

    cmd.executeGeneratedCommandsEXT(mExplicitPreprocess, mInfo);
}

void DGCSeq_ESShader::ExecutePrerocess(vk::CommandBuffer cmd,
                                       Buffer const& buffer,
                                       vk::CommandBuffer preprocessCmd) {
    assert(mExplicitPreprocess);

    mInfo.setIndirectAddress(buffer.GetDeviceAddress())
        .setIndirectAddressSize(buffer.GetStride());

    preprocessCmd.preprocessGeneratedCommandsEXT(mInfo, cmd);
}

uint32_t DGCSeq_ESShader::ShaderStageToIdx(
    vk::ShaderStageFlagBits stage) const {
    switch (stage) {
        case vk::ShaderStageFlagBits::eTaskEXT: return TaskShaderIdx;
        case vk::ShaderStageFlagBits::eMeshEXT: return MeshShaderIdx;
        case vk::ShaderStageFlagBits::eFragment: return FragmentShaderIdx;
        default: VE_ASSERT(false, "Invalid shader stage."); return 0;
    }
}

void DGCSeq_ESShader::MakeExecutionSet(
    ShaderIDInfo const& initialShaderIdInfo) {
    const auto name = ShaderManager::ParseShaderName(
        initialShaderIdInfo.name, initialShaderIdInfo.stage,
        initialShaderIdInfo.macros, initialShaderIdInfo.entry);
    InsertInitialObjectToMap(name.c_str(), 0);

    auto initialShaderObject = mShaderMgr.GetShaderObject(
        initialShaderIdInfo.name, initialShaderIdInfo.stage,
        initialShaderIdInfo.macros, initialShaderIdInfo.entry);

    mInitialShaders[0] = initialShaderObject->GetHandle();

    if (!mUseExecutionSet)
        return;

    const auto descLayouts = initialShaderObject->GetDescLayoutHandles();
    const auto pcRange = initialShaderObject->GetPushContantData();

    vk::IndirectExecutionSetShaderLayoutInfoEXT layoutInfo {};
    layoutInfo.setSetLayouts(descLayouts);

    vk::IndirectExecutionSetShaderInfoEXT esShaderInfo {};
    esShaderInfo.setInitialShaders(mInitialShaders)
        .setMaxShaderCount(mMaxESObjectCount)
        .setShaderCount(1)
        .setSetLayoutInfos(layoutInfo);

    if (pcRange)
        esShaderInfo.setPushConstantRanges(*pcRange);

    vk::IndirectExecutionSetCreateInfoEXT esCreateInfo {};
    esCreateInfo.setType(vk::IndirectExecutionSetInfoTypeEXT::eShaderObjects)
        .setInfo(&esShaderInfo);

    mExecutionSet =
        mContext.GetDevice()->createIndirectExecutionSetEXT(esCreateInfo);

    Type_STLVector<vk::WriteIndirectExecutionSetShaderEXT> writeIES {};
    writeIES.reserve(mObjectNamesIdxMap.size());

    for (const auto& [name, idx] : mObjectNamesIdxMap) {
        vk::WriteIndirectExecutionSetShaderEXT write {};
        write.setShader(mShaderMgr.GetShaderObject(name.c_str())->GetHandle())
            .setIndex(idx);
        writeIES.push_back(write);
    }

    mContext.GetDevice()->updateIndirectExecutionSetShaderEXT(mExecutionSet,
                                                              writeIES);
}

void DGCSeqBase::Finalize() {}

vk::PipelineBindPoint DGCSeqBase::GetPipelineBindPoint() {
    if (mIsCompute)
        return vk::PipelineBindPoint::eCompute;
    else
        return vk::PipelineBindPoint::eGraphics;
}

Type_STLVector<vk::ShaderStageFlagBits> DGCSeqBase::GetStageFlagBitArray() {
    if (mIsCompute)
        return {vk::ShaderStageFlagBits::eCompute};
    else
        return {vk::ShaderStageFlagBits::eTaskEXT,
                vk::ShaderStageFlagBits::eMeshEXT,
                vk::ShaderStageFlagBits::eFragment};
}

vk::ShaderStageFlags DGCSeqBase::GetStageFlags() {
    if (mIsCompute)
        return vk::ShaderStageFlagBits::eCompute;
    else
        return vk::ShaderStageFlagBits::eTaskEXT
             | vk::ShaderStageFlagBits::eMeshEXT
             | vk::ShaderStageFlagBits::eFragment;
}

}  // namespace IntelliDesign_NS::Vulkan::Core