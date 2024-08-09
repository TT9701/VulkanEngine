#include "Mesh.hpp"

#include "Core/VulkanCore/VulkanEngine.hpp"
#include "Core/VulkanCore/VulkanContext.hpp"

Mesh::Mesh(std::vector<Vertex> const&   vertices,
           std::vector<uint32_t> const& indices)
    : mVertices(vertices), mIndices(indices) {
}

void Mesh::GenerateBuffers(VulkanContext* context, VulkanEngine* engine) {
    const size_t vertexBufferSize = mVertices.size() * sizeof(mVertices[0]);
    const size_t indexBufferSize  = mIndices.size() * sizeof(mIndices[0]);

    mBuffers.mVertexBuffer = context->CreatePersistentBuffer(
        vertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer
                              | vk::BufferUsageFlagBits::eTransferDst
                              | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    mBuffers.mIndexBuffer = context->CreatePersistentBuffer(
        indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer
                             | vk::BufferUsageFlagBits::eTransferDst);

    vk::BufferDeviceAddressInfo deviceAddrInfo {};
    deviceAddrInfo.setBuffer(mBuffers.mVertexBuffer->GetHandle());

    mBuffers.mVertexBufferAddress =
        context->GetDeviceHandle().getBufferAddress(deviceAddrInfo);

    auto staging =
        context->CreateStagingBuffer(vertexBufferSize + indexBufferSize);

    void* data = staging->GetAllocationInfo().pMappedData;
    memcpy(data, mVertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, mIndices.data(), indexBufferSize);

    engine->GetImmediateSubmitManager()->Submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy {};
        vertexCopy.setSize(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(),
                       mBuffers.mVertexBuffer->GetHandle(),
                       vertexCopy);

        vk::BufferCopy indexCopy {};
        indexCopy.setSize(indexBufferSize).setSrcOffset(vertexBufferSize);
        cmd.copyBuffer(staging->GetHandle(), mBuffers.mIndexBuffer->GetHandle(),
                       indexCopy);
    });

    mConstants.mVertexBufferAddress = mBuffers.mVertexBufferAddress;
}