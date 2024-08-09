#include "Mesh.hpp"

Mesh::Mesh(std::vector<Vertex> const&   vertices,
           std::vector<uint32_t> const& indices)
    : mVertices(vertices), mIndices(indices) {
}