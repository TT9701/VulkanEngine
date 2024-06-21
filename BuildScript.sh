#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXTERNAL_SOURCE_DIR="${SCRIPT_DIR}/External"

BUILD_DIR="${SCRIPT_DIR}/Build"
EXTERNAL_BUILD_DIR="${EXTERNAL_SOURCE_DIR}/Build"

mkdir -p "${EXTERNAL_BUILD_DIR}"
cd "${EXTERNAL_BUILD_DIR}"
cmake .. 
cmake --build . --target install --config Debug
cmake --build . --target install --config Release

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DCUDA_VULKAN_INTEROP=1

rm -rf "${EXTERNAL_BUILD_DIR}"
exit