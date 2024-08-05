#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUILD_DIR="${SCRIPT_DIR}/Build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ../SDL -DCMAKE_INSTALL_PREFIX=../Output -DSDL_STATIC=OFF -DSDL_TEST=OFF
cmake --build . --target install --config Debug
cmake --build . --target install --config Release
rm -rf *

cmake ../Assimp/ -DASSIMP_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=../Output
cmake --build . --target install --config Debug
cmake --build . --target install --config Release
cd .. && rm -rf "${BUILD_DIR}"