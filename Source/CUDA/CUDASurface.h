#pragma once

#include "CUDAHelper.h"

#include <array>

namespace CUDA {

template <typename T, int NDim>
struct CudaSurfaceArray {
    T data[NDim] {};
};

template <typename T, int Xdim, int YDim>
struct CudaSurfaceArray2D {
    CudaSurfaceArray<T, Xdim> data[YDim] {};

    __device__ __host__ T operator()(int yIdx, int xIdx) {
        return data[yIdx].data[xIdx];
    }
};

template <typename T, int Xdim, int YDim, int ZDim>
struct CudaSurfaceArray3D {
    CudaSurfaceArray2D<T, Xdim, YDim> data[ZDim] {};

    __device__ __host__ T operator()(int zIdx, int yIdx, int xIdx) {
        return data[zIdx].data[yIdx].data[xIdx];
    }
};

namespace _Surface {
enum class IOType { Read, Write };

template <int StartIdx, int ComponentCount>
__device__ constexpr auto EffectiveIOPolicy() {
    constexpr auto instructionByteLength = ::std::array {4, 2, 1};
    if constexpr (StartIdx == 0) {
        constexpr int count = ::std::find_if(
            instructionByteLength.begin(), instructionByteLength.end(),
            [](int r) { return r <= ComponentCount; });
        return {0, count};
    } else {
        constexpr int count = ::std::find_if(
            instructionByteLength.begin(), instructionByteLength.end(),
            [](int r) { return r <= ComponentCount && StartIdx % r == 0; });
        return {StartIdx, count};
    }
}

template <int TexelElemCount, int StartIdx, int ComponentCount>
    requires(StartIdx >= 0 && ComponentCount > 0
             && StartIdx + ComponentCount <= TexelElemCount)
struct IO32Bits {
    // for 1D surface
    template <IOType type>
    static __device__ constexpr void Execute(cudaSurfaceObject_t surf, int x,
                                             float* array) {
        auto [start, count] = EffectiveIOPolicy<StartIdx, ComponentCount>();
        IO32Bits<TexelElemCount, start, count>::Execute<type>(surf, x, array);
        if constexpr (ComponentCount - count > 0) {
            IO32Bits<TexelElemCount, start + count,
                     ComponentCount - count>::Execute<type>(surf, x,
                                                            array + count);
        }
    }

    // for 2D surface
    template <IOType type>
    static __device__ constexpr void Execute(cudaSurfaceObject_t surf, int x,
                                             int y, float* array) {
        auto [start, count] = EffectiveIOPolicy<StartIdx, ComponentCount>();
        IO32Bits<TexelElemCount, start, count>::Execute<type>(surf, x, y,
                                                              array);
        if constexpr (ComponentCount - count > 0) {
            IO32Bits<TexelElemCount, start + count,
                     ComponentCount - count>::Execute<type>(surf, x, y,
                                                            array + count);
        }
    }

    // for 3D surface
    template <IOType type>
    static __device__ constexpr void Execute(cudaSurfaceObject_t surf, int x,
                                             int y, int z, float* array) {
        auto [start, count] = EffectiveIOPolicy<StartIdx, ComponentCount>();
        IO32Bits<TexelElemCount, start, count>::Execute<type>(surf, x, y, z,
                                                              array);
        if constexpr (ComponentCount - count > 0) {
            IO32Bits<TexelElemCount, start + count,
                     ComponentCount - count>::Execute<type>(surf, x, y, z,
                                                            array + count);
        }
    }
};

template <int TexelElemCount, int StartIdx>
struct IO32Bits<TexelElemCount, StartIdx, 1> {
    // for 1D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, float* array) {
        surf1Dread(array, surf,
                   (x * TexelElemCount + StartIdx) * sizeof(float));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, float* array) {
        surf1Dwrite(array[0], surf,
                    (x * TexelElemCount + StartIdx) * sizeof(float));
    }

    // for 2D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, float* array) {
        surf2Dread(array, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                   y);
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, float* array) {
        surf2Dwrite(array[0], surf,
                    (x * TexelElemCount + StartIdx) * sizeof(float), y);
    }

    // for 3D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   int z, float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, int z,
                                                 float* array) {
        surf3Dread(array, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                   y, z);
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, int z,
                                                  float* array) {
        surf3Dwrite(array[0], surf,
                    (x * TexelElemCount + StartIdx) * sizeof(float), y, z);
    }
};

template <int TexelElemCount, int StartIdx>
    requires(StartIdx != 1 && TexelElemCount >= 2)
struct IO32Bits<TexelElemCount, StartIdx, 2> {
    // for 1D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {};
        surf1Dread(&temp, surf,
                   (x * TexelElemCount + StartIdx) * sizeof(float));
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {array[0],
                                                                array[1]};
        surf1Dwrite(temp, surf,
                    (x * TexelElemCount + StartIdx) * sizeof(float));
    }

    // for 2D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {};
        surf2Dread(&temp, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                   y);
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {array[0],
                                                                array[1]};
        surf2Dwrite(temp, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                    y);
    }

    // for 3D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   int z, float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, int z,
                                                 float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {};
        surf3Dread(&temp, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                   y, z);
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, int z,
                                                  float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 2>> temp {array[0],
                                                                array[1]};
        surf3Dwrite(temp, surf, (x * TexelElemCount + StartIdx) * sizeof(float),
                    y, z);
    }
};

template <int TexelElemCount>
    requires(TexelElemCount == 4)
struct IO32Bits<TexelElemCount, 0, 4> {
    // for 1D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {};
        surf1Dread(&temp, surf, x * TexelElemCount * sizeof(float));
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {
            array[0], array[1], array[2], array[3]};
        surf1Dwrite(temp, surf, x * TexelElemCount * sizeof(float));
    }

    // for 2D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {};
        surf2Dread(&temp, surf, x * TexelElemCount * sizeof(float), y);
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {
            array[0], array[1], array[2], array[3]};
        surf2Dwrite(temp, surf, x * TexelElemCount * sizeof(float), y);
    }

    // for 3D surface
    template <IOType type>
    static __device__ void Execute(cudaSurfaceObject_t surf, int x, int y,
                                   int z, float* array);

    template <>
    static __device__ void Execute<IOType::Read>(cudaSurfaceObject_t surf,
                                                 int x, int y, int z,
                                                 float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {};
        surf3Dread(&temp, surf, x * TexelElemCount * sizeof(float), y, z);
        memcpy(array, &temp, sizeof(temp));
    }

    template <>
    static __device__ void Execute<IOType::Write>(cudaSurfaceObject_t surf,
                                                  int x, int y, int z,
                                                  float* array) {
        CudaTexelTypeBinder_t<CudaSurfaceArray<float, 4>> temp {
            array[0], array[1], array[2], array[3]};
        surf3Dwrite(temp, surf, x * TexelElemCount * sizeof(float), y, z);
    }
};

// 1D loop
template <int TexelWidth, int TexelElemCount, IOType ioType, int Length,
          int Offset, int NDim>
__device__ constexpr void IOLoop(cudaSurfaceObject_t surf, int texelStartIndex,
                                 float* array) {
    if constexpr (Length > 0) {
        constexpr int texelCompStartIdx =
            TexelElemCount == 1 ? 0
            : Offset >= 0       ? Offset % TexelElemCount
                                : TexelElemCount + Offset % TexelElemCount;
        constexpr int texelCompLength =
            NDim > TexelElemCount - texelCompStartIdx
                ? TexelElemCount - texelCompStartIdx
                : NDim;
        if (texelStartIndex < 0 || texelStartIndex >= TexelWidth) {
            if constexpr (ioType == _Surface::IOType::Read) {
                constexpr ::std::array bv {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
                memcpy(array, &bv, sizeof(float) * texelCompLength);
            }
        } else {
            _Surface::IO32Bits<
                TexelElemCount, texelCompStartIdx,
                texelCompLength>::Execute<ioType>(surf, texelStartIndex, array);
        }
        IOLoop<TexelWidth, TexelElemCount, ioType, Length - texelCompLength, 0,
               NDim>(surf, texelStartIndex + 1, array + texelCompLength);
    }
}

// 2D loop
template <int TexelWidth, int Height, int TexelElemCount, IOType ioType,
          int Length, int Offset, int NDim>
__device__ constexpr void IOLoop(cudaSurfaceObject_t surf, int texelStartIndex,
                                 int y, float* array) {
    if constexpr (Length > 0) {
        constexpr int texelCompStartIdx =
            TexelElemCount == 1 ? 0
            : Offset >= 0       ? Offset % TexelElemCount
                                : TexelElemCount + Offset % TexelElemCount;
        constexpr int texelCompLength =
            NDim > TexelElemCount - texelCompStartIdx
                ? TexelElemCount - texelCompStartIdx
                : NDim;
        if (texelStartIndex < 0 || texelStartIndex >= TexelWidth || y < 0
            || y >= Height) {
            if constexpr (ioType == _Surface::IOType::Read) {
                constexpr ::std::array bv {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
                memcpy(array, &bv, sizeof(float) * texelCompLength);
            }
        } else {
            _Surface::IO32Bits<
                TexelElemCount, texelCompStartIdx,
                texelCompLength>::Execute<ioType>(surf, texelStartIndex, y,
                                                  array);
        }
        IOLoop<TexelWidth, Height, TexelElemCount, ioType,
               Length - texelCompLength, 0, NDim>(surf, texelStartIndex + 1, y,
                                                  array + texelCompLength);
    }
}

// 3D loop
template <int TexelWidth, int Height, int Depth, int TexelElemCount,
          IOType ioType, int Length, int Offset, int NDim>
__device__ constexpr void IOLoop(cudaSurfaceObject_t surf, int texelStartIndex,
                                 int y, int z, float* array) {
    if constexpr (Length > 0) {
        constexpr int texelCompStartIdx =
            TexelElemCount == 1 ? 0
            : Offset >= 0       ? Offset % TexelElemCount
                                : TexelElemCount + Offset % TexelElemCount;
        constexpr int texelCompLength =
            NDim > TexelElemCount - texelCompStartIdx
                ? TexelElemCount - texelCompStartIdx
                : NDim;
        if (texelStartIndex < 0 || texelStartIndex >= TexelWidth || y < 0
            || y >= Height || z < 0 || z >= Depth) {
            if constexpr (ioType == _Surface::IOType::Read) {
                constexpr ::std::array bv {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
                memcpy(array, &bv, sizeof(float) * texelCompLength);
            }
        } else {
            _Surface::IO32Bits<
                TexelElemCount, texelCompStartIdx,
                texelCompLength>::Execute<ioType>(surf, texelStartIndex, y, z,
                                                  array);
        }
        IOLoop<TexelWidth, Height, Depth, TexelElemCount, ioType,
               Length - texelCompLength, 0, NDim>(surf, texelStartIndex + 1, y,
                                                  z, array + texelCompLength);
    }
}
}  // namespace _Surface

template <typename UserType, int Dimension>
class CUDASurfaceBase {
protected:
    using CudaTexelDataType = CudaTexelTypeBinder_t<UserType>;
    using Bit32_t           = float;
    static constexpr int mTexelElemCount =
        sizeof(CudaTexelDataType) / sizeof(float);

protected:
    CUDASurfaceBase(cudaResourceDesc const* desc) {
        cudaCreateSurfaceObject(&mSurfObj, desc);
    }

    virtual ~CUDASurfaceBase() {
        if (mSurfObj)
            cudaDestroySurfaceObject(mSurfObj);
    }

public:
    __device__ __host__ constexpr cudaSurfaceObject_t GetObj() const {
        return mSurfObj;
    }

private:
    cudaSurfaceObject_t mSurfObj {};
};

/**
 * 1D Surface on GPU
 */
template <typename UserType, int TexelWidth>
class CudaSurface1D : CUDASurfaceBase<UserType, 1> {
    using Base = CUDASurfaceBase<UserType, 1>;
    using typename Base::Bit32_t;

public:
    CudaSurface1D(cudaResourceDesc const* desc) : Base(desc) {}

    ~CudaSurface1D() override = default;

    // 1D read
    template <int Offset, int NDim>
    __device__ constexpr void Read1D(CudaSurfaceArray<Bit32_t, NDim>* array,
                                     int                              x);

    // 1D write
    template <int Offset, int NDim>
    __device__ constexpr void Write1D(CudaSurfaceArray<Bit32_t, NDim>& array,
                                      int                              x);
};

template <typename UserType, int TexelWidth>
template <int Offset, int NDim>
__device__ constexpr void CudaSurface1D<UserType, TexelWidth>::Read1D(
    CudaSurfaceArray<Bit32_t, NDim>* array, int x) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Base::mTexelElemCount, _Surface::IOType::Read,
                     NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex,
        reinterpret_cast<Bit32_t*>(array->data));
}

template <typename UserType, int TexelWidth>
template <int Offset, int NDim>
__device__ constexpr void CudaSurface1D<UserType, TexelWidth>::Write1D(
    CudaSurfaceArray<Bit32_t, NDim>& array, int x) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Base::mTexelElemCount, _Surface::IOType::Write,
                     NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex,
        reinterpret_cast<Bit32_t*>(array.data));
}

/**
 * 2D Surface on GPU
 */
template <typename UserType, int TexelWidth, int Height>
class CUDASurface2D : CUDASurfaceBase<UserType, 2> {
    using Base = CUDASurfaceBase<UserType, 2>;
    using typename Base::Bit32_t;

public:
    CUDASurface2D(cudaResourceDesc const* desc) : Base(desc) {}

    ~CUDASurface2D() override = default;

    // 1D read
    template <int Offset, int NDim>
    __device__ constexpr void Read1D(CudaSurfaceArray<Bit32_t, NDim>* array,
                                     int x, int y);

    // 1D write
    template <int Offset, int NDim>
    __device__ constexpr void Write1D(CudaSurfaceArray<Bit32_t, NDim>& array,
                                      int x, int y);

    // 2D read
    template <int XOffset, int XDim, int YOffset, int YDim, int Idx = 0>
    __device__ constexpr void Read2D(
        CudaSurfaceArray2D<Bit32_t, XDim, YDim>* array, int x, int y);

    // 2D write
    template <int XOffset, int XDim, int YOffset, int YDim, int Idx = 0>
    __device__ constexpr void Write2D(
        CudaSurfaceArray2D<Bit32_t, XDim, YDim>& array, int x, int y);
};

template <typename UserType, int TexelWidth, int Height>
template <int Offset, int NDim>
__device__ constexpr void CUDASurface2D<UserType, TexelWidth, Height>::Read1D(
    CudaSurfaceArray<Bit32_t, NDim>* array, int x, int y) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Height, Base::mTexelElemCount,
                     _Surface::IOType::Read, NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex, y,
        reinterpret_cast<Bit32_t*>(array->data));
}

template <typename UserType, int TexelWidth, int Height>
template <int Offset, int NDim>
__device__ constexpr void CUDASurface2D<UserType, TexelWidth, Height>::Write1D(
    CudaSurfaceArray<Bit32_t, NDim>& array, int x, int y) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Height, Base::mTexelElemCount,
                     _Surface::IOType::Write, NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex, y,
        reinterpret_cast<Bit32_t*>(array.data));
}

template <typename UserType, int TexelWidth, int Height>
template <int XOffset, int XDim, int YOffset, int YDim, int Idx>
__device__ constexpr void CUDASurface2D<UserType, TexelWidth, Height>::Read2D(
    CudaSurfaceArray2D<Bit32_t, XDim, YDim>* array, int x, int y) {
    if constexpr (Idx < YDim) {
        Read1D<XOffset>(&array->data[Idx], x, y + YOffset);
        Read2D<XOffset, XDim, YOffset, YDim, Idx + 1>(array, x, y + 1);
    }
}

template <typename UserType, int TexelWidth, int Height>
template <int XOffset, int XDim, int YOffset, int YDim, int Idx>
__device__ constexpr void CUDASurface2D<UserType, TexelWidth, Height>::Write2D(
    CudaSurfaceArray2D<Bit32_t, XDim, YDim>& array, int x, int y) {
    if constexpr (Idx < YDim) {
        Write1D<XOffset>(array.data[Idx], x, y + YOffset);
        Write2D<XOffset, XDim, YOffset, YDim, Idx + 1>(array, x, y + 1);
    }
}

/**
 * 3D Surface on GPU
 */
template <typename UserType, int TexelWidth, int Height, int Depth>
class CUDASurface3D : CUDASurfaceBase<UserType, 3> {
    using Base = CUDASurfaceBase<UserType, 3>;
    using typename Base::Bit32_t;

public:
    CUDASurface3D(cudaResourceDesc const* desc) : Base(desc) {}

    ~CUDASurface3D() override = default;

    // 1D read
    template <int Offset, int NDim>
    __device__ constexpr void Read1D(CudaSurfaceArray<Bit32_t, NDim>* array,
                                     int x, int y, int z);

    // 1D write
    template <int Offset, int NDim>
    __device__ constexpr void Write1D(CudaSurfaceArray<Bit32_t, NDim>& array,
                                      int x, int y, int z);

    // 2D read
    template <int XOffset, int XDim, int YOffset, int YDim, int Idx = 0>
    __device__ constexpr void Read2D(
        CudaSurfaceArray2D<Bit32_t, XDim, YDim>* array, int x, int y, int z);

    // 2D write
    template <int XOffset, int XDim, int YOffset, int YDim, int Idx = 0>
    __device__ constexpr void Write2D(
        CudaSurfaceArray2D<Bit32_t, XDim, YDim>& array, int x, int y, int z);

    // 3D read
    template <int XOffset, int XDim, int YOffset, int YDim, int ZOffset,
              int ZDim, int YIdx = 0, int ZIdx = 0>
    __device__ constexpr void Read3D(
        CudaSurfaceArray3D<Bit32_t, XDim, YDim, ZDim>* array, int x, int y,
        int z);

    // 3D write
    template <int XOffset, int XDim, int YOffset, int YDim, int ZOffset,
              int ZDim, int YIdx = 0, int ZIdx = 0>
    __device__ constexpr void Write3D(
        CudaSurfaceArray3D<Bit32_t, XDim, YDim, ZDim>& array, int x, int y,
        int z);
};

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int Offset, int NDim>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Read1D(
    CudaSurfaceArray<Bit32_t, NDim>* array, int x, int y, int z) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Height, Depth, Base::mTexelElemCount,
                     _Surface::IOType::Read, NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex, y, z,
        reinterpret_cast<Bit32_t*>(array->data));
}

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int Offset, int NDim>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Write1D(
    CudaSurfaceArray<Bit32_t, NDim>& array, int x, int y, int z) {
    constexpr int texelOffset =
        (Offset >= 0 ? Offset : Offset - (Base::mTexelElemCount - 1))
        / Base::mTexelElemCount;
    int texelStartIndex = x + texelOffset;
    _Surface::IOLoop<TexelWidth, Height, Depth, Base::mTexelElemCount,
                     _Surface::IOType::Write, NDim, Offset, NDim>(
        Base::GetObj(), texelStartIndex, y, z,
        reinterpret_cast<Bit32_t*>(array.data));
}

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int XOffset, int XDim, int YOffset, int YDim, int Idx>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Read2D(
    CudaSurfaceArray2D<Bit32_t, XDim, YDim>* array, int x, int y, int z) {
    if constexpr (Idx < YDim) {
        Read1D<XOffset>(&array->data[Idx], x, y + YOffset, z);
        Read2D<XOffset, XDim, YOffset, YDim, Idx + 1>(array, x, y + 1, z);
    }
}

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int XOffset, int XDim, int YOffset, int YDim, int Idx>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Write2D(
    CudaSurfaceArray2D<Bit32_t, XDim, YDim>& array, int x, int y, int z) {
    if constexpr (Idx < YDim) {
        Write1D<XOffset>(array.data[Idx], x, y + YOffset, z);
        Write2D<XOffset, XDim, YOffset, YDim, Idx + 1>(array, x, y + 1, z);
    }
}

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int XOffset, int XDim, int YOffset, int YDim, int ZOffset, int ZDim,
          int YIdx, int ZIdx>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Read3D(
    CudaSurfaceArray3D<Bit32_t, XDim, YDim, ZDim>* array, int x, int y, int z) {
    if constexpr (ZIdx < ZDim) {
        Read2D<XOffset, XDim, YOffset, YDim, YIdx>(&array->data[ZIdx], x, y,
                                                   z + ZOffset);
        Read3D<XOffset, XDim, YOffset, YDim, ZOffset, ZDim, YIdx, ZIdx + 1>(
            array, x, y, z + 1);
    }
}

template <typename UserType, int TexelWidth, int Height, int Depth>
template <int XOffset, int XDim, int YOffset, int YDim, int ZOffset, int ZDim,
          int YIdx, int ZIdx>
__device__ constexpr void
CUDASurface3D<UserType, TexelWidth, Height, Depth>::Write3D(
    CudaSurfaceArray3D<Bit32_t, XDim, YDim, ZDim>& array, int x, int y, int z) {
    if constexpr (ZIdx < ZDim) {
        Write2D<XOffset, XDim, YOffset, YDim, YIdx>(array.data[ZIdx], x, y,
                                                    z + ZOffset);
        Write3D<XOffset, XDim, YOffset, YDim, ZOffset, ZDim, YIdx, ZIdx + 1>(
            array, x, y, z + 1);
    }
}

}  // namespace CUDA