#include "Camera.h"

#include <algorithm>

#include "MemoryPool.h"

namespace IntelliDesign_NS::Core {

namespace IDCMCore_NS = CMCore_NS;

Camera::Camera(PersperctiveInfo info, MathCore::Float3 position,
               MathCore::Float3 up, float yaw, float pitch)
    : mPosition(position),
      mFront(MathCore::Float3(0.0f, 0.0f, -1.0f)),
      mWorldUp(up),
      mEulerAngles {yaw, pitch},
      mMovementSpeed(CameraSpeed),
      mMouseSensitivity(CameraSensitivity),
      mZoom(MathCore::ConvertToDegrees(info.mFov)),
      mPerspectiveInfo(info) {
    Update();

    if (mPerspectiveInfo.mNear > mPerspectiveInfo.mFar) {
        mReversedZ = true;
    }
}

void Camera::SetAspect(float aspect) {
    mPerspectiveInfo.mAspect = aspect;
}

MathCore::Mat4 Camera::GetViewMatrix() const {
    using namespace IDCMCore_NS;

    auto position = mPosition.GetSIMD();
    auto focus = VectorAdd(position, mFront.GetSIMD());

    return MatrixLookAt(position, focus, mUp.GetSIMD());
}

MathCore::Mat4 Camera::GetProjectionMatrix() const {
    using namespace IDCMCore_NS;

    Mat4 mat {
        MatrixPerspectiveFov(mPerspectiveInfo.mFov, mPerspectiveInfo.mAspect,
                             mPerspectiveInfo.mNear, mPerspectiveInfo.mFar)};

    mat(1, 1) *= -1.0f;

    return mat;
}

MathCore::Mat4 Camera::GetViewProjMatrix() const {
    return GetViewMatrix().GetSIMD() * GetProjectionMatrix().GetSIMD();
}

MathCore::Mat4 Camera::GetInvViewMatrix() const {
    return MathCore::MatrixInverse(nullptr, GetViewMatrix().GetSIMD());
}

MathCore::Mat4 Camera::GetInvProjectionMatrix() const {
    return MathCore::MatrixInverse(nullptr, GetProjectionMatrix().GetSIMD());
}

MathCore::Mat4 Camera::GetInvViewProjMatrix() const {
    return MathCore::MatrixInverse(nullptr, GetViewProjMatrix().GetSIMD());
}

void Camera::ProcessSDLEvent(SDL_Event* e, float deltaTime) {
    if (mCaptureKeyboard)
        ProcessKeyboard(e, deltaTime);

    ProcessMouseButton(e);

    if (mCaptureMouseMovement)
        ProcessMouseMovement(e);

    // ProcessMouseScroll(e);
}

void Camera::AdjustPosition(MathCore::Float3 lookAt, MathCore::Float3 extent) {
    using namespace IDCMCore_NS;

    mEulerAngles = {CameraYaw, CameraPitch};

    auto aspect = extent.x / extent.y > mPerspectiveInfo.mAspect
                    ? extent.x / mPerspectiveInfo.mAspect
                    : extent.y;

    auto dist = aspect / tan(mPerspectiveInfo.mFov * 0.5f);

    mPosition = VectorAdd(lookAt.GetSIMD(), {0.0f, 0.0f, extent.z + dist});

    Update();
}

void Camera::AdjustPosition(MathCore::BoundingBox const& boundingBox) {
    AdjustPosition(boundingBox.Center, boundingBox.Extents);
}

MathCore::BoundingFrustum Camera::GetFrustum() const {
    using namespace IDCMCore_NS;

    float near = mPerspectiveInfo.mNear;
    float far = mPerspectiveInfo.mFar;

    if (mReversedZ)
        ::std::swap(near, far);

    BoundingFrustum frustum {
        Mat4 {MatrixPerspectiveFov(mPerspectiveInfo.mFov,
                                   mPerspectiveInfo.mAspect, near, far)}
            .GetSIMD(),
        true};

    auto invViewMat = MatrixInverse(nullptr, GetViewMatrix().GetSIMD());

    frustum.Transform(frustum, invViewMat);

    return frustum;
}

void Camera::Update() {
    using namespace IDCMCore_NS;

    auto radYaw = ConvertToRadians(mEulerAngles.mYaw);
    auto radPitch = ConvertToRadians(mEulerAngles.mPitch);

    Float3 front {cos(radYaw) * cos(radPitch), sin(radPitch),
                  sin(radYaw) * cos(radPitch)};
    auto frontVec = front.GetSIMD();

    mFront = Vector3Normalize(frontVec);

    mRight = Vector3Normalize(Vector3Cross(frontVec, mWorldUp.GetSIMD()));

    mUp = Vector3Normalize(Vector3Cross(mRight.GetSIMD(), frontVec));
}

void Camera::ProcessKeyboard(SDL_Event* e, float deltaTime) {
    using namespace IDCMCore_NS;

    if (e->type == SDL_KEYDOWN) {
        mMovementSpeed += mMovementSpeed * 0.05f;

        float velocity = mMovementSpeed * deltaTime;
        SIMD_Vec posVelocityVec {velocity, velocity, velocity};
        SIMD_Vec negVelocityVec {-velocity, -velocity, -velocity};

        auto position = mPosition.GetSIMD();

        if (e->key.keysym.sym == SDLK_w) {
            mPosition =
                VectorMultiplyAdd(mFront.GetSIMD(), posVelocityVec, position);
        }
        if (e->key.keysym.sym == SDLK_s) {
            mPosition =
                VectorMultiplyAdd(mFront.GetSIMD(), negVelocityVec, position);
        }
        if (e->key.keysym.sym == SDLK_a) {
            mPosition =
                VectorMultiplyAdd(mRight.GetSIMD(), negVelocityVec, position);
        }
        if (e->key.keysym.sym == SDLK_d) {
            mPosition =
                VectorMultiplyAdd(mRight.GetSIMD(), posVelocityVec, position);
        }
        if (e->key.keysym.sym == SDLK_SPACE) {
            mPosition =
                VectorMultiplyAdd(mUp.GetSIMD(), posVelocityVec, position);
        }
    }

    if (e->type == SDL_KEYUP) {
        mMovementSpeed = CameraSpeed;
    }
}

void Camera::ProcessMouseButton(SDL_Event* e) {
    if (e->type == SDL_MOUSEBUTTONDOWN) {
        if (e->button.button == SDL_BUTTON_RIGHT) {
            mCaptureMouseMovement = true;
        }
    }

    if (e->type == SDL_MOUSEBUTTONUP) {
        if (e->button.button == SDL_BUTTON_RIGHT) {
            mCaptureMouseMovement = false;
        }
    }
}

void Camera::ProcessMouseMovement(SDL_Event* e) {
    if (e->type == SDL_MOUSEMOTION) {
        mEulerAngles.mYaw += (float)e->motion.xrel * mMouseSensitivity;
        mEulerAngles.mPitch -= (float)e->motion.yrel * mMouseSensitivity;
    }

    mEulerAngles.mPitch = std::clamp(mEulerAngles.mPitch, -89.0f, 89.0f);

    Update();
}

void Camera::ProcessMouseScroll(SDL_Event* e) {
    if (e->type == SDL_MOUSEWHEEL) {
        mZoom -= static_cast<float>(e->wheel.y);

        mZoom = ::std::clamp(mZoom, 1.0f, 60.0f);

        mPerspectiveInfo.mFov = MathCore::ConvertToRadians(mZoom);
    }
}

}  // namespace IntelliDesign_NS::Core