#include "Camera.h"

#include <algorithm>

#include "MemoryPool.h"

namespace IntelliDesign_NS::Core {

Camera::Camera(PersperctiveInfo info, MathCore::Float3 position,
               MathCore::Float3 up, float yaw, float pitch)
    : mPosition(position),
      mFront(MathCore::Float3(0.0f, 0.0f, -1.0f)),
      mWorldUp(up),
      mEulerAngles {yaw, pitch},
      mMovementSpeed(CameraSpeed),
      mMouseSensitivity(CameraSensitivity),
      mZoom(CameraZoom),
      mPerspectiveInfo(info) {
    Update();
}

Camera::Camera(float posX, float posY, float posZ, float upX, float upY,
               float upZ, float yaw, float pitch)
    : mPosition(posX, posY, posZ),
      mFront(MathCore::Float3(0.0f, 0.0f, -1.0f)),
      mWorldUp(MathCore::Float3(upX, upY, upZ)),
      mEulerAngles {yaw, pitch},
      mMovementSpeed(CameraSpeed),
      mMouseSensitivity(CameraSensitivity),
      mZoom(CameraZoom) {
    Update();
}

void Camera::SetAspect(float aspect) {
    mPerspectiveInfo.mAspect = aspect;
}

MathCore::Mat4 Camera::GetViewMatrix() const {
    using namespace IDCMCore_NS;

    auto position = mPosition.GetSIMD();
    auto front = mFront.GetSIMD();
    auto up = mUp.GetSIMD();
    auto focus = VectorAdd(position, front);

    Mat4 ret {MatrixLookAt(position, focus, up)};

    return ret;
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

void Camera::ProcessSDLEvent(SDL_Event* e, float deltaTime) {
    if (mCaptureKeyboard)
        ProcessKeyboard(e, deltaTime);

    ProcessMouseButton(e);

    if (mCaptureMouseMovement)
        ProcessMouseMovement(e);

    ProcessMouseScroll(e);
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

void Camera::Update() {
    using namespace IDCMCore_NS;

    Float3 front {};

    auto radYaw = ConvertToRadians(mEulerAngles.mYaw);
    auto radPitch = ConvertToRadians(mEulerAngles.mPitch);

    front.x = cos(radYaw) * cos(radPitch);
    front.y = sin(radPitch);
    front.z = sin(radYaw) * cos(radPitch);

    auto frontVec = front.GetSIMD();
    mFront = Vector3Normalize(frontVec);

    auto rightVec =
        Vector3Normalize(Vector3Cross(frontVec, mWorldUp.GetSIMD()));

    mRight = rightVec;

    mUp = Vector3Normalize(Vector3Cross(rightVec, frontVec));
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
        mZoom += static_cast<float>(e->wheel.y);

        mZoom = ::std::clamp(mZoom, 1.0f, 45.0f);
    }
}

}  // namespace IntelliDesign_NS::Core