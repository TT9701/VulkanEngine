#include "Camera.h"

#include "Defines.h"
#include "MemoryPool.h"

namespace IntelliDesign_NS::Core {

Camera::Camera(PersperctiveInfo info, CMCore_NS::XMFLOAT3 position,
               CMCore_NS::XMFLOAT3 up, float yaw, float pitch)
    : mPosition(position),
      mFront(CMCore_NS::XMFLOAT3(0.0f, 0.0f, -1.0f)),
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
      mFront(CMCore_NS::XMFLOAT3(0.0f, 0.0f, -1.0f)),
      mWorldUp(CMCore_NS::XMFLOAT3(upX, upY, upZ)),
      mEulerAngles {yaw, pitch},
      mMovementSpeed(CameraSpeed),
      mMouseSensitivity(CameraSensitivity),
      mZoom(CameraZoom) {
    Update();
}

void Camera::SetAspect(float aspect) {
    mPerspectiveInfo.mAspect = aspect;
}

CMCore_NS::XMFLOAT4X4 Camera::GetViewMatrix() {
    using namespace IntelliDesign_NS;

    auto position = CMCore_NS::XMLoadFloat3(&mPosition);
    auto front = CMCore_NS::XMLoadFloat3(&mFront);
    auto up = CMCore_NS::XMLoadFloat3((CMCore_NS::XMFLOAT3*)&mUp);
    auto focus = CMCore_NS::XMVectorAdd(position, front);

    CMCore_NS::XMFLOAT4X4 ret {};
    CMCore_NS::XMStoreFloat4x4(
        &ret, CMCore_NS::XMMatrixLookAtRH(position, focus, up));

    return ret;
}

CMCore_NS::XMFLOAT4X4 Camera::GetProjectionMatrix() {
    CMCore_NS::XMFLOAT4X4 mat {};

    CMCore_NS::XMStoreFloat4x4(
        &mat, CMCore_NS::XMMatrixPerspectiveFovRH(
                  mPerspectiveInfo.mFov, mPerspectiveInfo.mAspect,
                  mPerspectiveInfo.mNear, mPerspectiveInfo.mFar));

    mat(1, 1) *= -1.0f;

    return mat;
}

CMCore_NS::XMFLOAT4X4 Camera::GetViewProjMatrix() {
    auto view = GetViewMatrix();
    auto proj = GetProjectionMatrix();

    auto viewMat = CMCore_NS::XMLoadFloat4x4(&view);
    auto projMat = CMCore_NS::XMLoadFloat4x4(&proj);

    CMCore_NS::XMFLOAT4X4 mat {};
    CMCore_NS::XMStoreFloat4x4(&mat,
                               CMCore_NS::XMMatrixMultiply(viewMat, projMat));

    return mat;
}

void Camera::ProcessSDLEvent(SDL_Event* e, float deltaTime) {
    if (mCaptureKeyboard)
        ProcessKeyboard(e, deltaTime);

    ProcessMouseButton(e);

    if (mCaptureMouseMovement)
        ProcessMouseMovement(e);

    ProcessMouseScroll(e);
}

void Camera::AdjustPosition(CMCore_NS::XMFLOAT3 lookAt,
                            CMCore_NS::XMFLOAT3 extent) {
    mEulerAngles = {CameraYaw, CameraPitch};

    auto aspect = extent.x / extent.y > mPerspectiveInfo.mAspect
                    ? extent.x / mPerspectiveInfo.mAspect
                    : extent.y;

    auto dist = aspect / tan(mPerspectiveInfo.mFov * 0.5f);

    CMCore_NS::XMVECTOR displacement {0.0f, 0.0f, extent.z + dist};

    auto l = CMCore_NS::XMLoadFloat3(&lookAt);

    CMCore_NS::XMStoreFloat3(&mPosition,
                             CMCore_NS::XMVectorAdd(l, displacement));

    Update();
}

void Camera::Update() {
    CMCore_NS::XMFLOAT3 front {};

    auto radYaw = CMCore_NS::XMConvertToRadians(mEulerAngles.mYaw);
    auto radPitch = CMCore_NS::XMConvertToRadians(mEulerAngles.mPitch);

    front.x = cos(radYaw) * cos(radPitch);
    front.y = sin(radPitch);
    front.z = sin(radYaw) * cos(radPitch);

    auto frontVec = CMCore_NS::XMLoadFloat3(&front);
    CMCore_NS::XMStoreFloat3(&mFront, CMCore_NS::XMVector3Normalize(frontVec));

    auto worldUpVec = CMCore_NS::XMLoadFloat3(&mWorldUp);

    auto rightVec = CMCore_NS::XMVector3Normalize(
        CMCore_NS::XMVector3Cross(frontVec, worldUpVec));
    CMCore_NS::XMStoreFloat3(&mRight, rightVec);

    CMCore_NS::XMStoreFloat3(
        &mUp, CMCore_NS::XMVector3Normalize(
                  CMCore_NS::XMVector3Cross(rightVec, frontVec)));
}

void Camera::ProcessKeyboard(SDL_Event* e, float deltaTime) {
    if (e->type == SDL_KEYDOWN) {
        mMovementSpeed += mMovementSpeed * 0.05f;

        float velocity = mMovementSpeed * deltaTime;
        CMCore_NS::XMVECTOR posVelocityVec {velocity, velocity, velocity};
        CMCore_NS::XMVECTOR negVelocityVec {-velocity, -velocity, -velocity};

        auto position = CMCore_NS::XMLoadFloat3(&mPosition);

        auto pos = CMCore_NS::XMLoadFloat3(&mPosition);
        if (e->key.keysym.sym == SDLK_w) {
            auto front = CMCore_NS::XMLoadFloat3(&mFront);
            CMCore_NS::XMStoreFloat3(&mPosition,
                                     CMCore_NS::XMVectorMultiplyAdd(
                                         front, posVelocityVec, position));
        }
        if (e->key.keysym.sym == SDLK_s) {
            auto front = CMCore_NS::XMLoadFloat3(&mFront);
            CMCore_NS::XMStoreFloat3(&mPosition,
                                     CMCore_NS::XMVectorMultiplyAdd(
                                         front, negVelocityVec, position));
        }
        if (e->key.keysym.sym == SDLK_a) {
            auto right = CMCore_NS::XMLoadFloat3(&mRight);
            CMCore_NS::XMStoreFloat3(&mPosition,
                                     CMCore_NS::XMVectorMultiplyAdd(
                                         right, negVelocityVec, position));
        }
        if (e->key.keysym.sym == SDLK_d) {
            auto right = CMCore_NS::XMLoadFloat3(&mRight);
            CMCore_NS::XMStoreFloat3(&mPosition,
                                     CMCore_NS::XMVectorMultiplyAdd(
                                         right, posVelocityVec, position));
        }
        if (e->key.keysym.sym == SDLK_SPACE) {
            auto up = CMCore_NS::XMLoadFloat3(&mUp);
            CMCore_NS::XMStoreFloat3(
                &mPosition,
                CMCore_NS::XMVectorMultiplyAdd(up, posVelocityVec, position));
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

    if (mEulerAngles.mPitch > 89.0f)
        mEulerAngles.mPitch = 89.0f;
    if (mEulerAngles.mPitch < -89.0f)
        mEulerAngles.mPitch = -89.0f;

    Update();
}

void Camera::ProcessMouseScroll(SDL_Event* e) {
    if (e->type == SDL_MOUSEWHEEL) {
        mZoom += static_cast<float>(e->wheel.y);

        if (mZoom < 1.0f)
            mZoom = 1.0f;
        if (mZoom > 45.0f)
            mZoom = 45.0f;
    }
}

}  // namespace IntelliDesign_NS::Core