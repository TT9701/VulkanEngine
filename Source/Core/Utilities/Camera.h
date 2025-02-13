#pragma once

#include <SDL_events.h>

#include "Core/Math/MathCore.h"

namespace IntelliDesign_NS::Core {

enum class CameraMovement : uint32_t { Forward, Backward, Left, Right };

constexpr float CameraYaw = -90.0f;
constexpr float CameraPitch = 0.0f;
constexpr float CameraSpeed = 30.f;
constexpr float CameraSensitivity = 0.1f;
constexpr float CameraZoom = 45.0f;

struct EulerAngles {
    float mYaw;
    float mPitch;
};

struct PersperctiveInfo {
    float mNear;
    float mFar;
    float mFov;
    float mAspect;
};

class Camera {
public:
    // camera Attributes

    CMCore_NS::XMFLOAT3 mPosition;
    CMCore_NS::XMFLOAT3 mFront;
    CMCore_NS::XMFLOAT3 mUp;
    CMCore_NS::XMFLOAT3 mRight;
    CMCore_NS::XMFLOAT3 mWorldUp;

    // euler Angles
    EulerAngles mEulerAngles;

    // camera options
    float mMovementSpeed;
    float mMouseSensitivity;
    float mZoom;

    bool mCaptureMouseMovement {false};
    bool mCaptureKeyboard {true};

    Camera(PersperctiveInfo info,
           CMCore_NS::XMFLOAT3 position = CMCore_NS::XMFLOAT3(0.0f, 0.0f, 0.0f),
           CMCore_NS::XMFLOAT3 up = CMCore_NS::XMFLOAT3(0.0f, 1.0f, 0.0f),
           float yaw = CameraYaw,
           float pitch = CameraPitch);

    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ,
           float yaw, float pitch);

    void SetAspect(float aspect);

    CMCore_NS::XMFLOAT4X4 GetViewMatrix();
    CMCore_NS::XMFLOAT4X4 GetProjectionMatrix();
    CMCore_NS::XMFLOAT4X4 GetViewProjMatrix();

    void ProcessSDLEvent(SDL_Event* e, float deltaTime);

    void AdjustPosition(CMCore_NS::XMFLOAT3 lookAt, CMCore_NS::XMFLOAT3 extent);

private:
    void Update();

    void ProcessKeyboard(SDL_Event* e, float deltaTime);
    void ProcessMouseButton(SDL_Event* e);
    void ProcessMouseMovement(SDL_Event* e);
    void ProcessMouseScroll(SDL_Event* e);

    PersperctiveInfo mPerspectiveInfo;
};

}  // namespace IntelliDesign_NS::Core