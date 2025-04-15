#pragma once

#include <SDL_events.h>

#include "Core/Math/MathCore.h"

namespace IntelliDesign_NS::Core {

enum class CameraMovement : uint32_t { Forward, Backward, Left, Right };

constexpr float CameraYaw = -90.0f;
constexpr float CameraPitch = 0.0f;
constexpr float CameraSpeed = 3000.f;
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

    MathCore::Float3 mPosition;
    MathCore::Float3 mFront;
    MathCore::Float3 mUp;
    MathCore::Float3 mRight;
    MathCore::Float3 mWorldUp;

    // euler Angles
    EulerAngles mEulerAngles;

    // camera options
    float mMovementSpeed;
    float mMouseSensitivity;
    float mZoom;

    bool mCaptureMouseMovement {false};
    bool mCaptureKeyboard {true};

    Camera(PersperctiveInfo info,
           MathCore::Float3 position = MathCore::Float3(0.0f, 0.0f, 0.0f),
           MathCore::Float3 up = MathCore::Float3(0.0f, 1.0f, 0.0f),
           float yaw = CameraYaw, float pitch = CameraPitch);

    void SetAspect(float aspect);

    MathCore::Mat4 GetViewMatrix() const;
    MathCore::Mat4 GetProjectionMatrix() const;
    MathCore::Mat4 GetViewProjMatrix() const;

    MathCore::Mat4 GetInvViewMatrix() const;
    MathCore::Mat4 GetInvProjectionMatrix() const;
    MathCore::Mat4 GetInvViewProjMatrix() const;

    void ProcessSDLEvent(SDL_Event* e, float deltaTime);

    void AdjustPosition(MathCore::Float3 lookAt, MathCore::Float3 extent);

    MathCore::BoundingFrustum GetFrustum() const;

private:
    void Update();

    void ProcessKeyboard(SDL_Event* e, float deltaTime);
    void ProcessMouseButton(SDL_Event* e);
    void ProcessMouseMovement(SDL_Event* e);
    void ProcessMouseScroll(SDL_Event* e);

    PersperctiveInfo mPerspectiveInfo;
    bool mReversedZ {false};
};

}  // namespace IntelliDesign_NS::Core