#pragma once

#include <SDL_events.h>

#include "Core/Math/MathCore.h"

namespace IntelliDesign_NS::Core {

enum class CameraMovement : uint32_t {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down
};

constexpr float CameraYaw = -90.0f;
constexpr float CameraPitch = 0.0f;
constexpr float CameraSpeed = 1.f;
constexpr float CameraSensitivity = 0.005f;

struct PersperctiveInfo {
    float mNear;
    float mFar;
    float mFov;
    float mAspect;
};

class Camera {
public:
    bool mCaptureMouseMovement {false};
    bool mCaptureKeyboard {true};

    Camera(PersperctiveInfo info,
           MathCore::Float3 position = MathCore::Float3(0.0f, 0.0f, 0.0f),
           MathCore::Float3 up = MathCore::Float3(0.0f, 1.0f, 0.0f));

    void SetAspect(float aspect);

    void SetViewMatrix(MathCore::Mat4 const& viewMatrix);

    // Strafe/Walk the camera a distance d.
    void Strafe(float d);
    void Walk(float d);
    void JumpUp(float d);

    // Rotate the camera.
    void Pitch(float angle);
    void RotateY(float angle);

    void LookAt(const MathCore::Float3& pos, const MathCore::Float3& target,
                const MathCore::Float3& up);

    MathCore::Float3 GetPosition() const;
    MathCore::Float3 GetLookAt();

    MathCore::Mat4 GetViewMatrix();
    MathCore::Mat4 GetProjectionMatrix() const;
    MathCore::Mat4 GetViewProjMatrix();

    MathCore::Mat4 GetInvViewMatrix();
    MathCore::Mat4 GetInvProjectionMatrix() const;
    MathCore::Mat4 GetInvViewProjMatrix();

    void ProcessSDLEvent(SDL_Event* e, float deltaTime);
    void ProcessInput();

    void AdjustPosition(MathCore::Float3 lookAt, MathCore::Float3 extent);
    void AdjustPosition(MathCore::BoundingBox const& boundingBox,
                        MathCore::Float3 scale = MathCore::Float3(1.0f, 1.0f,
                                                                  1.0f));

    MathCore::BoundingFrustum GetFrustum();

    void UpdateViewMatrix();

private:
    void ProcessKeyboard(SDL_Event* e, float deltaTime);
    void ProcessMouseButton(SDL_Event* e);
    void ProcessMouseMovement(SDL_Event* e);
    void ProcessMouseScroll(SDL_Event* e);

    float mMovementSpeed;
    float mMouseSensitivity;
    float mZoom;

    float mXVelocity {0.0f};
    float mYVelocity {0.0f};
    float mZVelocity {0.0f};

    PersperctiveInfo mPerspectiveInfo;

    MathCore::Float3 mPosition = {0.0f, 0.0f, 0.0f};
    MathCore::Float3 mRight = {1.0f, 0.0f, 0.0f};
    MathCore::Float3 mUp = {0.0f, 1.0f, 0.0f};
    MathCore::Float3 mLook = {0.0f, 0.0f, -1.0f};

    bool mReversedZ {false};

    bool mViewDirty {true};

    MathCore::Mat4 mView = MathCore::Identity4x4();
    MathCore::Mat4 mProj = MathCore::Identity4x4();

    MathCore::Mat4 mInvView = MathCore::Identity4x4();
    MathCore::Mat4 mInvProj = MathCore::Identity4x4();
};

}  // namespace IntelliDesign_NS::Core