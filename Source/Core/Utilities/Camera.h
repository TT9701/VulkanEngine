#pragma once

#include <SDL_events.h>
#include <glm/glm.hpp>

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
    glm::vec3 mPosition;
    glm::vec3 mFront;
    glm::vec3 mUp;
    glm::vec3 mRight;
    glm::vec3 mWorldUp;

    // euler Angles
    EulerAngles mEulerAngles;

    // camera options
    float mMovementSpeed;
    float mMouseSensitivity;
    float mZoom;

    bool mCaptureMouseMovement {false};
    bool mCaptureKeyboard {true};

    Camera(PersperctiveInfo info,
           glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = CameraYaw,
           float pitch = CameraPitch);

    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ,
           float yaw, float pitch);

    void SetAspect(float aspect);

    glm::mat4 GetViewMatrix();
    glm::mat4 GetProjectionMatrix();

    void ProcessSDLEvent(SDL_Event* e, float deltaTime);

    void AdjustPosition(glm::vec3 lookAt, glm::vec3 extent);

private:
    void Update();

    void ProcessKeyboard(SDL_Event* e, float deltaTime);
    void ProcessMouseButton(SDL_Event* e);
    void ProcessMouseMovement(SDL_Event* e);
    void ProcessMouseScroll(SDL_Event* e);

    PersperctiveInfo mPerspectiveInfo;
};