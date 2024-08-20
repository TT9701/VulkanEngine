#pragma once

#include <SDL_events.h>
#include <glm/glm.hpp>

enum class CameraMovement : uint32_t { Forward, Backward, Left, Right };

constexpr float CameraYaw = -90.0f;
constexpr float CameraPitch = 0.0f;
constexpr float CameraSpeed = 5.f;
constexpr float CameraSensitivity = 0.1f;
constexpr float CameraZoom = 45.0f;

class Camera {
public:
    // camera Attributes
    glm::vec3 mPosition;
    glm::vec3 mFront;
    glm::vec3 mUp;
    glm::vec3 mRight;
    glm::vec3 mWorldUp;

    // euler Angles
    float mYaw;
    float mPitch;

    // camera options
    float mMovementSpeed;
    float mMouseSensitivity;
    float mZoom;

    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = CameraYaw,
           float pitch = CameraPitch);

    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ,
           float yaw, float pitch);

    glm::mat4 GetViewMatrix();

    void ProcessSDLEvent(SDL_Event* e, float deltaTime);

private:
    void Update();

    void ProcessKeyboard(SDL_Event* e, float deltaTime);
    void ProcessMouseMovement(SDL_Event* e);
    void ProcessMouseScroll(SDL_Event* e);
};