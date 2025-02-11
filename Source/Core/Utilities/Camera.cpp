#include "Camera.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

Camera::Camera(PersperctiveInfo info, glm::vec3 position, glm::vec3 up,
               float yaw, float pitch)
    : mPosition(position),
      mFront(glm::vec3(0.0f, 0.0f, -1.0f)),
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
      mFront(glm::vec3(0.0f, 0.0f, -1.0f)),
      mWorldUp(glm::vec3(upX, upY, upZ)),
      mEulerAngles {yaw, pitch},
      mMovementSpeed(CameraSpeed),
      mMouseSensitivity(CameraSensitivity),
      mZoom(CameraZoom) {
    Update();
}

void Camera::SetAspect(float aspect) {
    mPerspectiveInfo.mAspect = aspect;
}

glm::mat4 Camera::GetViewMatrix() {
    return glm::lookAt(mPosition, mPosition + mFront, mUp);
}

glm::mat4 Camera::GetProjectionMatrix() {
    return glm::perspective(mPerspectiveInfo.mFov, mPerspectiveInfo.mAspect,
                            mPerspectiveInfo.mNear, mPerspectiveInfo.mFar);
}

void Camera::ProcessSDLEvent(SDL_Event* e, float deltaTime) {
    if (mCaptureKeyboard)
        ProcessKeyboard(e, deltaTime);

    ProcessMouseButton(e);

    if (mCaptureMouseMovement)
        ProcessMouseMovement(e);

    ProcessMouseScroll(e);
}

void Camera::AdjustPosition(glm::vec3 lookAt, glm::vec3 extent) {
    mEulerAngles = {CameraYaw, CameraPitch};

    auto aspect = extent.x / extent.y > mPerspectiveInfo.mAspect
                    ? extent.x / mPerspectiveInfo.mAspect
                    : extent.y;

    auto dist = aspect * 0.5f / tan(mPerspectiveInfo.mFov * 0.5f);

    mPosition = lookAt + glm::vec3 {0.0f, 0.0f, extent.z * 0.5f + dist};

    Update();
}

void Camera::Update() {
    glm::vec3 front;
    front.x = cos(glm::radians(mEulerAngles.mYaw))
            * cos(glm::radians(mEulerAngles.mPitch));
    front.y = sin(glm::radians(mEulerAngles.mPitch));
    front.z = sin(glm::radians(mEulerAngles.mYaw))
            * cos(glm::radians(mEulerAngles.mPitch));

    mFront = glm::normalize(front);
    mRight = glm::normalize(glm::cross(mFront, mWorldUp));
    mUp = glm::normalize(glm::cross(mRight, mFront));
}

void Camera::ProcessKeyboard(SDL_Event* e, float deltaTime) {
    if (e->type == SDL_KEYDOWN) {
        mMovementSpeed += mMovementSpeed * 0.05f;
        float velocity = mMovementSpeed * deltaTime;
        if (e->key.keysym.sym == SDLK_w) {
            mPosition += mFront * velocity;
        }
        if (e->key.keysym.sym == SDLK_s) {
            mPosition -= mFront * velocity;
        }
        if (e->key.keysym.sym == SDLK_a) {
            mPosition -= mRight * velocity;
        }
        if (e->key.keysym.sym == SDLK_d) {
            mPosition += mRight * velocity;
        }
        if (e->key.keysym.sym == SDLK_SPACE) {
            mPosition += mUp * velocity;
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