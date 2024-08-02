#include "Camera.hpp"

#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/transform.hpp"

glm::mat4 Camera::GetViewMatrix() {
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation    = GetRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::GetRotationMatrix() {
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3 {1.f, 0.f, 0.f});
    glm::quat yawRotation   = glm::angleAxis(yaw, glm::vec3 {0.f, -1.f, 0.f});

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::ProcessSDLEvent(SDL_Event* e) {
    if (e->type == SDL_KEYDOWN) {
        if (e->key.keysym.sym == SDLK_w) {
            velocity.z = -1;
        }
        if (e->key.keysym.sym == SDLK_s) {
            velocity.z = 1;
        }
        if (e->key.keysym.sym == SDLK_a) {
            velocity.x = -1;
        }
        if (e->key.keysym.sym == SDLK_d) {
            velocity.x = 1;
        }
    }

    if (e->type == SDL_KEYUP) {
        if (e->key.keysym.sym == SDLK_w) {
            velocity.z = 0;
        }
        if (e->key.keysym.sym == SDLK_s) {
            velocity.z = 0;
        }
        if (e->key.keysym.sym == SDLK_a) {
            velocity.x = 0;
        }
        if (e->key.keysym.sym == SDLK_d) {
            velocity.x = 0;
        }
    }

    if (e->type == SDL_MOUSEMOTION) {
        yaw += (float)e->motion.xrel / 1000.f;
        pitch -= (float)e->motion.yrel / 1000.f;
    }
}

void Camera::Update() {
    glm::mat4 cameraRotation = GetRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.005f, 0.f));
}