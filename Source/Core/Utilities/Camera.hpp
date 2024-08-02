#pragma once

#include <SDL_events.h>
#include <glm/glm.hpp>

class Camera {
public:
    glm::vec3 velocity {0.0f};
    glm::vec3 position {0.0f, 0.0f, 5.0f};

    // vertical rotation
    float pitch {0.f};
    // horizontal rotation
    float yaw {0.f};

    glm::mat4 GetViewMatrix();
    glm::mat4 GetRotationMatrix();

    void ProcessSDLEvent(SDL_Event* e);

    void Update();
};