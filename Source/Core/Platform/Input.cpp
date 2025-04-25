#include "Input.h"

namespace IntelliDesign_NS::Core::Input {

KeyboardInput::KeyboardInput(::std::pmr::memory_resource* pMemPool)
    : pMemPool(pMemPool),
      mCustomKeyInfos(pMemPool),
      mRegisterdKeyMap(pMemPool) {}

void KeyboardInput::RegisterKey(Type_Key key) {
    mRegisterdKeyMap.try_emplace(key, 0.0f);
    mCustomKeyInfos.try_emplace(
        key, MemoryPool::Type_STLVector<KeyEventInfos>(pMemPool));
}

void KeyboardInput::RegisterKeyEvent(
    Type_Key key, MemoryPool::Type_STLVector<KeyEventInfos> const& infos) {
    RegisterKey(key);

    auto& keyInfos = mCustomKeyInfos.at(key);
    keyInfos.reserve(keyInfos.size() + infos.size());
    for (auto& info : infos) {
        keyInfos.push_back(info);
    }
}

void KeyboardInput::UnregisterKey(Type_Key key) {
    mRegisterdKeyMap.erase(key);
    mCustomKeyInfos.erase(key);
}

void KeyboardInput::Update(float deltaTime) {
    for (auto& [key, state] : mRegisterdKeyMap) {
        if (state.state == KeyState::Pressed) {
            state.duration += deltaTime;
        } else {
            state.duration = 0.0f;
        }
    }
}

void KeyboardInput::PollEvents(SDL_Event* e) {
    if (e->type == SDL_KEYDOWN) {
        auto key = static_cast<Type_Key>(e->key.keysym.scancode);
        if (mRegisterdKeyMap.contains(key)) {
            auto& stateInfo = mRegisterdKeyMap.at(key);
            stateInfo.state = KeyState::Pressed;
            for (auto& info : mCustomKeyInfos.at(key)) {
                if (info.checkEvent == KeyEvent::OnPress) {
                    info.func(stateInfo.duration);
                }
            }
        }
    } else if (e->type == SDL_KEYUP) {
        auto key = static_cast<Type_Key>(e->key.keysym.scancode);
        if (mRegisterdKeyMap.contains(key)) {
            auto& stateInfo = mRegisterdKeyMap.at(key);
            stateInfo.state = KeyState::Released;
            for (auto& info : mCustomKeyInfos.at(key)) {
                if (info.checkEvent == KeyEvent::OnRelease) {
                    info.func(stateInfo.duration);
                }
            }
        }
    }
}

bool KeyboardInput::IsKeyHeld(Type_Key key, float* duration) const {
    auto const& it = mRegisterdKeyMap.find(key);
    if (it == mRegisterdKeyMap.end()) {
        return false;
    }

    if (it->second.state == KeyState::Pressed) {
        auto& stateInfo = it->second;
        if (duration)
            *duration = stateInfo.duration;
        return true;
    } else {
        return false;
    }
}

}  // namespace IntelliDesign_NS::Core::Input