#pragma once

#include "Core/Utilities/MemoryPool.h"

#include <SDL2/SDL_events.h>

#include <array>
#include <functional>
#include <optional>
#include <span>

namespace IntelliDesign_NS::Core::Input {

class InputBase {
public:
    virtual ~InputBase() = default;
};

using Type_Key = uint8_t;

struct KeyEventInfos;
using Type_CallbackFunc = ::std::function<void(float /* duration */)>;

static constexpr uint32_t MaxKeys = 256;

enum class KeyState { Released, Pressed, Num };

enum class KeyEvent { OnRelease, OnPress, Num };

struct KeyStateInfos {
    float duration;
    KeyState state {KeyState::Released};
};

struct KeyEventInfos {
    KeyEvent checkEvent;
    Type_CallbackFunc func;
};

class KeyboardInput : public InputBase {
    // key - duration
    using Type_RegisterdKeyMap =
        MemoryPool::Type_STLUnorderedMap<Type_Key, KeyStateInfos>;

    // key - event infos
    using Type_KeyInfosMap = MemoryPool::Type_STLUnorderedMap<
        Type_Key, MemoryPool::Type_STLVector<KeyEventInfos>>;

public:
    KeyboardInput(::std::pmr::memory_resource* pMemPool);

    void RegisterKey(Type_Key key);

    void RegisterKeyEvent(
        Type_Key key, MemoryPool::Type_STLVector<KeyEventInfos> const& infos);

    void UnregisterKey(Type_Key key);

    void Update(float deltaTime);

    void PollEvents(SDL_Event* e);

    /*
     * check key states
    */
    bool IsKeyHeld(Type_Key key, float* duration = nullptr) const;

private:
    ::std::pmr::memory_resource* pMemPool;

    Type_RegisterdKeyMap mRegisterdKeyMap;

    // respond by checking key state
    Type_KeyInfosMap mCustomKeyInfos;
};

}  // namespace IntelliDesign_NS::Core::Input