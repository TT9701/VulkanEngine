#pragma once

#include <functional>
#include <string>
#include <vector>

#include <SDL2/SDL_events.h>
#include <SDL_timer.h>
#include "Core/Utilities/Defines.h"

/**
 * @brief 使用 SDL2 库创建各平台兼容的窗口系统
 * @details 管理窗口系统的尺寸信息、窗口的创建与销毁、窗口事件的处理、以及获取创建图形系统实例所需的拓展信息
 */
class SDLWindow {
public:
    /**
     * @brief 根据窗口名字、尺寸创建SDL窗口对象，并记录相关信息；创建 event 事件对象。
     *
     * @param name 窗口名字。
     * @param width 窗口宽度。
     * @param height 窗口高度。
     */
    SDLWindow(const char* name, int width, int height);

    /**
     * @brief 销毁窗口 event 事件对象，销毁窗口对象。
     */
    ~SDLWindow();

    /**
     * @brief 定义窗口不能复制不能移动。
     */
    CLASS_NO_COPY_MOVE(SDLWindow);

public:
    /**
      * @brief 处理各类窗口 event 事件。
      * 
      * @param quit 标志窗口程序退出。
      * @param stopRendering 标志窗口程序停止渲染。
      * @param eventFunc 事件处理回调函数。
      * @param onWindowResized 窗口大小修改时调用的回调函数。
      */
    void PollEvents(bool& quit, bool& stopRendering,
                    ::std::function<void(SDL_Event*)>&& eventFunc,
                    ::std::function<void()>&& onWindowResized);

public:
    /**
      * @brief 获取创建 Vulkan 实例所需要的拓展名。
      * 
      * @return std::vector<std::string> 创建 Vulkan 实例所需要的拓展名。
      */
    std::vector<std::string> GetVulkanInstanceExtension() const;

    /**
     * @brief 获得 SDL_Window 窗口指针
     * 
     * @return SDL_Window* 窗口指针
     */
    SDL_Window* GetPtr() const { return mWindow; }

    /**
     * @brief 获得窗口宽度尺寸
     * 
     * @return int 窗口宽度尺寸
     */
    int GetWidth() const { return mWidth; }

    /**
     * @brief 获得窗口高度尺寸
     * 
     * @return int 窗口高度尺寸
     */
    int GetHeight() const { return mHeight; }

    /**
     * @brief 获得窗口事件
     * 
     * @return SDL_Event* 窗口事件指针
     */
    SDL_Event* GetEvent() const { return mEvent; }

private:
    ::std::string mName;  ///< 窗口名字
    int mWidth;           ///< 窗口宽度
    int mHeight;          ///< 窗口高度

    SDL_Window* mWindow;          ///< SDL 窗口指针
    SDL_Event* mEvent {nullptr};  ///< SDL 窗口事件
};