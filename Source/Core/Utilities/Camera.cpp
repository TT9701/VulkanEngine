#include "Camera.h"

#include <algorithm>

#include "Core/Platform/Input.h"
#include "MemoryPool.h"

namespace IntelliDesign_NS::Core {

namespace IDCMCore_NS = CMCore_NS;

static constexpr float MinVel = 0.01f;
static constexpr float DampFactor = 0.3f;

Camera::Camera(PersperctiveInfo info, MathCore::Float3 position,
               MathCore::Float3 up)
    : mPosition(position), mPerspectiveInfo(info) {
    SetAspect(info.mAspect);

    if (mPerspectiveInfo.mNear > mPerspectiveInfo.mFar) {
        mReversedZ = true;
    }
}

void Camera::SetAspect(float aspect) {
    mPerspectiveInfo.mAspect = aspect;

    mProj = MathCore::MatrixPerspectiveFov(mPerspectiveInfo.mFov, aspect,
                                           mPerspectiveInfo.mNear,
                                           mPerspectiveInfo.mFar);

    mProj(1, 1) *= -1.0f;

    mInvProj = MathCore::MatrixInverse(nullptr, mProj.GetSIMD());
}

void Camera::SetViewMatrix(MathCore::Mat4 const& viewMatrix) {
    using namespace IDCMCore_NS;

    Mat4 invView = MatrixInverse(nullptr, viewMatrix.GetSIMD());

    mRight = Float3 {invView(0, 0), invView(0, 1), invView(0, 2)};
    mUp = Float3 {invView(1, 0), invView(1, 1), invView(1, 2)};
    mLook = Float3 {-invView(2, 0), -invView(2, 1), -invView(2, 2)};
    mPosition = Float3 {invView(3, 0), invView(3, 1), invView(3, 2)};

    mViewDirty = true;
}

void Camera::Strafe(float d) {
    // mPosition += d*mRight
    auto s = DirectX::XMVectorReplicate(d);
    auto r = mRight.GetSIMD();
    auto p = mPosition.GetSIMD();
    mPosition = IDCMCore_NS::VectorMultiplyAdd(s, r, p);

    mViewDirty = true;
}

void Camera::Walk(float d) {
    // mPosition += d*mLook
    auto s = DirectX::XMVectorReplicate(d);
    auto l = mLook.GetSIMD();
    auto p = mPosition.GetSIMD();
    mPosition = IDCMCore_NS::VectorMultiplyAdd(s, l, p);

    mViewDirty = true;
}

void Camera::JumpUp(float d) {
    mPosition.y += d;

    mViewDirty = true;
}

void Camera::Pitch(float angle) {
    // Rotate up and look vector about the right vector.

    auto R = DirectX::XMMatrixRotationAxis(mRight.GetSIMD(), angle);

    mUp = XMVector3TransformNormal(mUp.GetSIMD(), R);
    mLook = XMVector3TransformNormal(mLook.GetSIMD(), R);

    mViewDirty = true;
}

void Camera::RotateY(float angle) {
    // Rotate the basis vectors about the world y-axis.

    auto R = DirectX::XMMatrixRotationY(angle);

    mRight = XMVector3TransformNormal(mRight.GetSIMD(), R);
    mUp = XMVector3TransformNormal(mUp.GetSIMD(), R);
    mLook = XMVector3TransformNormal(mLook.GetSIMD(), R);

    mViewDirty = true;
}

void Camera::LookAt(const MathCore::Float3& pos, const MathCore::Float3& target,
                    const MathCore::Float3& up) {
    auto L = IDCMCore_NS::Vector3Normalize(
        IDCMCore_NS::VectorSubtract(target.GetSIMD(), pos.GetSIMD()));
    auto R = IDCMCore_NS::Vector3Normalize(
        IDCMCore_NS::Vector3Cross(L, up.GetSIMD()));
    auto U = IDCMCore_NS::Vector3Cross(R, L);

    mPosition = pos;
    mLook = L;
    mRight = R;
    mUp = U;

    mViewDirty = true;
}

MathCore::Float3 Camera::GetPosition() const {
    return mPosition;
}

MathCore::Float3 Camera::GetLookAt() {
    if (mViewDirty) {
        UpdateViewMatrix();
    }

    return mLook;
}

MathCore::Mat4 Camera::GetViewMatrix() {
    if (mViewDirty) {
        UpdateViewMatrix();
    }

    return mView;
}

MathCore::Mat4 Camera::GetProjectionMatrix() const {
    return mProj;
}

MathCore::Mat4 Camera::GetViewProjMatrix() {
    return GetViewMatrix().GetSIMD() * GetProjectionMatrix().GetSIMD();
}

MathCore::Mat4 Camera::GetInvViewMatrix() {
    if (mViewDirty) {
        UpdateViewMatrix();
    }

    return mInvView;
}

MathCore::Mat4 Camera::GetInvProjectionMatrix() const {
    return mInvProj;
}

MathCore::Mat4 Camera::GetInvViewProjMatrix() {
    return MathCore::MatrixInverse(nullptr, GetViewProjMatrix().GetSIMD());
}

void Camera::ProcessSDLEvent(SDL_Event* e, float deltaTime) {
    ProcessMouseButton(e);

    if (mCaptureMouseMovement) 
        ProcessMouseMovement(e);

    ProcessMouseScroll(e);
}

void Camera::RespondToKeyboardInput(Input::KeyboardInput const& input,
                                    float deltaTime) {
    using namespace IDCMCore_NS;

    if (!mCaptureKeyboard)
        return;

    float duration = 0.0f;

    if (input.IsKeyHeld(SDL_SCANCODE_W, &duration)) {
        mZVelocity = CameraAccel * duration;
    } else {
        if (mZVelocity > MinVel)
        mZVelocity *= DampFactor;
    }

    if (input.IsKeyHeld(SDL_SCANCODE_S, &duration)) {
        mZVelocity = -CameraAccel * duration;
    } else {
        if (mZVelocity < -MinVel)
            mZVelocity *= DampFactor;
    }

    if (input.IsKeyHeld(SDL_SCANCODE_A, &duration)) {
        mXVelocity = -CameraAccel * duration;
    } else {
        if (mXVelocity < -MinVel)
            mXVelocity *= DampFactor;
    }

    if (input.IsKeyHeld(SDL_SCANCODE_D, &duration)) {
        mXVelocity = CameraAccel * duration;
    } else {
        if (mXVelocity > MinVel)
            mXVelocity *= DampFactor;
    }

    if (input.IsKeyHeld(SDL_SCANCODE_SPACE, &duration)) {
        mYVelocity = CameraAccel * duration;
    } else {
        if (mYVelocity > MinVel)
            mYVelocity *= DampFactor;
    }

    if (::std::abs(mZVelocity) > MinVel) {
        Walk(deltaTime * mZVelocity);
    }
    if (::std::abs(mXVelocity) > MinVel) {
        Strafe(deltaTime * mXVelocity);
    }
    if (::std::abs(mYVelocity) > MinVel) {
        JumpUp(deltaTime * mYVelocity);
    }
}

void Camera::AdjustPosition(MathCore::Float3 lookAt, MathCore::Float3 extent) {
    using namespace IDCMCore_NS;
    auto mul = DirectX::XMVectorReplicate(2.0f);
    auto length = DirectX::XMVector2Length(extent.GetSIMD());
    length = VectorMultiply(length, mul);
    mPosition = VectorSubtract(lookAt.GetSIMD(),
                               (VectorMultiply(mLook.GetSIMD(), length)));

    mViewDirty = true;
}

void Camera::AdjustPosition(MathCore::BoundingBox const& boundingBox,
                            MathCore::Float3 scale) {
    using namespace IDCMCore_NS;

    auto center = VectorMultiply(boundingBox.Center.GetSIMD(), scale.GetSIMD());
    auto extents =
        VectorMultiply(boundingBox.Extents.GetSIMD(), scale.GetSIMD());

    AdjustPosition(center, extents);
}

MathCore::BoundingFrustum Camera::GetFrustum() {
    using namespace IDCMCore_NS;

    float near = mPerspectiveInfo.mNear;
    float far = mPerspectiveInfo.mFar;

    if (mReversedZ)
        ::std::swap(near, far);

    BoundingFrustum frustum {
        Mat4 {MatrixPerspectiveFov(mPerspectiveInfo.mFov,
                                   mPerspectiveInfo.mAspect, near, far)}
            .GetSIMD(),
        true};

    auto invViewMat = MatrixInverse(nullptr, GetViewMatrix().GetSIMD());

    frustum.Transform(frustum, invViewMat);

    return frustum;
}

void Camera::UpdateViewMatrix() {
    using namespace IDCMCore_NS;

    if (!mViewDirty)
        return;

    auto R = mRight.GetSIMD();
    auto L = mLook.GetSIMD();
    auto P = mPosition.GetSIMD();

    L = Vector3Normalize(L);
    auto U = Vector3Normalize(Vector3Cross(R, L));

    R = Vector3Cross(L, U);

    mRight = R;
    mUp = U;
    mLook = L;

    auto view = MatrixLookAt(P, VectorAdd(P, L), U);
    mView = view;

    auto invView = MatrixInverse(nullptr, view);
    mInvView = invView;

    mViewDirty = false;
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
    if (!mCaptureMouse)
        return;

    if (e->type == SDL_MOUSEMOTION) {
        Pitch(-(float)e->motion.yrel * CameraRotationSensitivity);
        RotateY(-(float)e->motion.xrel * CameraRotationSensitivity);
    }
}

void Camera::ProcessMouseScroll(SDL_Event* e) {
    if (!mCaptureMouse)
        return;
    
    if (e->type == SDL_MOUSEWHEEL) {
        Walk((float)e->wheel.y);
    }
}

}  // namespace IntelliDesign_NS::Core