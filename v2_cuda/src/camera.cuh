#pragma once
#include <math.h>
#include "vec3.cuh"
#include "ray.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct Camera {
    Vec3  eye;
    Vec3  lower_left;
    Vec3  horizontal;
    Vec3  vertical;
    int   width, height;

    __host__ __device__ Camera() : width(0), height(0) {}

    __host__
    Camera(const Vec3& eye, const Vec3& target, const Vec3& up,
           float fov_deg, int w, int h)
        : eye(eye), width(w), height(h)
    {
        float aspect = (float)w / h;
        float theta  = fov_deg * (float)M_PI / 180.f;
        float half_h = tanf(theta * 0.5f);
        float half_w = aspect * half_h;

        Vec3 forward = (target - eye).normalized();
        Vec3 right   = forward.cross(up).normalized();
        Vec3 up_cam  = right.cross(forward);

        lower_left = eye + forward - right * half_w - up_cam * half_h;
        horizontal = right  * (2.f * half_w);
        vertical   = up_cam * (2.f * half_h);
    }

    __host__ __device__
    Ray get_ray(float u, float v) const {
        Vec3 t = lower_left + horizontal * u + vertical * v;
        return Ray(eye, t - eye);
    }
};
