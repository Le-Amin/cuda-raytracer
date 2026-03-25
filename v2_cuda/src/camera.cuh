#pragma once
#include <math.h>
#include "vec3.cuh"
#include "ray.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// POD — utiliser Camera::create() côté hôte pour construire
struct Camera {
    Vec3  eye;
    Vec3  lower_left;
    Vec3  horizontal;
    Vec3  vertical;
    int   width, height;

    // Fonction de construction côté hôte (remplace le constructeur)
    static __host__ Camera create(const Vec3& eye, const Vec3& target,
                                  const Vec3& up, float fov_deg,
                                  int w, int h)
    {
        float aspect = (float)w / h;
        float theta  = fov_deg * (float)M_PI / 180.f;
        float half_h = tanf(theta * 0.5f);
        float half_w = aspect * half_h;

        Vec3 forward = (target - eye).normalized();
        Vec3 right   = forward.cross(up).normalized();
        Vec3 up_cam  = right.cross(forward);

        Camera cam;
        cam.eye        = eye;
        cam.width      = w;
        cam.height     = h;
        cam.lower_left = eye + forward - right * half_w - up_cam * half_h;
        cam.horizontal = right  * (2.f * half_w);
        cam.vertical   = up_cam * (2.f * half_h);
        return cam;
    }

    __host__ __device__
    Ray get_ray(float u, float v) const {
        Vec3 t = lower_left + horizontal * u + vertical * v;
        return Ray(eye, t - eye);
    }
};
