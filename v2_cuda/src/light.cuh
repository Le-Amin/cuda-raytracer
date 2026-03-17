#pragma once
#include "vec3.cuh"

struct Light {
    Vec3  position;
    Vec3  color;
    float intensity;

    __host__ __device__
    Light(const Vec3& pos, const Vec3& col, float intens = 1.f)
        : position(pos), color(col), intensity(intens) {}

    __host__ __device__ Light() : intensity(1.f) {}
};
