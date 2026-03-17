#pragma once
#include <math.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"

struct Plane {
    Vec3     point;
    Vec3     normal;
    Material material;

    __host__ __device__ Plane() {}

    __host__ __device__
    Plane(const Vec3& p, const Vec3& n, const Material& mat)
        : point(p), normal(n.normalized()), material(mat) {}

    __host__ __device__
    float intersect(const Ray& ray) const {
        float denom = ray.direction.dot(normal);
        if (fabsf(denom) < 1e-6f) return -1.f;
        float t = (point - ray.origin).dot(normal) / denom;
        return (t > 1e-4f) ? t : -1.f;
    }

    __host__ __device__
    Vec3 normal_at(const Vec3& /*p*/) const { return normal; }
};
