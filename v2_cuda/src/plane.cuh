#pragma once
#include <math.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"

// POD — Plane p = {{px,py,pz}, {nx,ny,nz}, material};
// ATTENTION : normaliser la normale avant de remplir la struct
struct Plane {
    Vec3     point;
    Vec3     normal;
    Material material;

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
