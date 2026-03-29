#pragma once
#include <math.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"

// POD — Sphere s = {{cx,cy,cz}, radius, material};
struct Sphere {
    Vec3     center;
    float    radius;
    Material material;

    __host__ __device__
    float intersect(const Ray& ray) const {
        Vec3  oc   = ray.origin - center;
        float a    = ray.direction.dot(ray.direction);
        float b    = 2.f * oc.dot(ray.direction);
        float c    = oc.dot(oc) - radius * radius;
        float disc = b*b - 4.f*a*c;
        if (disc < 0.f) return -1.f;
        float sq = sqrtf(disc);
        float t1 = (-b - sq) / (2.f * a);
        float t2 = (-b + sq) / (2.f * a);
        const float EPS = 1e-4f;
        if (t1 > EPS) return t1;
        if (t2 > EPS) return t2;
        return -1.f;
    }

    __host__ __device__
    Vec3 normal_at(const Vec3& p) const {
        return (p - center).normalized();
    }
};
