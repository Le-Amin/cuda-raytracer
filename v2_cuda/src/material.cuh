#pragma once
#include "vec3.cuh"

struct Material {
    Vec3  color;
    float ambient;
    float diffuse;
    float specular;
    float shininess;

    __host__ __device__
    Material()
        : color(1.f,1.f,1.f), ambient(0.1f), diffuse(0.8f),
          specular(0.3f), shininess(32.f) {}

    __host__ __device__
    Material(const Vec3& col, float ka, float kd, float ks, float shin)
        : color(col), ambient(ka), diffuse(kd), specular(ks), shininess(shin) {}
};
