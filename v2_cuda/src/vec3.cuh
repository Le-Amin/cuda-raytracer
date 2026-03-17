#pragma once
#include <math.h>

// =============================================================
//  Vec3 CUDA — même logique que v1_cpu/vec3.h mais avec les
//  qualificatifs __device__ __host__ pour être utilisable
//  à la fois dans le kernel GPU et le code CPU hôte.
// =============================================================
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3()                          : x(0.f), y(0.f), z(0.f)  {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x),   y(y),   z(z)    {}
    __host__ __device__ explicit Vec3(float v)          : x(v),   y(v),   z(v)    {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    __host__ __device__ Vec3 operator*(float t)       const { return {x*t,   y*t,   z*t};   }
    __host__ __device__ Vec3 operator/(float t)       const { return {x/t,   y/t,   z/t};   }
    __host__ __device__ Vec3 operator-()              const { return {-x, -y, -z};           }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return {x*v.x, y*v.y, z*v.z}; }

    __host__ __device__ Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    __host__ __device__ Vec3& operator*=(float t)       { x*=t;   y*=t;   z*=t;   return *this; }

    __host__ __device__ float dot(const Vec3& v)  const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ Vec3  cross(const Vec3& v) const {
        return { y*v.z - z*v.y,
                 z*v.x - x*v.z,
                 x*v.y - y*v.x };
    }

    __host__ __device__ float length_sq() const { return x*x + y*y + z*z; }
    __host__ __device__ float length()    const { return sqrtf(length_sq()); }

    __host__ __device__ Vec3 normalized() const {
        float l = length();
        return (l > 1e-8f) ? (*this / l) : Vec3(0.f);
    }

    __host__ __device__ Vec3 clamped() const {
        return { fmaxf(0.f, fminf(1.f, x)),
                 fmaxf(0.f, fminf(1.f, y)),
                 fmaxf(0.f, fminf(1.f, z)) };
    }
};

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
