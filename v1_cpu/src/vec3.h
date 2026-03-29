#pragma once
#include <cmath>
#include <algorithm>

// =============================================================
//  Vec3 : vecteur 3D (float) utilisé pour positions, directions
//         et couleurs (RGB).
//  Équation du rayon : p = O + d*t  →  voir ray.h
// =============================================================
struct Vec3 {
    float x, y, z;

    Vec3()                          : x(0.f), y(0.f), z(0.f)  {}
    Vec3(float x, float y, float z) : x(x),   y(y),   z(z)    {}
    explicit Vec3(float v)          : x(v),   y(v),   z(v)    {}

    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(float t)       const { return {x*t,   y*t,   z*t};   }
    Vec3 operator/(float t)       const { return {x/t,   y/t,   z/t};   }
    Vec3 operator-()              const { return {-x, -y, -z};           }
    // Produit composante par composante (pour la couleur)
    Vec3 operator*(const Vec3& v) const { return {x*v.x, y*v.y, z*v.z}; }

    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator*=(float t)       { x*=t;   y*=t;   z*=t;   return *this; }

    float dot(const Vec3& v)  const { return x*v.x + y*v.y + z*v.z; }
    Vec3  cross(const Vec3& v) const {
        return { y*v.z - z*v.y,
                 z*v.x - x*v.z,
                 x*v.y - y*v.x };
    }

    float length_sq() const { return x*x + y*y + z*z; }
    float length()    const { return std::sqrt(length_sq()); }

    Vec3 normalized() const {
        float l = length();
        return (l > 1e-8f) ? (*this / l) : Vec3(0.f);
    }

    // Clamp chaque composante dans [0,1]
    Vec3 clamped() const {
        return { std::max(0.f, std::min(1.f, x)),
                 std::max(0.f, std::min(1.f, y)),
                 std::max(0.f, std::min(1.f, z)) };
    }
};

inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
