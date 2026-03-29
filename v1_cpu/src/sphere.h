#pragma once
#include <cmath>
#include "vec3.h"
#include "ray.h"
#include "material.h"

// =============================================================
//  Sphere : surface implicite définie par (p-c)·(p-c) - r² = 0
//
//  Intersection rayon-sphère (formule [5] → équation [6]) :
//    En substituant p = O + t*d dans (p-c)·(p-c) = r² :
//      a*t² + b*t + c = 0
//    avec :
//      a = d·d
//      b = 2*(O-c)·d
//      c = (O-c)·(O-c) - r²
//    discriminant Δ = b²-4ac
//    Si Δ < 0 : pas d'intersection
//    Sinon : t = (-b ± √Δ) / (2a),  on prend le plus petit t > ε
// =============================================================
struct Sphere {
    Vec3     center;
    float    radius;
    Material material;

    Sphere(const Vec3& c, float r, const Material& mat)
        : center(c), radius(r), material(mat) {}

    // Retourne t ∈ (ε, +∞) si intersection, sinon -1
    float intersect(const Ray& ray) const {
        Vec3  oc = ray.origin - center;
        float a  =  ray.direction.dot(ray.direction);
        float b  =  2.f * oc.dot(ray.direction);
        float c  =  oc.dot(oc) - radius * radius;
        float disc = b*b - 4.f*a*c;

        if (disc < 0.f) return -1.f;

        float sq = std::sqrt(disc);
        float t1 = (-b - sq) / (2.f * a);
        float t2 = (-b + sq) / (2.f * a);

        const float EPS = 1e-4f;
        if (t1 > EPS) return t1;
        if (t2 > EPS) return t2;
        return -1.f;
    }

    // Normale sortante au point p (supposé sur la sphère)
    Vec3 normal_at(const Vec3& p) const {
        return (p - center).normalized();
    }
};
