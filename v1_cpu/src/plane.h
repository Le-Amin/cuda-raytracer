#pragma once
#include <cmath>
#include "vec3.h"
#include "ray.h"
#include "material.h"

// =============================================================
//  Plane : plan défini par un point a et une normale n
//
//  Intersection rayon-plan (formule [3] → [4]) :
//    Un point p est sur le plan ssi (p-a)·n = 0
//    Avec p = O + t*d :
//      (O + t*d - a)·n = 0
//      t = (a - O)·n / (d·n)
//    Si d·n ≈ 0 : rayon parallèle au plan → pas d'intersection
// =============================================================
struct Plane {
    Vec3     point;     // un point a appartenant au plan
    Vec3     normal;    // normale n (normalisée)
    Material material;

    Plane(const Vec3& p, const Vec3& n, const Material& mat)
        : point(p), normal(n.normalized()), material(mat) {}

    // Retourne t > ε si intersection, sinon -1
    float intersect(const Ray& ray) const {
        float denom = ray.direction.dot(normal);
        if (std::fabs(denom) < 1e-6f) return -1.f;  // rayon || plan

        float t = (point - ray.origin).dot(normal) / denom;
        return (t > 1e-4f) ? t : -1.f;
    }

    // La normale est constante sur tout le plan
    Vec3 normal_at(const Vec3& /*p*/) const { return normal; }
};
