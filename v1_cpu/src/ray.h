#pragma once
#include "vec3.h"

// =============================================================
//  Ray : rayon défini par p = O + d*t  (formule [1])
//    O  = origine
//    d  = direction (normalisée)
//    t  ∈ [0, +∞[  pour les intersections devant la caméra
// =============================================================
struct Ray {
    Vec3 origin;
    Vec3 direction;  // toujours normalisée

    Ray(const Vec3& o, const Vec3& d)
        : origin(o), direction(d.normalized()) {}

    // Retourne le point p = O + d*t
    Vec3 at(float t) const { return origin + direction * t; }
};
