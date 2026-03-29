#pragma once
#include "vec3.h"

// =============================================================
//  Light : source lumineuse ponctuelle
// =============================================================
struct Light {
    Vec3  position;   // position dans la scène
    Vec3  color;      // couleur de la lumière (RGB ∈ [0,1])
    float intensity;  // intensité multiplicative

    Light(const Vec3& pos, const Vec3& col, float intens = 1.f)
        : position(pos), color(col), intensity(intens) {}
};
