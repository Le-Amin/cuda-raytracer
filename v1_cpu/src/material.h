#pragma once
#include "vec3.h"

// =============================================================
//  Material : propriétés d'un matériau (modèle de Phong + réflexion)
//
//  Couleur finale = ka*Ia
//                 + Σ_lumières [ kd * max(0, N·L) * Id
//                              + ks * max(0, R·V)^shininess * Is ]
//
//  reflectivity : 0 = opaque, 1 = miroir parfait
// =============================================================
struct Material {
    Vec3  color;         // couleur de base (diffuse)
    float ambient;       // ka
    float diffuse;       // kd
    float specular;      // ks
    float shininess;     // n (exposant Phong)
    float reflectivity;  // coefficient de réflexion [0,1]

    Material()
        : color(1.f, 1.f, 1.f),
          ambient(0.1f), diffuse(0.8f), specular(0.3f),
          shininess(32.f), reflectivity(0.f) {}

    Material(const Vec3& col,
             float ka, float kd, float ks, float shin, float refl = 0.f)
        : color(col),
          ambient(ka), diffuse(kd), specular(ks),
          shininess(shin), reflectivity(refl) {}
};
