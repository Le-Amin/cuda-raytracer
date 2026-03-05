#pragma once
#include "vec3.h"

// =============================================================
//  Material : propriétés d'un matériau (modèle de Phong)
//
//  Couleur finale = ka*Ia
//                 + Σ_lumières [ kd * max(0, N·L) * Id
//                              + ks * max(0, R·V)^shininess * Is ]
//
//  ka  : coefficient ambiant
//  kd  : coefficient diffus
//  ks  : coefficient spéculaire
//  shininess : exposant Phong (plus grand = reflet plus net)
// =============================================================
struct Material {
    Vec3  color;       // couleur de base (diffuse)
    float ambient;     // ka
    float diffuse;     // kd
    float specular;    // ks
    float shininess;   // n (exposant Phong)

    Material()
        : color(1.f, 1.f, 1.f),
          ambient(0.1f), diffuse(0.8f), specular(0.3f), shininess(32.f) {}

    Material(const Vec3& col,
             float ka, float kd, float ks, float shin)
        : color(col),
          ambient(ka), diffuse(kd), specular(ks), shininess(shin) {}
};
