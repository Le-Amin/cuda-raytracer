#pragma once
#include "vec3.cuh"

// =============================================================
//  Material : propriétés d'un matériau (modèle de Phong + réflexion)
//
//  POD — initialisation par agrégat :
//    Material mat = {{r,g,b}, ka, kd, ks, shininess, reflectivity};
//
//  reflectivity : 0 = opaque, 1 = miroir parfait
// =============================================================
struct Material {
    Vec3  color;
    float ambient;
    float diffuse;
    float specular;
    float shininess;
    float reflectivity;  // Coefficient de réflexion [0,1]
};
