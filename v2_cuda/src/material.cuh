#pragma once
#include "vec3.cuh"

// POD — initialisation par agrégat :
//   Material mat = {{r,g,b}, ka, kd, ks, shininess};
struct Material {
    Vec3  color;
    float ambient;
    float diffuse;
    float specular;
    float shininess;
};
