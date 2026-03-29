#pragma once
#include "vec3.cuh"

// POD — Light l = {{px,py,pz}, {r,g,b}, intensity};
struct Light {
    Vec3  position;
    Vec3  color;
    float intensity;
};
