#pragma once
#include <cmath>
#include "vec3.h"
#include "ray.h"

// =============================================================
//  Camera : fenêtre d'observation couverte de pixels
//
//  Construit un repère caméra (right, up, forward) à partir
//  de la position, la cible et le vecteur "haut monde".
//  Pour chaque pixel (i,j), get_ray() produit un rayon
//  depuis l'œil vers le centre du pixel.
// =============================================================
struct Camera {
    Vec3  eye;          // position de la caméra
    Vec3  lower_left;   // coin bas-gauche du plan image
    Vec3  horizontal;   // vecteur horizontal du plan image
    Vec3  vertical;     // vecteur vertical   du plan image
    int   width, height;

    // eye    : position de la caméra
    // target : point visé
    // up     : vecteur "haut" monde (généralement Y+)
    // fov    : champ de vision vertical en degrés
    // w, h   : résolution en pixels
    Camera(const Vec3& eye, const Vec3& target, const Vec3& up,
           float fov_deg, int w, int h)
        : eye(eye), width(w), height(h)
    {
        float aspect  = static_cast<float>(w) / h;
        float theta   = fov_deg * static_cast<float>(M_PI) / 180.f;
        float half_h  = std::tan(theta * 0.5f);
        float half_w  = aspect * half_h;

        Vec3 forward = (target - eye).normalized();
        Vec3 right   = forward.cross(up).normalized();
        Vec3 up_cam  = right.cross(forward);        // up corrigé

        // lower_left = eye + forward - right*half_w - up*half_h
        lower_left = eye + forward
                         - right  * half_w
                         - up_cam * half_h;

        horizontal = right  * (2.f * half_w);
        vertical   = up_cam * (2.f * half_h);
    }

    // u ∈ [0,1] : fraction horizontale,  v ∈ [0,1] : fraction verticale
    Ray get_ray(float u, float v) const {
        Vec3 target = lower_left + horizontal * u + vertical * v;
        return Ray(eye, target - eye);
    }
};
