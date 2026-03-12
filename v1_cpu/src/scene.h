#pragma once
#include <vector>
#include <limits>
#include <cmath>
#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "light.h"
#include "sphere.h"
#include "plane.h"

// =============================================================
//  HitRecord : résultat d'une intersection rayon-objet
// =============================================================
struct HitRecord {
    float    t        = std::numeric_limits<float>::max();
    Vec3     point    = {};
    Vec3     normal   = {};
    Material material = {};
    bool     hit      = false;
};

// =============================================================
//  Scene : conteneur d'objets + sources de lumière
//          + logique de rendu (Phong + ombres)
// =============================================================
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Plane>  planes;
    std::vector<Light>  lights;

    // Couleur d'arrière-plan (noir par défaut)
    Vec3 background = Vec3(0.f);

    void add(const Sphere& s) { spheres.push_back(s); }
    void add(const Plane&  p) { planes.push_back(p);  }
    void add(const Light&  l) { lights.push_back(l);  }

    // ----------------------------------------------------------------
    //  Trouve l'intersection la plus proche (t_min ∈ [ε, +∞[)
    // ----------------------------------------------------------------
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;
        rec.t = std::numeric_limits<float>::max();
        rec.hit = false;

        for (const auto& sphere : spheres) {
            float t = sphere.intersect(ray);
            if (t > 0.f && t < rec.t) {
                rec.t        = t;
                rec.point    = ray.at(t);
                rec.normal   = sphere.normal_at(rec.point);
                rec.material = sphere.material;
                rec.hit      = true;
            }
        }

        for (const auto& plane : planes) {
            float t = plane.intersect(ray);
            if (t > 0.f && t < rec.t) {
                rec.t        = t;
                rec.point    = ray.at(t);
                rec.normal   = plane.normal_at(rec.point);
                rec.material = plane.material;
                rec.hit      = true;
            }
        }

        return rec;
    }

    // ----------------------------------------------------------------
    //  Teste si un point est à l'ombre d'une lumière donnée
    //  (rayon d'ombre depuis hit_point vers la lumière)
    // ----------------------------------------------------------------
    bool in_shadow(const Vec3& hit_point, const Light& light) const {
        Vec3  to_light = light.position - hit_point;
        float dist     = to_light.length();
        Ray   shadow_ray(hit_point, to_light);

        for (const auto& sphere : spheres) {
            float t = sphere.intersect(shadow_ray);
            if (t > 1e-4f && t < dist) return true;
        }
        for (const auto& plane : planes) {
            float t = plane.intersect(shadow_ray);
            if (t > 1e-4f && t < dist) return true;
        }
        return false;
    }

    // ----------------------------------------------------------------
    //  Calcul de la couleur d'un pixel (modèle de Phong)
    //
    //  I = ka*Ia + Σ_lumières (si pas à l'ombre) :
    //        kd * max(0, N·L) * Id
    //      + ks * max(0, R·V)^n * Is
    //
    //  avec R = 2*(N·L)*N - L  (direction réfléchie)
    //       V = -ray.direction  (direction vers l'observateur)
    // ----------------------------------------------------------------
    Vec3 shade(const Ray& ray) const {
        HitRecord rec = intersect(ray);

        if (!rec.hit) return background;

        const Material& mat = rec.material;
        Vec3 N = rec.normal;

        // Terme ambiant (lumière diffuse globale)
        Vec3 color = mat.color * mat.ambient;

        // Direction vers l'observateur
        Vec3 V = (-ray.direction).normalized();

        for (const auto& light : lights) {
            if (in_shadow(rec.point, light)) continue;

            Vec3  L  = (light.position - rec.point).normalized();
            float NL = N.dot(L);
            if (NL <= 0.f) continue;  // face arrière

            // Terme diffus  : kd * (N·L) * Id
            Vec3 diffuse = mat.color * (mat.diffuse * NL);

            // Direction réfléchie R = 2*(N·L)*N - L
            Vec3  R  = (N * (2.f * NL) - L).normalized();
            float RV = std::max(0.f, R.dot(V));

            // Terme spéculaire : ks * (R·V)^n * Is
            Vec3 specular = Vec3(mat.specular)
                            * std::pow(RV, mat.shininess);

            Vec3 light_contrib = (diffuse + specular)
                                 * light.color * light.intensity;
            color += light_contrib;
        }

        return color.clamped();
    }
};
