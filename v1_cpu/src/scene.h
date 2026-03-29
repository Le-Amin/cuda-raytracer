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

// Profondeur maximale des réflexions
#define MAX_DEPTH 5

// =============================================================
//  HitRecord : résultat d'une intersection rayon-objet
//  object_type : 0 = sphère, 1 = plan (pour le motif damier)
// =============================================================
struct HitRecord {
    float    t           = std::numeric_limits<float>::max();
    Vec3     point       = {};
    Vec3     normal      = {};
    Material material    = {};
    bool     hit         = false;
    int      object_type = -1;  // 0 = sphère, 1 = plan
};

// =============================================================
//  Scene : conteneur d'objets + sources de lumière
//          + logique de rendu (Phong + ombres + réflexions + damier)
// =============================================================
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Plane>  planes;
    std::vector<Light>  lights;

    Vec3 background = Vec3(0.f);

    void add(const Sphere& s) { spheres.push_back(s); }
    void add(const Plane&  p) { planes.push_back(p);  }
    void add(const Light&  l) { lights.push_back(l);  }

    // ----------------------------------------------------------------
    //  Trouve l'intersection la plus proche
    // ----------------------------------------------------------------
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;
        rec.t   = std::numeric_limits<float>::max();
        rec.hit = false;

        for (const auto& sphere : spheres) {
            float t = sphere.intersect(ray);
            if (t > 0.f && t < rec.t) {
                rec.t           = t;
                rec.point       = ray.at(t);
                rec.normal      = sphere.normal_at(rec.point);
                rec.material    = sphere.material;
                rec.hit         = true;
                rec.object_type = 0;  // sphère
            }
        }

        for (const auto& plane : planes) {
            float t = plane.intersect(ray);
            if (t > 0.f && t < rec.t) {
                rec.t           = t;
                rec.point       = ray.at(t);
                rec.normal      = plane.normal_at(rec.point);
                rec.material    = plane.material;
                rec.hit         = true;
                rec.object_type = 1;  // plan
            }
        }

        return rec;
    }

    // ----------------------------------------------------------------
    //  Test d'ombre
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
    //  Motif damier sur le sol (coordonnées XZ)
    // ----------------------------------------------------------------
    Vec3 checkerboard_color(const Vec3& point, const Vec3& base_color) const {
        float scale = 2.0f;
        int cx = static_cast<int>(std::floor(point.x / scale));
        int cz = static_cast<int>(std::floor(point.z / scale));
        if ((cx + cz) & 1)
            return base_color;
        else
            return base_color * 0.15f;
    }

    // ----------------------------------------------------------------
    //  Calcul Phong local (sans réflexion)
    // ----------------------------------------------------------------
    Vec3 compute_phong(const Ray& ray, const HitRecord& rec) const {
        const Material& mat = rec.material;
        Vec3 N = rec.normal;
        Vec3 V = (-ray.direction).normalized();

        // Appliquer le damier si l'objet est un plan
        Vec3 base_color = mat.color;
        if (rec.object_type == 1)
            base_color = checkerboard_color(rec.point, mat.color);

        // Terme ambiant
        Vec3 color = base_color * mat.ambient;

        for (const auto& light : lights) {
            if (in_shadow(rec.point, light)) continue;

            Vec3  L  = (light.position - rec.point).normalized();
            float NL = N.dot(L);
            if (NL <= 0.f) continue;

            // Diffus
            Vec3 diffuse = base_color * (mat.diffuse * NL);

            // Spéculaire
            Vec3  R  = (N * (2.f * NL) - L).normalized();
            float RV = std::max(0.f, R.dot(V));
            Vec3 specular = Vec3(mat.specular) * std::pow(RV, mat.shininess);

            color += (diffuse + specular) * light.color * light.intensity;
        }

        return color;
    }

    // ----------------------------------------------------------------
    //  Shading avec réflexions itératives
    //  Même algorithme que la version CUDA
    // ----------------------------------------------------------------
    Vec3 shade(const Ray& initial_ray) const {
        Vec3 accumulated(0.f);
        Vec3 attenuation(1.f, 1.f, 1.f);
        Ray  current_ray = initial_ray;

        for (int depth = 0; depth < MAX_DEPTH; ++depth) {
            HitRecord rec = intersect(current_ray);
            if (!rec.hit) break;

            Vec3 local_color = compute_phong(current_ray, rec);

            float refl = rec.material.reflectivity;
            accumulated += attenuation * local_color * (1.f - refl);

            if (refl < 1e-4f) break;

            // Rayon réfléchi R = D - 2(D·N)N
            Vec3 D = current_ray.direction;
            Vec3 N = rec.normal;
            Vec3 reflect_dir = D - N * (2.f * D.dot(N));

            current_ray = Ray(rec.point + N * 1e-3f, reflect_dir);
            attenuation *= refl;
        }

        return accumulated.clamped();
    }
};
