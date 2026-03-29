// =============================================================
//  Raytracer CPU — v1 (amélioré)
//
//  Fonctionnalités identiques à la version CUDA v2 :
//    - Éclairage Phong (ambiant + diffus + spéculaire + ombres)
//    - Réflexions itératives (MAX_DEPTH rebonds)
//    - Anti-aliasing par supersampling (AA_SAMPLES rayons/pixel)
//    - Motif damier sur le sol
//
//  Usage :
//    ./raytracer_cpu [largeur hauteur]
// =============================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "light.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "scene.h"

// ---- Paramètres de rendu (identiques à la version CUDA) ----
#define AA_SAMPLES 4

// Résolution par défaut
static const int WIDTH  = 1280;
static const int HEIGHT =  720;

// ----------------------------------------------------------------
//  Convertit une couleur [0,1]³ en octet [0,255]
// ----------------------------------------------------------------
static inline int to_byte(float v) {
    return static_cast<int>(255.99f * v);
}

// ----------------------------------------------------------------
//  Fonction de hachage pour le jitter (identique à la version CUDA)
// ----------------------------------------------------------------
static inline float hash_float(int x, int y, int s) {
    unsigned int h = (unsigned int)(x * 374761393 + y * 668265263 + s * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)(h & 0x00FFFFFFu) / (float)0x01000000u;
}

// ----------------------------------------------------------------
//  Écriture PPM (format P3 — ASCII)
// ----------------------------------------------------------------
static void write_ppm(const std::string& filename,
                      const std::vector<Vec3>& pixels,
                      int width, int height)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "[ERROR] Impossible d'ouvrir : " << filename << "\n";
        return;
    }
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            const Vec3& c = pixels[j * width + i];
            out << to_byte(c.x) << " "
                << to_byte(c.y) << " "
                << to_byte(c.z) << "\n";
        }
    }
    std::cout << "[OK] Image sauvegardée : " << filename << "\n";
}

// ----------------------------------------------------------------
//  Construction de la scène (identique à v2 CUDA)
// ----------------------------------------------------------------
static Scene build_scene()
{
    Scene scene;
    scene.background = Vec3(0.f, 0.f, 0.f);

    //                         couleur                   ka   kd   ks   shin  refl
    Material mat_floor(Vec3(0.8f, 0.8f, 0.8f), 0.1f, 0.9f, 0.1f,   8.f, 0.2f);
    Material mat_red  (Vec3(0.9f, 0.1f, 0.1f), 0.1f, 0.8f, 0.2f,  16.f, 0.1f);
    Material mat_green(Vec3(0.1f, 0.8f, 0.2f), 0.1f, 0.7f, 0.5f,  64.f, 0.3f);
    Material mat_blue (Vec3(0.1f, 0.3f, 0.9f), 0.1f, 0.5f, 0.9f, 128.f, 0.8f);

    scene.add(Plane(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), mat_floor));

    scene.add(Sphere(Vec3(-2.f,  0.f, -5.f), 1.f,  mat_red));
    scene.add(Sphere(Vec3( 0.f,  0.5f,-4.f), 1.5f, mat_green));
    scene.add(Sphere(Vec3( 2.f,  0.f, -5.f), 1.f,  mat_blue));

    scene.add(Light(Vec3(-3.f, 5.f, -2.f), Vec3(1.f, 1.f, 1.f),     1.0f));
    scene.add(Light(Vec3( 4.f, 3.f, -1.f), Vec3(1.f, 0.9f, 0.7f),   0.6f));

    return scene;
}

// ----------------------------------------------------------------
//  Main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    int w = WIDTH;
    int h = HEIGHT;
    if (argc >= 3) {
        w = std::atoi(argv[1]);
        h = std::atoi(argv[2]);
    }

    std::cout << "=== Raytracer CPU v1 (amélioré) ===\n";
    std::cout << "Résolution : " << w << " x " << h << "\n";
    std::cout << "AA samples : " << AA_SAMPLES
              << "  |  Max réflexions : " << MAX_DEPTH << "\n";

    // Caméra (même position que v2 CUDA)
    Camera cam(
        Vec3(0.f, 1.f, 2.f),
        Vec3(0.f, 0.f, -4.f),
        Vec3(0.f, 1.f, 0.f),
        60.f, w, h
    );

    Scene scene = build_scene();

    std::vector<Vec3> pixels(w * h);

    // ---- Boucle principale avec anti-aliasing ----
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < h; ++j) {
        if (j % 50 == 0)
            std::cout << "  Ligne " << j << " / " << h << "\r" << std::flush;

        for (int i = 0; i < w; ++i) {
            Vec3 color(0.f);

            // Grille 2×2 stratifiée avec jitter (même que CUDA)
            for (int s = 0; s < AA_SAMPLES; ++s) {
                float jx = hash_float(i, j, s * 2);
                float jy = hash_float(i, j, s * 2 + 1);

                float sx = (float)(s % 2) * 0.5f + jx * 0.5f;
                float sy = (float)(s / 2) * 0.5f + jy * 0.5f;

                float u = (i + sx) / w;
                float v = (j + sy) / h;

                Ray ray = cam.get_ray(u, v);
                color += scene.shade(ray);
            }

            pixels[j * w + i] = (color * (1.f / AA_SAMPLES)).clamped();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nTemps de rendu : " << elapsed << " s\n";
    std::cout << "Performance    : "
              << static_cast<double>(w * h) / elapsed / 1e6
              << " Mpixels/s\n";

    write_ppm("../results/output_cpu.ppm", pixels, w, h);

    return 0;
}
