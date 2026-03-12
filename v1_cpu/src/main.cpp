// =============================================================
//  Raytracer CPU — v1
//
//  Algorithme :
//    Pour chaque pixel (i,j) de la résolution WIDTH x HEIGHT :
//      1. Lancer un rayon depuis le centre du pixel
//      2. Trouver l'intersection la plus proche (t_min ≥ 0)
//      3. Si intersection → calculer la couleur (Phong + ombres)
//         Sinon           → couleur noire (fond)
//      4. Écrire le pixel dans l'image PPM
//
//  Scène de test : sol (plan), 3 sphères, 2 lumières
//  Sortie        : results/output_cpu.ppm
// =============================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "light.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "scene.h"

// Résolution de l'image
static const int WIDTH  = 1280;
static const int HEIGHT =  720;

// ----------------------------------------------------------------
//  Convertit une couleur [0,1]³ en octet [0,255]
// ----------------------------------------------------------------
static inline int to_byte(float v) {
    return static_cast<int>(255.99f * v);
}

// ----------------------------------------------------------------
//  Écrit un fichier PPM (format P3 — ASCII)
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
//  Construction de la scène
// ----------------------------------------------------------------
static Scene build_scene()
{
    Scene scene;
    scene.background = Vec3(0.f, 0.f, 0.f);  // fond noir

    // --- Matériaux ---
    // Sol quadrillé (diffus, peu spéculaire)
    Material mat_floor(Vec3(0.8f, 0.8f, 0.8f), 0.1f, 0.9f, 0.1f, 8.f);

    // Sphère rouge (très diffuse)
    Material mat_red(Vec3(0.9f, 0.1f, 0.1f), 0.1f, 0.8f, 0.2f, 16.f);

    // Sphère verte (diffuse + légèrement spéculaire)
    Material mat_green(Vec3(0.1f, 0.8f, 0.2f), 0.1f, 0.7f, 0.5f, 64.f);

    // Sphère bleue (très spéculaire — aspect plastique brillant)
    Material mat_blue(Vec3(0.1f, 0.3f, 0.9f), 0.1f, 0.5f, 0.9f, 128.f);

    // --- Objets ---
    // Plan horizontal : normale (0,1,0), passe par y = -1
    scene.add(Plane(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), mat_floor));

    // Sphère rouge (gauche)
    scene.add(Sphere(Vec3(-2.f, 0.f, -5.f), 1.f, mat_red));

    // Sphère verte (centre)
    scene.add(Sphere(Vec3( 0.f, 0.5f, -4.f), 1.5f, mat_green));

    // Sphère bleue (droite)
    scene.add(Sphere(Vec3( 2.f, 0.f, -5.f), 1.f, mat_blue));

    // --- Lumières ---
    // Lumière blanche principale (au-dessus à gauche)
    scene.add(Light(Vec3(-3.f, 5.f, -2.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    // Lumière secondaire chaude (à droite, moins intense)
    scene.add(Light(Vec3( 4.f, 3.f, -1.f), Vec3(1.f, 0.9f, 0.7f), 0.6f));

    return scene;
}

// ----------------------------------------------------------------
//  Main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Résolution optionnellement passée en argument
    int w = WIDTH;
    int h = HEIGHT;
    if (argc == 3) {
        w = std::atoi(argv[1]);
        h = std::atoi(argv[2]);
    }

    std::cout << "=== Raytracer CPU v1 ===\n";
    std::cout << "Résolution : " << w << " x " << h << "\n";

    // Caméra
    Camera cam(
        Vec3(0.f, 1.f, 2.f),   // œil
        Vec3(0.f, 0.f, -4.f),  // cible
        Vec3(0.f, 1.f, 0.f),   // up
        60.f,                   // FOV vertical
        w, h
    );

    // Scène
    Scene scene = build_scene();

    // Buffer de pixels
    std::vector<Vec3> pixels(w * h);

    // ---- Boucle principale (pour chaque pixel) ----
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < h; ++j) {
        if (j % 50 == 0)
            std::cout << "  Ligne " << j << " / " << h << "\r" << std::flush;

        for (int i = 0; i < w; ++i) {
            // Centre du pixel en coordonnées normalisées [0,1]
            float u = (i + 0.5f) / w;
            float v = (j + 0.5f) / h;

            // 1. Lancer le rayon depuis le centre du pixel
            Ray ray = cam.get_ray(u, v);

            // 2 & 3. Intersection + calcul de couleur (Phong)
            pixels[j * w + i] = scene.shade(ray);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nTemps de rendu : " << elapsed << " s\n";
    std::cout << "Performance    : "
              << static_cast<double>(w * h) / elapsed / 1e6
              << " Mpixels/s\n";

    // Écriture de l'image
    write_ppm("../results/output_cpu.ppm", pixels, w, h);

    return 0;
}
