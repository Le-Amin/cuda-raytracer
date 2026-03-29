// =============================================================
//  Tests unitaires — Modèle de Phong et éclairage
//  Vérifie les calculs d'ombre, réflexion et éclairage
//
//  Compilation : g++ -std=c++17 -O2 -o test_phong test_phong.cpp
//  Exécution   : ./test_phong
// =============================================================

#include <iostream>
#include <cmath>
#include <cassert>
#include "../src/vec3.h"
#include "../src/ray.h"
#include "../src/material.h"
#include "../src/light.h"
#include "../src/sphere.h"
#include "../src/plane.h"
#include "../src/scene.h"

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { \
    tests_total++; \
    std::cout << "  [TEST] " << name << "... "; \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::cout << "OK\n"; \
} while(0)

static bool approx(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

// ============================================================
//  Tests éclairage
// ============================================================

void test_ambiant_only() {
    TEST("Éclairage ambiant seul (pas de lumière)");
    Scene scene;
    // Sphère avec ka = 0.3, pas de lumière
    Material mat(Vec3(1.f, 0.f, 0.f), 0.3f, 0.7f, 0.0f, 16.f);
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat));

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    Vec3 color = scene.shade(ray);

    // Sans lumière, seul le terme ambiant contribue : ka * color = 0.3 * (1,0,0)
    assert(approx(color.x, 0.3f));
    assert(approx(color.y, 0.f));
    assert(approx(color.z, 0.f));
    PASS();
}

void test_diffus() {
    TEST("Éclairage diffus (lumière frontale)");
    Scene scene;
    Material mat(Vec3(1.f, 1.f, 1.f), 0.0f, 1.0f, 0.0f, 16.f);  // Purement diffus
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat));
    // Lumière directement devant la sphère
    scene.add(Light(Vec3(0.f, 0.f, 0.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    Vec3 color = scene.shade(ray);

    // N·L devrait être ~1 (lumière alignée avec la normale)
    assert(color.x > 0.5f);  // Doit être bien éclairé
    PASS();
}

void test_ombre() {
    TEST("Ombre portée");
    Scene scene;
    Material mat(Vec3(1.f, 1.f, 1.f), 0.0f, 1.0f, 0.0f, 16.f);

    // Sphère cible (loin)
    scene.add(Sphere(Vec3(0.f, 0.f, -10.f), 1.f, mat));
    // Sphère bloquante (entre la lumière et la cible)
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat));
    // Lumière devant
    scene.add(Light(Vec3(0.f, 0.f, 0.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    // Vérifier que le point derrière la sphère bloquante est dans l'ombre
    Vec3 hit_point(0.f, 0.f, -9.f);  // Surface avant de la sphère cible
    bool shadow = scene.in_shadow(hit_point, scene.lights[0]);
    assert(shadow);
    PASS();
}

void test_pas_ombre() {
    TEST("Pas d'ombre (aucun obstacle)");
    Scene scene;
    Material mat(Vec3(1.f, 1.f, 1.f), 0.1f, 0.9f, 0.0f, 16.f);
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat));
    scene.add(Light(Vec3(0.f, 5.f, -5.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    // Point au sommet de la sphère, lumière au-dessus → pas d'ombre
    Vec3 hit_point(0.f, 1.f, -5.f);
    bool shadow = scene.in_shadow(hit_point, scene.lights[0]);
    assert(!shadow);
    PASS();
}

void test_reflexion_miroir() {
    TEST("Réflexion miroir (reflectivity = 0.8)");
    Scene scene;
    // Sphère miroir
    Material mat_mirror(Vec3(0.1f, 0.3f, 0.9f), 0.1f, 0.5f, 0.9f, 128.f, 0.8f);
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat_mirror));

    // Sphère rouge derrière le rayon (sera visible par réflexion)
    Material mat_red(Vec3(1.f, 0.f, 0.f), 0.1f, 0.9f, 0.0f, 16.f, 0.0f);
    scene.add(Sphere(Vec3(0.f, 0.f, -8.f), 1.f, mat_red));

    scene.add(Light(Vec3(0.f, 5.f, 0.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    Vec3 color = scene.shade(ray);

    // La couleur doit contenir une composante de la contribution locale
    assert(color.x >= 0.f && color.y >= 0.f && color.z >= 0.f);
    assert(color.x <= 1.f && color.y <= 1.f && color.z <= 1.f);
    PASS();
}

void test_reflexion_nulle() {
    TEST("Réflexion nulle (reflectivity = 0)");
    Scene scene;
    Material mat(Vec3(0.5f, 0.5f, 0.5f), 0.2f, 0.8f, 0.0f, 16.f, 0.0f);
    scene.add(Sphere(Vec3(0.f, 0.f, -5.f), 1.f, mat));
    scene.add(Light(Vec3(0.f, 5.f, 0.f), Vec3(1.f, 1.f, 1.f), 1.0f));

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    Vec3 color = scene.shade(ray);

    // Pas de réflexion → couleur purement locale
    assert(color.x > 0.f);
    PASS();
}

void test_fond_noir() {
    TEST("Rayon dans le vide → fond noir");
    Scene scene;
    scene.background = Vec3(0.f);

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 1.f, 0.f));  // Vers le haut, rien
    Vec3 color = scene.shade(ray);

    assert(approx(color.x, 0.f));
    assert(approx(color.y, 0.f));
    assert(approx(color.z, 0.f));
    PASS();
}

void test_damier() {
    TEST("Motif damier (alternance)");
    Scene scene;
    Material mat(Vec3(1.f, 1.f, 1.f), 0.1f, 0.9f, 0.0f, 16.f, 0.0f);
    scene.add(Plane(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 1.f, 0.f), mat));

    // Deux points adjacents devraient avoir des couleurs différentes
    Vec3 c1 = scene.checkerboard_color(Vec3(0.5f, 0.f, 0.5f), Vec3(1.f, 1.f, 1.f));
    Vec3 c2 = scene.checkerboard_color(Vec3(2.5f, 0.f, 0.5f), Vec3(1.f, 1.f, 1.f));

    // Un des deux est clair (1,1,1), l'autre sombre (0.15,0.15,0.15)
    bool different = !approx(c1.x, c2.x, 0.01f);
    assert(different);
    PASS();
}

// ---- Main ----
int main() {
    std::cout << "=== Tests unitaires : Phong et éclairage ===\n\n";

    test_ambiant_only();
    test_diffus();
    test_ombre();
    test_pas_ombre();
    test_reflexion_miroir();
    test_reflexion_nulle();
    test_fond_noir();
    test_damier();

    std::cout << "\n=== Résultat : " << tests_passed << "/" << tests_total
              << " tests passés ===\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
