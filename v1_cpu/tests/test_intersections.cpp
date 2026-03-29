// =============================================================
//  Tests unitaires — Intersections rayon-objet
//  Vérifie les calculs d'intersection rayon-sphère et rayon-plan
//
//  Compilation : g++ -std=c++17 -O2 -o test_intersections test_intersections.cpp
//  Exécution   : ./test_intersections
// =============================================================

#include <iostream>
#include <cmath>
#include <cassert>
#include "../src/vec3.h"
#include "../src/ray.h"
#include "../src/material.h"
#include "../src/sphere.h"
#include "../src/plane.h"

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

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approx_vec(const Vec3& a, const Vec3& b, float eps = 1e-4f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

// Matériau par défaut pour les tests
static Material default_mat(Vec3(0.5f, 0.5f, 0.5f), 0.1f, 0.8f, 0.2f, 16.f);

// ============================================================
//  Tests rayon-sphère
// ============================================================

void test_sphere_hit_direct() {
    TEST("Sphère — rayon frontal");
    // Sphère au centre (0,0,-5), rayon depuis l'origine vers -Z
    Sphere s(Vec3(0.f, 0.f, -5.f), 1.f, default_mat);
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    float t = s.intersect(ray);
    assert(t > 0.f);
    assert(approx(t, 4.f));  // touche à z = -4 (surface avant)
    PASS();
}

void test_sphere_miss() {
    TEST("Sphère — rayon manqué");
    Sphere s(Vec3(0.f, 0.f, -5.f), 1.f, default_mat);
    // Rayon qui passe au-dessus
    Ray ray(Vec3(0.f, 5.f, 0.f), Vec3(0.f, 0.f, -1.f));
    float t = s.intersect(ray);
    assert(t < 0.f);  // Pas d'intersection
    PASS();
}

void test_sphere_behind() {
    TEST("Sphère — derrière le rayon");
    Sphere s(Vec3(0.f, 0.f, 5.f), 1.f, default_mat);  // Derrière l'origine
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));  // Vers -Z
    float t = s.intersect(ray);
    assert(t < 0.f);  // Pas d'intersection (sphère derrière)
    PASS();
}

void test_sphere_inside() {
    TEST("Sphère — rayon depuis l'intérieur");
    Sphere s(Vec3(0.f, 0.f, 0.f), 10.f, default_mat);  // Grande sphère
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    float t = s.intersect(ray);
    assert(t > 0.f);
    assert(approx(t, 10.f));  // Surface à distance 10
    PASS();
}

void test_sphere_tangent() {
    TEST("Sphère — rayon tangent");
    Sphere s(Vec3(1.f, 0.f, -5.f), 1.f, default_mat);
    // Rayon qui passe juste au bord (tangent)
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));
    float t = s.intersect(ray);
    // Le rayon passe à distance 1 du centre => tangent
    assert(t > 0.f);  // Doit toucher (discriminant ≈ 0)
    PASS();
}

void test_sphere_normal() {
    TEST("Sphère — normale sortante");
    Sphere s(Vec3(0.f, 0.f, -5.f), 1.f, default_mat);
    Vec3 p(0.f, 0.f, -4.f);  // Point sur la surface avant
    Vec3 n = s.normal_at(p);
    assert(approx_vec(n, Vec3(0.f, 0.f, 1.f)));  // Normale vers +Z
    PASS();
}

// ============================================================
//  Tests rayon-plan
// ============================================================

void test_plan_hit() {
    TEST("Plan — intersection simple");
    // Plan horizontal y = -1, normale vers le haut
    Plane p(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), default_mat);
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, -1.f, 0.f));
    float t = p.intersect(ray);
    assert(t > 0.f);
    assert(approx(t, 1.f));  // Touche à y = -1
    PASS();
}

void test_plan_parallel() {
    TEST("Plan — rayon parallèle");
    Plane p(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), default_mat);
    // Rayon horizontal (parallèle au plan)
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(1.f, 0.f, 0.f));
    float t = p.intersect(ray);
    assert(t < 0.f);
    PASS();
}

void test_plan_behind() {
    TEST("Plan — derrière le rayon");
    Plane p(Vec3(0.f, 1.f, 0.f), Vec3(0.f, 1.f, 0.f), default_mat);
    // Rayon vers le bas, plan au-dessus avec normale vers le haut
    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 1.f, 0.f));
    float t = p.intersect(ray);
    // t = (1 - 0) * 1 / (0*0 + 1*1 + 0*0) = 1 > 0 → intersection
    // (le rayon va vers le plan)
    assert(t > 0.f);
    PASS();
}

void test_plan_normal() {
    TEST("Plan — normale constante");
    Plane p(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), default_mat);
    Vec3 n1 = p.normal_at(Vec3(0.f, -1.f, 0.f));
    Vec3 n2 = p.normal_at(Vec3(100.f, -1.f, -50.f));
    assert(approx_vec(n1, n2));  // Même normale partout
    assert(approx_vec(n1, Vec3(0.f, 1.f, 0.f)));
    PASS();
}

void test_plan_oblique() {
    TEST("Plan — rayon oblique");
    Plane p(Vec3(0.f, -1.f, 0.f), Vec3(0.f, 1.f, 0.f), default_mat);
    // Rayon oblique depuis (0,1,0) vers (0,-1,-1) normalisé
    Ray ray(Vec3(0.f, 1.f, 0.f), Vec3(0.f, -1.f, -1.f));
    float t = p.intersect(ray);
    assert(t > 0.f);
    Vec3 hit = ray.at(t);
    assert(approx(hit.y, -1.f, 1e-3f));  // Le point est bien sur le plan
    PASS();
}

// ============================================================
//  Tests intersection la plus proche
// ============================================================

void test_closest_intersection() {
    TEST("Plus proche intersection (2 sphères)");
    Sphere s1(Vec3(0.f, 0.f, -3.f), 0.5f, default_mat);  // Proche
    Sphere s2(Vec3(0.f, 0.f, -6.f), 0.5f, default_mat);  // Loin

    Ray ray(Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, -1.f));

    float t1 = s1.intersect(ray);
    float t2 = s2.intersect(ray);

    assert(t1 > 0.f && t2 > 0.f);
    assert(t1 < t2);  // s1 est plus proche
    PASS();
}

// ---- Main ----
int main() {
    std::cout << "=== Tests unitaires : Intersections ===\n\n";

    // Sphère
    test_sphere_hit_direct();
    test_sphere_miss();
    test_sphere_behind();
    test_sphere_inside();
    test_sphere_tangent();
    test_sphere_normal();

    // Plan
    test_plan_hit();
    test_plan_parallel();
    test_plan_behind();
    test_plan_normal();
    test_plan_oblique();

    // Combiné
    test_closest_intersection();

    std::cout << "\n=== Résultat : " << tests_passed << "/" << tests_total
              << " tests passés ===\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
