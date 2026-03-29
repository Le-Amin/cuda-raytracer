// =============================================================
//  Tests unitaires — Vec3
//  Vérifie les opérations vectorielles de base
//
//  Compilation : g++ -std=c++17 -O2 -o test_vec3 test_vec3.cpp
//  Exécution   : ./test_vec3
// =============================================================

#include <iostream>
#include <cmath>
#include <cassert>
#include "../src/vec3.h"

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

// Comparaison flottante avec tolérance
static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static bool approx_vec(const Vec3& a, const Vec3& b, float eps = 1e-5f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

// ---- Tests ----

void test_constructeurs() {
    TEST("Constructeur par défaut");
    Vec3 v;
    assert(v.x == 0.f && v.y == 0.f && v.z == 0.f);
    PASS();

    TEST("Constructeur (x,y,z)");
    Vec3 v2(1.f, 2.f, 3.f);
    assert(v2.x == 1.f && v2.y == 2.f && v2.z == 3.f);
    PASS();

    TEST("Constructeur scalaire");
    Vec3 v3(5.f);
    assert(v3.x == 5.f && v3.y == 5.f && v3.z == 5.f);
    PASS();
}

void test_addition() {
    TEST("Addition");
    Vec3 a(1.f, 2.f, 3.f);
    Vec3 b(4.f, 5.f, 6.f);
    Vec3 c = a + b;
    assert(approx_vec(c, Vec3(5.f, 7.f, 9.f)));
    PASS();

    TEST("Addition (+=)");
    Vec3 d(1.f, 1.f, 1.f);
    d += Vec3(2.f, 3.f, 4.f);
    assert(approx_vec(d, Vec3(3.f, 4.f, 5.f)));
    PASS();
}

void test_soustraction() {
    TEST("Soustraction");
    Vec3 a(5.f, 7.f, 9.f);
    Vec3 b(1.f, 2.f, 3.f);
    Vec3 c = a - b;
    assert(approx_vec(c, Vec3(4.f, 5.f, 6.f)));
    PASS();

    TEST("Négation");
    Vec3 d(1.f, -2.f, 3.f);
    Vec3 e = -d;
    assert(approx_vec(e, Vec3(-1.f, 2.f, -3.f)));
    PASS();
}

void test_multiplication() {
    TEST("Multiplication scalaire");
    Vec3 a(1.f, 2.f, 3.f);
    Vec3 b = a * 2.f;
    assert(approx_vec(b, Vec3(2.f, 4.f, 6.f)));
    PASS();

    TEST("Multiplication composante");
    Vec3 c(2.f, 3.f, 4.f);
    Vec3 d(0.5f, 0.5f, 0.5f);
    Vec3 e = c * d;
    assert(approx_vec(e, Vec3(1.f, 1.5f, 2.f)));
    PASS();
}

void test_produit_scalaire() {
    TEST("Produit scalaire");
    Vec3 a(1.f, 0.f, 0.f);
    Vec3 b(0.f, 1.f, 0.f);
    assert(approx(a.dot(b), 0.f));  // Perpendiculaires
    PASS();

    TEST("Produit scalaire (parallèles)");
    Vec3 c(1.f, 2.f, 3.f);
    assert(approx(c.dot(c), 14.f));  // 1+4+9
    PASS();
}

void test_produit_vectoriel() {
    TEST("Produit vectoriel (axes)");
    Vec3 x(1.f, 0.f, 0.f);
    Vec3 y(0.f, 1.f, 0.f);
    Vec3 z = x.cross(y);
    assert(approx_vec(z, Vec3(0.f, 0.f, 1.f)));
    PASS();

    TEST("Produit vectoriel (anti-commutativité)");
    Vec3 z2 = y.cross(x);
    assert(approx_vec(z2, Vec3(0.f, 0.f, -1.f)));
    PASS();
}

void test_norme() {
    TEST("Norme");
    Vec3 v(3.f, 4.f, 0.f);
    assert(approx(v.length(), 5.f));
    PASS();

    TEST("Norme au carré");
    assert(approx(v.length_sq(), 25.f));
    PASS();

    TEST("Normalisation");
    Vec3 n = v.normalized();
    assert(approx(n.length(), 1.f));
    assert(approx_vec(n, Vec3(0.6f, 0.8f, 0.f)));
    PASS();

    TEST("Normalisation vecteur nul");
    Vec3 zero;
    Vec3 nz = zero.normalized();
    assert(approx_vec(nz, Vec3(0.f, 0.f, 0.f)));
    PASS();
}

void test_clamp() {
    TEST("Clamp [0,1]");
    Vec3 v(1.5f, -0.3f, 0.5f);
    Vec3 c = v.clamped();
    assert(approx_vec(c, Vec3(1.f, 0.f, 0.5f)));
    PASS();
}

// ---- Main ----
int main() {
    std::cout << "=== Tests unitaires : Vec3 ===\n\n";

    test_constructeurs();
    test_addition();
    test_soustraction();
    test_multiplication();
    test_produit_scalaire();
    test_produit_vectoriel();
    test_norme();
    test_clamp();

    std::cout << "\n=== Résultat : " << tests_passed << "/" << tests_total
              << " tests passés ===\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
