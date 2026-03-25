// =============================================================
//  Raytracer CUDA — v2
//
//  Parallélisation : chaque thread GPU traite UN pixel.
//    - grille 2D de blocs 2D : block(16×16), grid(ceil(W/16) × ceil(H/16))
//    - thread (tx, ty) → pixel (x, y) = (blockIdx.x*16+tx, blockIdx.y*16+ty)
//
//  Le kernel `render_kernel` est __global__ ; toute la logique de
//  ray-casting / Phong est __device__ (inline dans les headers .cuh).
//
//  Mémoire :
//    - Scène (sphères, plans, lumières) copiée dans la mémoire
//      constante GPU (__constant__) — lecture seule, très rapide.
//    - Buffer de pixels : cudaMalloc → copie vers CPU → écriture PPM.
// =============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "light.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "camera.cuh"

// ---- Dimensions de la scène (tailles fixes pour la mémoire constante) ----
#define MAX_SPHERES  8
#define MAX_PLANES   4
#define MAX_LIGHTS   4

// ---- Mémoire constante GPU (accès en lecture seule, cachée) ----
__constant__ Sphere   d_spheres[MAX_SPHERES];
__constant__ Plane    d_planes [MAX_PLANES];
__constant__ Light    d_lights [MAX_LIGHTS];
__constant__ int      d_num_spheres;
__constant__ int      d_num_planes;
__constant__ int      d_num_lights;
__constant__ Camera   d_camera;

// ----------------------------------------------------------------
//  Intersection la plus proche (même logique que scene.h CPU)
// ----------------------------------------------------------------
struct HitInfo {
    float    t;
    Vec3     point;
    Vec3     normal;
    Material material;
    bool     hit;
};

// Crée un HitInfo "vide" (pas d'intersection)
__device__ inline HitInfo make_empty_hit() {
    HitInfo h;
    h.t   = 1e30f;
    h.hit = false;
    return h;
}

__device__
HitInfo find_hit(const Ray& ray)
{
    HitInfo rec = make_empty_hit();

    for (int i = 0; i < d_num_spheres; ++i) {
        float t = d_spheres[i].intersect(ray);
        if (t > 0.f && t < rec.t) {
            rec.t        = t;
            rec.point    = ray.at(t);
            rec.normal   = d_spheres[i].normal_at(rec.point);
            rec.material = d_spheres[i].material;
            rec.hit      = true;
        }
    }

    for (int i = 0; i < d_num_planes; ++i) {
        float t = d_planes[i].intersect(ray);
        if (t > 0.f && t < rec.t) {
            rec.t        = t;
            rec.point    = ray.at(t);
            rec.normal   = d_planes[i].normal_at(rec.point);
            rec.material = d_planes[i].material;
            rec.hit      = true;
        }
    }

    return rec;
}

// ----------------------------------------------------------------
//  Test d'ombre
// ----------------------------------------------------------------
__device__
bool in_shadow(const Vec3& origin, const Light& light)
{
    Vec3  to_light = light.position - origin;
    float dist     = to_light.length();
    Ray   sray(origin, to_light);

    for (int i = 0; i < d_num_spheres; ++i) {
        float t = d_spheres[i].intersect(sray);
        if (t > 1e-4f && t < dist) return true;
    }
    for (int i = 0; i < d_num_planes; ++i) {
        float t = d_planes[i].intersect(sray);
        if (t > 1e-4f && t < dist) return true;
    }
    return false;
}

// ----------------------------------------------------------------
//  Calcul Phong complet pour un pixel (même formule que v1 CPU)
// ----------------------------------------------------------------
__device__
Vec3 shade(const Ray& ray)
{
    HitInfo rec = find_hit(ray);
    if (!rec.hit) return Vec3{0.f, 0.f, 0.f};  // fond noir

    const Material& mat = rec.material;
    Vec3 N = rec.normal;
    Vec3 V = (-ray.direction).normalized();

    // Terme ambiant
    Vec3 color = mat.color * mat.ambient;

    for (int li = 0; li < d_num_lights; ++li) {
        const Light& light = d_lights[li];
        if (in_shadow(rec.point, light)) continue;

        Vec3  L  = (light.position - rec.point).normalized();
        float NL = N.dot(L);
        if (NL <= 0.f) continue;

        // Diffus
        Vec3 diffuse = mat.color * (mat.diffuse * NL);

        // Spéculaire
        Vec3  R  = (N * (2.f * NL) - L).normalized();
        float RV = fmaxf(0.f, R.dot(V));
        float s = mat.specular;
        Vec3 specular = Vec3{s, s, s} * powf(RV, mat.shininess);

        color += (diffuse + specular) * light.color * light.intensity;
    }

    return color.clamped();
}

// ----------------------------------------------------------------
//  Kernel principal : 1 thread = 1 pixel
//  pixels[y * width + x] = couleur du pixel (x,y)
// ----------------------------------------------------------------
__global__
void render_kernel(Vec3* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // hors image

    // Coordonnées normalisées u,v ∈ [0,1]
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    // Lancer le rayon depuis le centre du pixel
    Ray ray = d_camera.get_ray(u, v);

    // Calcul de couleur + écriture dans le buffer
    pixels[y * width + x] = shade(ray);
}

// ----------------------------------------------------------------
//  Macro de vérification des erreurs CUDA
// ----------------------------------------------------------------
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "[CUDA ERROR] %s : %s (%s:%d)\n",       \
                    #call, cudaGetErrorString(err),                  \
                    __FILE__, __LINE__);                             \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while(0)

// ----------------------------------------------------------------
//  Écriture PPM
// ----------------------------------------------------------------
static void write_ppm(const char* filename,
                      const Vec3* pixels, int w, int h)
{
    std::ofstream out(filename);
    if (!out) {
        fprintf(stderr, "[ERROR] Impossible d'ouvrir : %s\n", filename);
        return;
    }
    out << "P3\n" << w << " " << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            const Vec3& c = pixels[j * w + i];
            out << (int)(255.99f * c.x) << " "
                << (int)(255.99f * c.y) << " "
                << (int)(255.99f * c.z) << "\n";
        }
    }
    printf("[OK] Image sauvegardée : %s\n", filename);
}

// ----------------------------------------------------------------
//  Construction de la scène (identique à v1 CPU)
// ----------------------------------------------------------------
static void build_scene(Sphere* spheres, int& ns,
                        Plane*  planes,  int& np,
                        Light*  lights,  int& nl)
{
    Material mat_floor = {{0.8f, 0.8f, 0.8f}, 0.1f, 0.9f, 0.1f,  8.f};
    Material mat_red   = {{0.9f, 0.1f, 0.1f}, 0.1f, 0.8f, 0.2f, 16.f};
    Material mat_green = {{0.1f, 0.8f, 0.2f}, 0.1f, 0.7f, 0.5f, 64.f};
    Material mat_blue  = {{0.1f, 0.3f, 0.9f}, 0.1f, 0.5f, 0.9f,128.f};

    // Normale déjà unitaire {0,1,0} → pas besoin de normaliser
    planes [0] = {{0.f,-1.f, 0.f}, {0.f, 1.f, 0.f}, mat_floor};
    np = 1;

    spheres[0] = {{-2.f, 0.f,-5.f}, 1.f,  mat_red};
    spheres[1] = {{ 0.f, 0.5f,-4.f},1.5f, mat_green};
    spheres[2] = {{ 2.f, 0.f,-5.f}, 1.f,  mat_blue};
    ns = 3;

    lights [0] = {{-3.f, 5.f,-2.f}, {1.f, 1.f, 1.f},     1.0f};
    lights [1] = {{ 4.f, 3.f,-1.f}, {1.f, 0.9f, 0.7f},   0.6f};
    nl = 2;
}

// ----------------------------------------------------------------
//  Main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    int W = 1280, H = 720;
    if (argc == 3) { W = atoi(argv[1]); H = atoi(argv[2]); }

    printf("=== Raytracer CUDA v2 ===\n");
    printf("Résolution : %d x %d\n", W, H);

    // --- Affiche le GPU utilisé ---
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU : %s  |  SM : %d  |  VRAM : %.0f MB\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    // --- Scène ---
    Sphere h_spheres[MAX_SPHERES];
    Plane  h_planes [MAX_PLANES];
    Light  h_lights [MAX_LIGHTS];
    int    ns = 0, np = 0, nl = 0;
    build_scene(h_spheres, ns, h_planes, np, h_lights, nl);

    // --- Caméra ---
    Camera h_cam = Camera::create(
        Vec3{0.f, 1.f, 2.f}, Vec3{0.f, 0.f, -4.f},
        Vec3{0.f, 1.f, 0.f}, 60.f, W, H);

    // --- Copie vers mémoire constante GPU ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres,     h_spheres, ns * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_planes,      h_planes,  np * sizeof(Plane)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_lights,      h_lights,  nl * sizeof(Light)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &ns, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_planes,  &np, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_lights,  &nl, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_camera,      &h_cam,    sizeof(Camera)));

    // --- Buffer GPU ---
    Vec3* d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, W * H * sizeof(Vec3)));

    // --- Lancement du kernel ---
    //   Bloc 2D de 16×16 threads → 256 threads/bloc
    dim3 block(16, 16);
    dim3 grid( (W + block.x - 1) / block.x,
               (H + block.y - 1) / block.y );

    printf("Grille : %d × %d blocs de %d × %d threads\n",
           grid.x, grid.y, block.x, block.y);

    // Synchronisation + chrono
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    render_kernel<<<grid, block>>>(d_pixels, W, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("Temps de rendu : %.4f s\n", dt);
    printf("Performance    : %.2f Mpixels/s\n",
           (double)(W * H) / dt / 1e6);

    // --- Copie résultat vers CPU ---
    std::vector<Vec3> h_pixels(W * H);
    CUDA_CHECK(cudaMemcpy(h_pixels.data(), d_pixels,
                          W * H * sizeof(Vec3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));

    // --- Écriture PPM ---
    write_ppm("../results/output_cuda.ppm", h_pixels.data(), W, H);

    return 0;
}
