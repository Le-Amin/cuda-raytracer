// =============================================================
//  Raytracer CUDA — v2 (amélioré)
//
//  Fonctionnalités :
//    - Parallélisation GPU : 1 thread = 1 pixel
//    - Éclairage Phong (ambiant + diffus + spéculaire + ombres)
//    - Réflexions itératives (jusqu'à MAX_DEPTH rebonds)
//    - Anti-aliasing par supersampling (AA_SAMPLES rayons/pixel)
//    - Motif damier sur le sol
//    - Mode animation : rotation de la caméra (N frames → vidéo)
//
//  Mémoire :
//    - Scène en mémoire __constant__ (lecture seule, cachée)
//    - Buffer de pixels : cudaMalloc → copie CPU → écriture PPM
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

// ---- Paramètres de rendu ----
#define MAX_SPHERES   8
#define MAX_PLANES    4
#define MAX_LIGHTS    4
#define MAX_DEPTH     5       // Profondeur max des réflexions
#define AA_SAMPLES    4       // Nombre d'échantillons par pixel (anti-aliasing)

// ---- Mémoire constante GPU ----
__constant__ Sphere   d_spheres[MAX_SPHERES];
__constant__ Plane    d_planes [MAX_PLANES];
__constant__ Light    d_lights [MAX_LIGHTS];
__constant__ int      d_num_spheres;
__constant__ int      d_num_planes;
__constant__ int      d_num_lights;
__constant__ Camera   d_camera;

// =================================================================
//  HitInfo : résultat d'une intersection rayon-objet
//  object_type : 0 = sphère, 1 = plan (pour le motif damier)
// =================================================================
struct HitInfo {
    float    t;
    Vec3     point;
    Vec3     normal;
    Material material;
    bool     hit;
    int      object_type;   // 0 = sphère, 1 = plan
};

// Crée un HitInfo vide (pas d'intersection)
__device__ inline HitInfo make_empty_hit() {
    HitInfo h;
    h.t           = 1e30f;
    h.hit         = false;
    h.object_type = -1;
    return h;
}

// ----------------------------------------------------------------
//  Intersection la plus proche parmi tous les objets
// ----------------------------------------------------------------
__device__
HitInfo find_hit(const Ray& ray)
{
    HitInfo rec = make_empty_hit();

    for (int i = 0; i < d_num_spheres; ++i) {
        float t = d_spheres[i].intersect(ray);
        if (t > 0.f && t < rec.t) {
            rec.t           = t;
            rec.point       = ray.at(t);
            rec.normal      = d_spheres[i].normal_at(rec.point);
            rec.material    = d_spheres[i].material;
            rec.hit         = true;
            rec.object_type = 0;  // sphère
        }
    }

    for (int i = 0; i < d_num_planes; ++i) {
        float t = d_planes[i].intersect(ray);
        if (t > 0.f && t < rec.t) {
            rec.t           = t;
            rec.point       = ray.at(t);
            rec.normal      = d_planes[i].normal_at(rec.point);
            rec.material    = d_planes[i].material;
            rec.hit         = true;
            rec.object_type = 1;  // plan
        }
    }

    return rec;
}

// ----------------------------------------------------------------
//  Test d'ombre : vérifie si un point est masqué d'une lumière
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
//  Motif damier sur le sol (coordonnées XZ)
//  Alterne entre la couleur du matériau et une version sombre
// ----------------------------------------------------------------
__device__
Vec3 checkerboard_color(const Vec3& point, const Vec3& base_color)
{
    float scale = 2.0f;  // Taille des carreaux
    int cx = (int)floorf(point.x / scale);
    int cz = (int)floorf(point.z / scale);
    if ((cx + cz) & 1)
        return base_color;              // Carreau clair
    else
        return base_color * 0.15f;      // Carreau sombre
}

// ----------------------------------------------------------------
//  Calcul Phong local pour un point d'intersection
//  Retourne la couleur locale (sans réflexion)
// ----------------------------------------------------------------
__device__
Vec3 compute_phong(const Ray& ray, const HitInfo& rec)
{
    Material mat = rec.material;
    Vec3 N = rec.normal;
    Vec3 V = (-ray.direction).normalized();

    // Appliquer le damier si l'objet touché est un plan
    Vec3 base_color = mat.color;
    if (rec.object_type == 1)
        base_color = checkerboard_color(rec.point, mat.color);

    // Terme ambiant
    Vec3 color = base_color * mat.ambient;

    for (int li = 0; li < d_num_lights; ++li) {
        const Light& light = d_lights[li];
        if (in_shadow(rec.point, light)) continue;

        Vec3  L  = (light.position - rec.point).normalized();
        float NL = N.dot(L);
        if (NL <= 0.f) continue;

        // Terme diffus : kd * (N·L) * couleur
        Vec3 diffuse = base_color * (mat.diffuse * NL);

        // Terme spéculaire : ks * (R·V)^n
        Vec3  R  = (N * (2.f * NL) - L).normalized();
        float RV = fmaxf(0.f, R.dot(V));
        float s  = mat.specular;
        Vec3 spec = Vec3{s, s, s} * powf(RV, mat.shininess);

        color += (diffuse + spec) * light.color * light.intensity;
    }

    return color;
}

// ----------------------------------------------------------------
//  Shading avec réflexions itératives
//
//  Pour chaque rebond :
//    couleur_finale += atténuation * couleur_locale * (1 - reflectivity)
//    atténuation   *= reflectivity
//    rayon          = rayon réfléchi R = D - 2(D·N)N
//
//  Boucle itérative (pas de récursion → adapté au GPU)
// ----------------------------------------------------------------
__device__
Vec3 shade(const Ray& initial_ray)
{
    Vec3 accumulated = {0.f, 0.f, 0.f};
    Vec3 attenuation = {1.f, 1.f, 1.f};
    Ray  current_ray = initial_ray;

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitInfo rec = find_hit(current_ray);
        if (!rec.hit) break;  // Fond noir

        // Couleur Phong locale
        Vec3 local_color = compute_phong(current_ray, rec);

        // Contribution pondérée par l'atténuation
        float refl = rec.material.reflectivity;
        accumulated += attenuation * local_color * (1.f - refl);

        // Si le matériau n'est pas réfléchissant, on arrête
        if (refl < 1e-4f) break;

        // Calculer le rayon réfléchi : R = D - 2(D·N)N
        Vec3 D = current_ray.direction;
        Vec3 N = rec.normal;
        Vec3 reflect_dir = D - N * (2.f * D.dot(N));

        // Décalage epsilon le long de la normale pour éviter l'auto-intersection
        current_ray = Ray(rec.point + N * 1e-3f, reflect_dir);

        // Atténuer par le coefficient de réflexion
        attenuation *= refl;
    }

    return accumulated.clamped();
}

// ----------------------------------------------------------------
//  Fonction de hachage pour le jitter déterministe (anti-aliasing)
//  Retourne un float dans [0, 1)
// ----------------------------------------------------------------
__device__ inline float hash_float(int x, int y, int s)
{
    unsigned int h = (unsigned int)(x * 374761393 + y * 668265263 + s * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)(h & 0x00FFFFFFu) / (float)0x01000000u;
}

// ----------------------------------------------------------------
//  Kernel principal : 1 thread = 1 pixel
//  Anti-aliasing : AA_SAMPLES rayons par pixel (grille 2×2 stratifiée)
// ----------------------------------------------------------------
__global__
void render_kernel(Vec3* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    Vec3 color = {0.f, 0.f, 0.f};

    // Grille 2×2 stratifiée avec jitter aléatoire
    for (int s = 0; s < AA_SAMPLES; ++s) {
        float jx = hash_float(x, y, s * 2);
        float jy = hash_float(x, y, s * 2 + 1);

        // Position du sous-pixel dans la grille 2×2
        float sx = (float)(s % 2) * 0.5f + jx * 0.5f;
        float sy = (float)(s / 2) * 0.5f + jy * 0.5f;

        float u = (x + sx) / (float)width;
        float v = (y + sy) / (float)height;

        Ray ray = d_camera.get_ray(u, v);
        color += shade(ray);
    }

    pixels[y * width + x] = (color * (1.f / AA_SAMPLES)).clamped();
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
//  Écriture PPM (format P3 — ASCII)
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
//  Construction de la scène
//  3 sphères + sol en damier + 2 lumières
//  Matériaux avec coefficient de réflexion
// ----------------------------------------------------------------
static void build_scene(Sphere* spheres, int& ns,
                        Plane*  planes,  int& np,
                        Light*  lights,  int& nl)
{
    // {couleur, ka, kd, ks, shininess, reflectivity}
    Material mat_floor = {{0.8f, 0.8f, 0.8f}, 0.1f, 0.9f, 0.1f,   8.f, 0.2f};   // Sol : légèrement réfléchissant
    Material mat_red   = {{0.9f, 0.1f, 0.1f}, 0.1f, 0.8f, 0.2f,  16.f, 0.1f};   // Rouge : presque opaque
    Material mat_green = {{0.1f, 0.8f, 0.2f}, 0.1f, 0.7f, 0.5f,  64.f, 0.3f};   // Vert : réflexion moyenne
    Material mat_blue  = {{0.1f, 0.3f, 0.9f}, 0.1f, 0.5f, 0.9f, 128.f, 0.8f};   // Bleu : quasi-miroir

    // Sol (plan horizontal y = -1, normale vers le haut)
    planes [0] = {{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, mat_floor};
    np = 1;

    // Sphère rouge (gauche), verte (centre), bleue (droite)
    spheres[0] = {{-2.f,  0.f, -5.f}, 1.f,  mat_red};
    spheres[1] = {{ 0.f,  0.5f,-4.f}, 1.5f, mat_green};
    spheres[2] = {{ 2.f,  0.f, -5.f}, 1.f,  mat_blue};
    ns = 3;

    // Lumière blanche principale (au-dessus à gauche)
    lights [0] = {{-3.f, 5.f, -2.f}, {1.f, 1.f, 1.f},     1.0f};
    // Lumière secondaire chaude (à droite, moins intense)
    lights [1] = {{ 4.f, 3.f, -1.f}, {1.f, 0.9f, 0.7f},   0.6f};
    nl = 2;
}

// ----------------------------------------------------------------
//  Main
//
//  Usage :
//    ./raytracer_cuda [largeur hauteur]           → 1 image
//    ./raytracer_cuda [largeur hauteur] [frames]  → animation
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    int W = 1280, H = 720;
    int num_frames = 1;  // Par défaut : une seule image

    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }
    if (argc >= 4) { num_frames = atoi(argv[3]); }

    printf("=== Raytracer CUDA v2 (amélioré) ===\n");
    printf("Résolution : %d x %d\n", W, H);
    printf("Frames     : %d\n", num_frames);
    printf("AA samples : %d  |  Max réflexions : %d\n", AA_SAMPLES, MAX_DEPTH);

    // --- Affiche le GPU utilisé ---
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU : %s  |  SM : %d  |  VRAM : %.0f MB\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    // --- Construction de la scène (ne change pas entre les frames) ---
    Sphere h_spheres[MAX_SPHERES];
    Plane  h_planes [MAX_PLANES];
    Light  h_lights [MAX_LIGHTS];
    int    ns = 0, np = 0, nl = 0;
    build_scene(h_spheres, ns, h_planes, np, h_lights, nl);

    // Copie scène vers mémoire constante GPU (une seule fois)
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres,     h_spheres, ns * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_planes,      h_planes,  np * sizeof(Plane)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_lights,      h_lights,  nl * sizeof(Light)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &ns, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_planes,  &np, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_lights,  &nl, sizeof(int)));

    // --- Buffer GPU (réutilisé pour chaque frame) ---
    Vec3* d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, W * H * sizeof(Vec3)));

    // Buffer CPU pour la copie
    std::vector<Vec3> h_pixels(W * H);

    // Grille de threads
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    printf("Grille : %d x %d blocs de %d x %d threads\n",
           grid.x, grid.y, block.x, block.y);

    // --- Paramètres de l'orbite caméra (pour l'animation) ---
    Vec3  cam_target = {0.f, 0.f, -4.f};     // Point visé (centre de la scène)
    Vec3  cam_up     = {0.f, 1.f,  0.f};      // Vecteur haut
    float cam_fov    = 60.f;
    float orbit_radius = 7.f;                  // Rayon de l'orbite
    float orbit_height = 2.0f;                 // Hauteur de la caméra

    // Chrono global pour l'ensemble des frames
    auto t_total_start = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < num_frames; ++frame) {

        // --- Position de la caméra ---
        Vec3 cam_eye;
        if (num_frames == 1) {
            // Mode image unique : position fixe
            cam_eye = {0.f, 1.f, 2.f};
        } else {
            // Mode animation : orbite autour de la scène
            float angle = 2.f * 3.14159265f * (float)frame / (float)num_frames;
            cam_eye.x = cam_target.x + orbit_radius * sinf(angle);
            cam_eye.y = orbit_height;
            cam_eye.z = cam_target.z + orbit_radius * cosf(angle);
        }

        // Construire et copier la caméra pour cette frame
        Camera h_cam = Camera::create(cam_eye, cam_target, cam_up, cam_fov, W, H);
        CUDA_CHECK(cudaMemcpyToSymbol(d_camera, &h_cam, sizeof(Camera)));

        // --- Lancement du kernel ---
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();

        render_kernel<<<grid, block>>>(d_pixels, W, H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1   = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        // --- Copie résultat vers CPU ---
        CUDA_CHECK(cudaMemcpy(h_pixels.data(), d_pixels,
                              W * H * sizeof(Vec3), cudaMemcpyDeviceToHost));

        // --- Écriture PPM ---
        char filename[256];
        if (num_frames == 1) {
            snprintf(filename, sizeof(filename), "../results/output_cuda.ppm");
        } else {
            snprintf(filename, sizeof(filename), "../results/frame_%04d.ppm", frame);
        }
        write_ppm(filename, h_pixels.data(), W, H);

        // Progression
        printf("  Frame %3d/%d  |  %.4f s  |  %.2f Mpixels/s\n",
               frame + 1, num_frames, dt,
               (double)(W * H) / dt / 1e6);
    }

    // --- Statistiques globales ---
    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();

    printf("\n=== Terminé ===\n");
    printf("Temps total       : %.2f s\n", total_time);
    printf("Temps moyen/frame : %.4f s\n", total_time / num_frames);

    if (num_frames > 1) {
        printf("\nPour créer la vidéo :\n");
        printf("  ffmpeg -framerate 30 -i ../results/frame_%%04d.ppm "
               "-c:v libx264 -pix_fmt yuv420p ../results/animation.mp4\n");
    }

    CUDA_CHECK(cudaFree(d_pixels));
    return 0;
}
