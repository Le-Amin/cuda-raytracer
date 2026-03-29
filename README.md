# cuda-raytracer

Implémentation d'un raytracer basique en deux versions :

- **v1_cpu** — version séquentielle en C++ pur
- **v2_cuda** — version parallélisée sur GPU avec CUDA (1 thread = 1 pixel)

Conçu pour être exécuté sur le supercalculateur **Romeo** via SLURM.

---

## Principe de fonctionnement

Pour chaque pixel de l'image :

1. Lancer un **rayon** depuis le centre du pixel vers la scène : `p = O + d·t`
2. Trouver l'**intersection la plus proche** avec les objets (t ≥ 0)
3. Si intersection → calculer la **couleur** via le modèle de Phong (ambiant + diffus + spéculaire + ombres)
4. Sinon → pixel **noir**

### Objets supportés

| Objet  | Équation               | Intersection                                  |
| ------ | ---------------------- | --------------------------------------------- |
| Sphère | `(p−c)·(p−c) − r² = 0` | `at² + bt + c = 0`, discriminant `Δ = b²−4ac` |
| Plan   | `(p−a)·n = 0`          | `t = (a−O)·n / d·n`                           |

### Éclairage (Phong)

```
I = ka·Ia  +  Σ_lumières [ kd·max(0, N·L)·Id  +  ks·max(0, R·V)^n·Is ]
```

avec `R = 2·(N·L)·N − L` et `V = −d` (direction observateur).
Un **rayon d'ombre** est lancé depuis chaque point d'intersection vers chaque lumière.

---

## Structure du projet

```
cuda-raytracer/
├── v1_cpu/
│   ├── Makefile
│   └── src/
│       ├── main.cpp      ← boucle pixel séquentielle
│       ├── vec3.h        ← vecteur 3D
│       ├── ray.h         ← rayon p = O + d·t
│       ├── material.h    ← matériau (Phong)
│       ├── light.h       ← source lumineuse
│       ├── sphere.h      ← sphère + intersection
│       ├── plane.h       ← plan  + intersection
│       ├── camera.h      ← fenêtre d'observation
│       └── scene.h       ← scène + rendu Phong
├── v2_cuda/
│   ├── Makefile
│   └── src/
│       ├── main.cu       ← kernel GPU + host
│       ├── vec3.cuh      ← vecteur 3D (__device__ __host__)
│       ├── ray.cuh
│       ├── material.cuh
│       ├── light.cuh
│       ├── sphere.cuh
│       ├── plane.cuh
│       └── camera.cuh
├── slurm/
│   ├── job_cpu.slurm     ← script SLURM version CPU
│   └── job_gpu.slurm     ← script SLURM version GPU
└── results/              ← images PPM générées
```

---

## Compilation et exécution locale

### v1 — CPU

```bash
cd v1_cpu
make          # compile → raytracer_cpu
make run      # compile + lance le rendu 1280×720
```

### v2 — CUDA

```bash
# Adapter -arch dans v2_cuda/Makefile selon votre GPU :
# A100 → sm_80  |  V100 → sm_70  |  H100 → sm_90

cd v2_cuda
make          # compile → raytracer_cuda
make run      # compile + lance le rendu 1280×720
```

L'image est sauvegardée dans `results/output_cpu.ppm` ou `results/output_cuda.ppm`.

---

## Exécution sur Romeo (SLURM)

### 1. Pousser le code sur Romeo

```bash
# Depuis votre PC (après git push)
ssh user@romeo.univ-reims.fr
cd /path/to/cuda-raytracer
git pull
```

### 2. Soumettre les jobs

```bash
# Vérifier la partition GPU disponible
sinfo

# Version CPU
sbatch slurm/job_cpu.slurm

# Version GPU (après avoir adapté --partition et -arch)
sbatch slurm/job_gpu.slurm

# Suivi
squeue -u $USER
```

### 3. Adapter les scripts SLURM

Dans `slurm/job_gpu.slurm`, modifier selon les ressources de Romeo :

```bash
#SBATCH --partition=<nom_partition_gpu>   # ex: gpu, gpuv100, gpua100
module load cuda/<version>                # ex: cuda/12.2
```

Dans `v2_cuda/Makefile`, adapter l'architecture :

```makefile
NVCCFLAGS := ... -arch=sm_80   # A100=sm_80 | V100=sm_70 | H100=sm_90
```

---

## Scène de rendu

| Objet           | Position     | Matériau             |
| --------------- | ------------ | -------------------- |
| Plan (sol)      | y = -1       | Blanc diffus         |
| Sphère rouge    | (-2, 0, -5)  | Diffuse, r=1         |
| Sphère verte    | (0, 0.5, -4) | Diffuse+spéc, r=1.5  |
| Sphère bleue    | (2, 0, -5)   | Très spéculaire, r=1 |
| Lumière blanche | (-3, 5, -2)  | Intensité 1.0        |
| Lumière chaude  | (4, 3, -1)   | Intensité 0.6        |

---

## Résultats attendus

| Version | Résolution | Temps indicatif |
| ------- | ---------- | --------------- |
| v1 CPU  | 1280×720   | ~2–5 s (1 cœur) |
| v2 CUDA | 1280×720   | < 0.1 s (GPU)   |
