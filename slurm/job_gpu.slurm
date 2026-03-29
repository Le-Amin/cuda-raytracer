#!/bin/bash
# =============================================================
#  Job SLURM — Raytracer CUDA v2 (amélioré)
#  Soumission : cd ~/cuda-raytracer && sbatch slurm/job_gpu.slurm
#  Animation  : cd ~/cuda-raytracer && sbatch slurm/job_gpu.slurm anim
#  Suivi      : squeue -u $USER
#  Logs       : slurm/logs/gpu_<jobid>.out
#
#  ROMEO — URCA ( H100, ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_cuda
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err

# 1 nœud, 1 tâche, 1 GPU H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=r250127
#SBATCH --constraint=armgpu

# Mémoire CPU hôte et temps
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# ---- On se place à la racine du projet ----
cd $SLURM_SUBMIT_DIR

# ---- Environnement ----
echo "================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Nœud       : $SLURMD_NODENAME"
echo "  Répertoire : $(pwd)"
echo "  Date       : $(date)"
echo "  Arch       : $(uname -m)"
echo "================================================"

romeo_load_armgpu_env
spack load cuda@12.6.3

which nvcc && nvcc --version

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader

# ---- Compilation ----
echo "[INFO] Compilation du raytracer CUDA (sm_90 — H100)..."
cd v2_cuda
make clean
make

# ---- Exécution ----
# Si argument "anim" passé → mode animation (120 frames)
# Sinon → image unique
if [ "$1" = "anim" ]; then
    echo "[INFO] Mode animation : 120 frames à 1280x720..."
    time ./raytracer_cuda 1280 720 120
    echo "[INFO] Pour créer la vidéo :"
    echo "  ffmpeg -framerate 30 -i ../results/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p ../results/animation.mp4"
else
    echo "[INFO] Mode image unique : 1280x720..."
    time ./raytracer_cuda 1280 720
fi

echo "[INFO] Job terminé à $(date)"
