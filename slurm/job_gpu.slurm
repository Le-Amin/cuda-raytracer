#!/bin/bash
# =============================================================
#  Job SLURM — Raytracer CUDA v2
#  Soumission : sbatch slurm/job_gpu.slurm
#  Suivi      : squeue -u $USER
#  Logs       : slurm/logs/gpu_<jobid>.out
#
#  ROMEO — URCA (Grace Hopper H100, ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_cuda
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err

# Partition ARM GPU de ROMEO (H100)
#SBATCH --partition=armgpu

# 1 nœud, 1 tâche, 1 GPU H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Mémoire CPU hôte et temps
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# ---- Environnement ----
echo "================================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Nœud     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "  Arch     : $(uname -m)"
echo "================================================"

# Charger l'environnement ARM GPU de ROMEO (CUDA + compilateurs)
romeo_load_armgpu_env

# Vérifier que nvcc est accessible
which nvcc && nvcc --version

# Infos sur le GPU alloué
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader

cd $SLURM_SUBMIT_DIR

mkdir -p slurm/logs
mkdir -p results

# ---- Compilation ----
echo "[INFO] Compilation du raytracer CUDA (sm_90 — H100)..."
cd v2_cuda
make clean
make
cd ..

# ---- Exécution ----
echo "[INFO] Lancement du rendu 1280x720 sur GPU H100..."
cd v2_cuda
time ./raytracer_cuda 1280 720
cd ..

echo "[INFO] Job terminé à $(date)"
echo "Image disponible : results/output_cuda.ppm"
