#!/bin/bash
# =============================================================
#  Job SLURM — Raytracer CPU v1
#  Soumission : sbatch slurm/job_cpu.slurm
#  Suivi      : squeue -u $USER
#  Logs       : slurm/logs/cpu_<jobid>.out
#
#  ROMEO — URCA (Grace Hopper, ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_cpu
#SBATCH --output=slurm/logs/cpu_%j.out
#SBATCH --error=slurm/logs/cpu_%j.err

# Partition ARM GPU de ROMEO (les nœuds Grace Hopper ont aussi des CPU ARM)
#SBATCH --partition=armgpu

# Ressources : 1 nœud, 1 tâche, 1 cœur CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Mémoire et temps maximum
#SBATCH --mem=2G
#SBATCH --time=00:30:00

# ---- Environnement ----
echo "================================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Nœud     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "  Arch     : $(uname -m)"
echo "================================================"

# Charger l'environnement ARM GPU de ROMEO
romeo_load_armgpu_env

cd $SLURM_SUBMIT_DIR

# Créer les dossiers nécessaires
mkdir -p slurm/logs
mkdir -p results

# ---- Compilation ----
echo "[INFO] Compilation du raytracer CPU..."
cd v1_cpu
make clean
make
cd ..

# ---- Exécution ----
echo "[INFO] Lancement du rendu 1280x720..."
cd v1_cpu
time ./raytracer_cpu 1280 720
cd ..

echo "[INFO] Job terminé à $(date)"
echo "Image disponible : results/output_cpu.ppm"
