#!/bin/bash
# =============================================================
#  Job SLURM — Tests unitaires (CPU)
#  Compile et exécute les tests unitaires du raytracer
#
#  Soumission : cd ~/cuda-raytracer && sbatch slurm/job_tests.slurm
#
#  ROMEO — URCA (ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_tests
#SBATCH --output=slurm/logs/tests_%j.out
#SBATCH --error=slurm/logs/tests_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=r250127
#SBATCH --constraint=armgpu

#SBATCH --mem=2G
#SBATCH --time=00:10:00

cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "  TESTS UNITAIRES — Raytracer CPU"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Nœud   : $SLURMD_NODENAME"
echo "  Date   : $(date)"
echo "=============================================="

# ---- Environnement ----
romeo_load_armgpu_env

# ---- Compilation et exécution des tests ----
echo ""
echo "[INFO] Compilation des tests..."
cd v1_cpu/tests && make clean && make

echo ""
make test

echo ""
echo "=== Tests terminés ==="
