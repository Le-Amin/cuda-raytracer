#!/bin/bash
# =============================================================
#  Job SLURM — Benchmark CPU vs GPU
#  Compare les temps de rendu à plusieurs résolutions
#
#  Soumission : cd ~/cuda-raytracer && sbatch slurm/job_benchmark.slurm
#  Résultats  : results/benchmark.txt
#
#  ROMEO — URCA (H100, ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_bench
#SBATCH --output=slurm/logs/bench_%j.out
#SBATCH --error=slurm/logs/bench_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=r250127
#SBATCH --constraint=armgpu

#SBATCH --mem=8G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "  BENCHMARK CPU vs GPU — Raytracer"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Nœud   : $SLURMD_NODENAME"
echo "  Date   : $(date)"
echo "  Arch   : $(uname -m)"
echo "=============================================="

# ---- Environnement ----
romeo_load_armgpu_env
spack load cuda@12.6.3

# Infos matériel
echo ""
echo "=== CPU ==="
lscpu | grep "Model name" || cat /proc/cpuinfo | grep "model name" | head -1
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ---- Compilation des deux versions ----
echo "[INFO] Compilation CPU..."
cd v1_cpu && make clean && make && cd ..

echo "[INFO] Compilation CUDA..."
cd v2_cuda && make clean && make && cd ..

# ---- Résolutions à tester ----
RESOLUTIONS="640x360 1280x720 1920x1080 3840x2160"

# Fichier de résultats
BENCH_FILE="results/benchmark.txt"
echo "=== BENCHMARK RAYTRACER — CPU vs GPU ===" > $BENCH_FILE
echo "Date : $(date)" >> $BENCH_FILE
echo "Nœud : $SLURMD_NODENAME" >> $BENCH_FILE
echo "" >> $BENCH_FILE
printf "%-12s | %12s | %12s | %10s\n" "Résolution" "CPU (s)" "GPU (s)" "Speedup" >> $BENCH_FILE
echo "-------------|--------------|--------------|------------" >> $BENCH_FILE

echo ""
echo "=== Début du benchmark ==="
echo ""

for res in $RESOLUTIONS; do
    W=$(echo $res | cut -d'x' -f1)
    H=$(echo $res | cut -d'x' -f2)
    PIXELS=$((W * H))

    echo "--- Résolution : ${W} x ${H} (${PIXELS} pixels) ---"

    # ---- CPU ----
    echo "  [CPU] Rendu en cours..."
    CPU_OUTPUT=$(cd v1_cpu && ./raytracer_cpu $W $H 2>&1)
    CPU_TIME=$(echo "$CPU_OUTPUT" | grep "Temps de rendu" | awk '{print $(NF-1)}')
    echo "  [CPU] Temps : ${CPU_TIME} s"

    # ---- GPU ----
    echo "  [GPU] Rendu en cours..."
    GPU_OUTPUT=$(cd v2_cuda && ./raytracer_cuda $W $H 2>&1)
    GPU_TIME=$(echo "$GPU_OUTPUT" | grep "Temps total" | awk '{print $(NF-1)}')
    echo "  [GPU] Temps : ${GPU_TIME} s"

    # ---- Speedup ----
    if [ -n "$CPU_TIME" ] && [ -n "$GPU_TIME" ]; then
        SPEEDUP=$(awk "BEGIN {printf \"%.1f\", $CPU_TIME / $GPU_TIME}")
    else
        SPEEDUP="N/A"
    fi
    echo "  [SPEEDUP] ${SPEEDUP}x"

    # Écrire dans le fichier
    printf "%-12s | %12s | %12s | %10sx\n" "${W}x${H}" "$CPU_TIME" "$GPU_TIME" "$SPEEDUP" >> $BENCH_FILE

    echo ""
done

echo "" >> $BENCH_FILE
echo "=== Configuration ===" >> $BENCH_FILE
echo "AA samples     : 4 (grille 2x2 stratifiée)" >> $BENCH_FILE
echo "Max réflexions : 5" >> $BENCH_FILE
echo "Scène          : 3 sphères + 1 plan damier + 2 lumières" >> $BENCH_FILE

echo "=== Benchmark terminé ==="
echo "Résultats dans : $BENCH_FILE"
cat $BENCH_FILE
