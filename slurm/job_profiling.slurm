#!/bin/bash
# =============================================================
#  Job SLURM — Profiling CUDA (nsys + ncu)
#  Analyse les performances du kernel GPU
#
#  Soumission : cd ~/cuda-raytracer && sbatch slurm/job_profiling.slurm
#  Résultats  : results/profiling/
#
#  ROMEO — URCA (H100, ARM aarch64)
# =============================================================

#SBATCH --job-name=raytracer_prof
#SBATCH --output=slurm/logs/prof_%j.out
#SBATCH --error=slurm/logs/prof_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=r250127
#SBATCH --constraint=armgpu

#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "  PROFILING CUDA — Raytracer"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Nœud   : $SLURMD_NODENAME"
echo "  Date   : $(date)"
echo "=============================================="

# ---- Environnement ----
romeo_load_armgpu_env
spack load cuda@12.6.3

# Dossier de sortie
PROF_DIR="results/profiling"
mkdir -p $PROF_DIR

# ---- Compilation ----
echo "[INFO] Compilation CUDA..."
cd v2_cuda && make clean && make && cd ..

echo ""
echo "=== 1. Profiling avec nsys (Nsight Systems) ==="
echo "    Analyse globale : timeline, kernels, memcpy, API calls"
echo ""

# Nsight Systems — trace globale
if command -v nsys &> /dev/null; then
    nsys profile \
        --output=$PROF_DIR/nsys_report \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --stats=true \
        ./v2_cuda/raytracer_cuda 1920 1080 2>&1 | tee $PROF_DIR/nsys_summary.txt

    echo ""
    echo "[OK] Rapport nsys : $PROF_DIR/nsys_report.nsys-rep"
else
    echo "[WARN] nsys non disponible, profiling nsys ignoré"
fi

echo ""
echo "=== 2. Profiling avec ncu (Nsight Compute) ==="
echo "    Analyse détaillée du kernel : occupancy, bande passante, cache"
echo ""

# Nsight Compute — métriques kernel
if command -v ncu &> /dev/null; then
    ncu --set full \
        --target-processes all \
        --launch-skip 0 \
        --launch-count 1 \
        --export $PROF_DIR/ncu_report \
        --force-overwrite \
        ./v2_cuda/raytracer_cuda 1280 720 2>&1 | tee $PROF_DIR/ncu_summary.txt

    echo ""
    echo "[OK] Rapport ncu : $PROF_DIR/ncu_report.ncu-rep"
else
    echo "[WARN] ncu non disponible, profiling ncu ignoré"
fi

echo ""
echo "=== 3. Profiling mémoire et occupancy ==="
echo ""

# Métriques spécifiques si ncu est disponible
if command -v ncu &> /dev/null; then
    echo "--- Occupancy et utilisation des SM ---"
    ncu --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 0 \
        --launch-count 1 \
        ./v2_cuda/raytracer_cuda 1920 1080 2>&1 | tee $PROF_DIR/ncu_metrics.txt

    echo ""
fi

echo ""
echo "=== 4. Résolution multiple — temps kernel détaillé ==="
echo ""

# Temps kernel à différentes résolutions (pour analyse de scaling)
PROF_FILE="$PROF_DIR/scaling.txt"
echo "=== SCALING GPU — Temps kernel par résolution ===" > $PROF_FILE
echo "Date : $(date)" >> $PROF_FILE
echo "" >> $PROF_FILE
printf "%-12s | %10s | %10s | %14s | %12s\n" "Résolution" "Pixels" "Blocs" "Kernel (s)" "Mpixels/s" >> $PROF_FILE
echo "-------------|------------|------------|----------------|-------------" >> $PROF_FILE

for res in 320x180 640x360 960x540 1280x720 1920x1080 2560x1440 3840x2160; do
    W=$(echo $res | cut -d'x' -f1)
    H=$(echo $res | cut -d'x' -f2)
    PIXELS=$((W * H))
    BLOCS=$(( ((W+15)/16) * ((H+15)/16) ))

    OUTPUT=$(cd v2_cuda && ./raytracer_cuda $W $H 2>&1)
    KTIME=$(echo "$OUTPUT" | grep "Frame.*|" | head -1 | awk -F'|' '{print $2}' | awk '{print $1}' | tr -d '\n\r')
    MPPS=$(echo "$OUTPUT" | grep "Frame.*|" | head -1 | awk -F'|' '{print $3}' | awk '{print $1}' | tr -d '\n\r')

    printf "%-12s | %10d | %10d | %14s | %12s\n" "${W}x${H}" "$PIXELS" "$BLOCS" "$KTIME" "$MPPS" >> $PROF_FILE
    echo "  ${W}x${H} : ${KTIME} s  (${MPPS} Mpixels/s)"
done

echo ""
echo "[OK] Scaling GPU : $PROF_FILE"
cat $PROF_FILE

echo ""
echo "=== Profiling terminé ==="
echo "Résultats dans : $PROF_DIR/"
ls -la $PROF_DIR/
