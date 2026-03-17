# =============================================================
#  Makefile — Raytracer CUDA v2
#  Utilisation :
#    make          → compile l'exécutable
#    make run      → compile + lance le rendu (1280×720)
#    make clean    → supprime les fichiers compilés
#
#  Sur Romeo, charger les modules avant make :
#    module load cuda/12.x  (ou la version disponible)
# =============================================================

NVCC      := nvcc
# -O3      : optimisation maximale
# -arch    : cible l'architecture du GPU Romeo
#            A100 → sm_80  |  V100 → sm_70  |  ajuster si besoin
# --std    : C++17
# H100 (Hopper) → sm_90  |  A100 → sm_80  |  V100 → sm_70
NVCCFLAGS := -std=c++17 -O3 -arch=sm_90 \
             -Xcompiler "-Wall -Wextra" \
             --expt-relaxed-constexpr

TARGET    := raytracer_cuda
SRC_DIR   := src
SRCS      := $(SRC_DIR)/main.cu

RESULTS_DIR := ../results

.PHONY: all run clean

all: $(RESULTS_DIR) $(TARGET)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(TARGET): $(SRCS) $(wildcard $(SRC_DIR)/*.cuh)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)
	@echo "[OK] Compilé : $(TARGET)"

run: all
	./$(TARGET) 1280 720

clean:
	rm -f $(TARGET)
	@echo "[OK] Nettoyé"
