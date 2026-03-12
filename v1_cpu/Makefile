# =============================================================
#  Makefile — Raytracer CPU v1
#  Utilisation :
#    make          → compile l'exécutable
#    make run      → compile + lance le rendu (1280×720)
#    make clean    → supprime les fichiers compilés
# =============================================================

CXX      := g++
# -march=native s'adapte automatiquement à l'architecture (x86 ou ARM aarch64)
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
TARGET   := raytracer_cpu
SRC_DIR  := src
SRCS     := $(SRC_DIR)/main.cpp

# Dossier de sortie pour les images
RESULTS_DIR := ../results

.PHONY: all run clean

all: $(RESULTS_DIR) $(TARGET)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(TARGET): $(SRCS) $(wildcard $(SRC_DIR)/*.h)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)
	@echo "[OK] Compilé : $(TARGET)"

run: all
	./$(TARGET) 1280 720

clean:
	rm -f $(TARGET)
	@echo "[OK] Nettoyé"
