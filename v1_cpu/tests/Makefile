# =============================================================
#  Makefile — Tests unitaires du raytracer CPU
#  Exécution :
#    make          → compile tous les tests
#    make test     → compile + exécute tous les tests
#    make clean    → nettoyage
# =============================================================

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

TESTS := test_vec3 test_intersections test_phong

.PHONY: all test clean

all: $(TESTS)

test_vec3: test_vec3.cpp ../src/vec3.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_intersections: test_intersections.cpp ../src/vec3.h ../src/ray.h ../src/material.h ../src/sphere.h ../src/plane.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_phong: test_phong.cpp ../src/vec3.h ../src/ray.h ../src/material.h ../src/light.h ../src/sphere.h ../src/plane.h ../src/scene.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test: all
	@echo ""
	@echo "========================================"
	@echo "  Exécution de tous les tests unitaires"
	@echo "========================================"
	@echo ""
	@failed=0; \
	for t in $(TESTS); do \
		echo "--- $$t ---"; \
		./$$t; \
		if [ $$? -ne 0 ]; then failed=$$((failed+1)); fi; \
		echo ""; \
	done; \
	echo "========================================"; \
	if [ $$failed -eq 0 ]; then \
		echo "  TOUS LES TESTS ONT RÉUSSI"; \
	else \
		echo "  $$failed suite(s) de tests en ÉCHEC"; \
	fi; \
	echo "========================================"

clean:
	rm -f $(TESTS)
