#!/usr/bin/env python3
from gamblersruin import GamblersRuin

a = GamblersRuin(1000, 50, 0.5001)
b = a.simulate_game(300, [0.48, 0.49, 0.5])
a.duration_analysis(b)
