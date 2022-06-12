#!/usr/bin/env python3
from pprint import pprint
import sympy

from gamblersruin import GamblersRuin, GamblingFunction
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

gambler = GamblersRuin(2000, 50, 0.5)
games = gambler.simulate_game(num_games=200, p_vec=0.48)
gambler.visualize_games(games)
gambler.survival_analysis(games)
