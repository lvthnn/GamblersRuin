#!/usr/bin/env python3
from gamblersruin import GamblersRuin
import matplotlib.pyplot as plt
import seaborn as sns

a = GamblersRuin(1000, 50, 0.5001)
b = a.simulate_game(300, [0.48, 0.49, 0.5])

err = a.average_error(b)
averages = [sub["avg_error"][0] for sub in err]
sns.histplot(x=averages)
plt.xlabel("Average error")
plt.tight_layout()
plt.show()
