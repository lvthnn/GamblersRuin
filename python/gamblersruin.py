#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from lifelines import KaplanMeierFitter

from collections import defaultdict
class GamblersRuin:
    def __init__(self, sim_duration, initial_capital, p):
        self._duration = sim_duration
        self._timevec = np.arange(1, sim_duration + 1)
        self._initial_capital = initial_capital
        self._kmf = KaplanMeierFitter()
        self._p = p
        self._expectation = self.__expected_model__()
        sns.set_theme()

    def __expected_model__(self):
        """ Calculates expected model to be stored as private field. """
        xi = 2 * self._p - 1
        e = np.repeat(self._initial_capital, self._duration) + xi * self._timevec
        return e

    def __simulate__(self, p=None):
        """ Runs a simulation of x Bernoulli trials where x is value of private
        field duration """
        if p is None:
            p = self._p
        elif not (0 < p < 1):
            raise ValueError("Gamble probability must be between 0 and 1.")

        capital = self._initial_capital
        rolls = np.random.rand(self._duration)
        logical = np.greater(p, rolls)

        outcomes = []
        for lo in logical:
            change = 1 if lo else -1
            capital += change
            outcomes.append(capital)

        data = {"time": self._timevec, "capital": outcomes, "p": p}
        return pd.DataFrame(data)

    def __group_by_p__(self, games):
        grouped = defaultdict(list)
        p_values = []
        for game in games:
            p_values.append(game["p"][0])
        p_values = np.unique(p_values)

        for key in p_values:
            grouped[key] = [game for game in games if key == game["p"][0]]
        return grouped

    def __convert_to_surv(self, games):
        games = self.__group_by_p__(games)
        times, events, ps = [], [], []

        for key in games:
            matches = [(np.where(game["capital"] == 0)) for game in games[key]]
            matches_cleaned = [match if np.size(match) > 0 else self._duration for match in matches]
            time = [np.min(match) for match in matches_cleaned]
            censor = np.less(time, self._duration)

            num_ps = len(matches)
            times.extend(time)
            events.extend(censor)
            ps.extend(np.repeat(key, num_ps))

        data = {"time": times, "event": events, "p": ps}
        return pd.DataFrame(data)

    def expectation(self):
        """
        Returns expected game trend of model.

        Uses vector calculations to predict average-case model trend
        where C(t) = C0 + C'(t)*t where C'(t) = 2p - 1.

        Returns:
        list: Array of expected values that can be combined with time vector
        to form a timeseries of the expected value.

        """
        return self._expectation

    def simulate_game(self, num_games=1, p_vec=None):
        """
        Simulates a number of games with the possibility of variable probability.
  
        Extended description of function.
  
        Parameters:
        num_games (int): Number of games to be simulated  
        p_vec (list): Array of float values (0;1] which give probability of success in gamble
        of stochastic game in question

        Returns:
        int: Description of return value

        """
        if p_vec is None:
            p_vec = []
        np_p_vec = np.array(p_vec)
        games = []
        if np_p_vec.size > 0:
            if np_p_vec.size != num_games:
                np_p_vec = np.repeat(np_p_vec, np.ceil(num_games / np_p_vec.size))[0:num_games]
            for i in range(num_games):
                sim = self.__simulate__(np_p_vec[i])
                games.append(sim)
        else:
            for _ in range(num_games):
                sim = self.__simulate__()
                games.append(sim)
        return games

    def visualize_games(self, games, show_theoretical=False):
        """
        Summary line.
  
        Extended description of function.
  
        Parameters:
        games (list): List of simulated games to be visualized
        show_theoretical (boolean): Whether to add the theoretical prediction to the plot.

        """
        num_games = len(games)
        plt.figure()

        for i in range(num_games):
            sns.lineplot(x=games[i]["time"], y=games[i]["capital"], alpha=0.1, color="C0")
        if show_theoretical:
            sns.lineplot(x=self._timevec, y=self._expectation, color="red", label="Predicted trend", linewidth=3)

        plt.xlabel("Time")
        plt.ylabel("Capital")
        plt.show()

    def survival_analysis(self, games):
        """
        Estimates survival function for a vector of simulated games. Values of p can vary.

        Extended description of function.

        Parameters:
        games (list): List of simulated games to be estimated using the Kaplan-Meier estimator.

        """
        ax = plt.subplot()
        data = self.__convert_to_surv(games)
        unique_ps = np.unique(data["p"])

        for p in unique_ps:
            subset = data[data["p"] == p]
            self._kmf.fit(durations=subset["time"], event_observed=subset["event"], label=f"{p}")
            self._kmf.plot_survival_function(ax=ax)
        plt.title("Duration analysis for different p values")
        plt.xlabel("Number of gambles")
        plt.ylabel("Proportion of games in play")
        plt.ylim(-0.05, 1.05)
        plt.show()

    def median_survival(self, games):
        data = self.__convert_to_surv(games)
        unique_ps = np.unique(data["p"])
        result = pd.DataFrame(columns=["p", "median"])

        for p in unique_ps:
            subset = data[data["p"] == p]
            self._kmf.fit(durations=subset["time"], event_observed=subset["event"], label=f"{p}")
            result = result.append({"p": p, "median": self._kmf.median_survival_time_}, ignore_index=True)
        return result

    def average_error(self, games):
        err = []
        for game in games:
            temp = self._expectation - game["capital"]
            avg_err = np.average(temp)
            result = {"avg_error": avg_err, "ind_error": temp}
            err.append(pd.DataFrame(result))
        return err
