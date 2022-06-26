#!/usr/bin/env python3
from collections import defaultdict
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class GamblingFunction:
    def __init__(self, bin_vec=None, payoff_vec=None):
        self._bin_vec = np.sort(bin_vec)
        self._payoff_vec = payoff_vec
        self._mapping = self.__construct_mapping__()

    def __construct_mapping__(self):
        return {self._bin_vec[i]: self._payoff_vec[i] for i in range(len(self._bin_vec))}

    @property
    def mapping(self):
        return self._mapping

    def get_value(self, r=None):
        elmin, elmax = np.min(self._bin_vec, 0), np.max(self._bin_vec)
        if r is None:
            r = np.random.uniform(elmin, elmax)
        d = np.digitize(r, self._bin_vec, right=True)
        k = self._bin_vec[d - 1]
        if callable(self._mapping[k]):
            return self._mapping[k](r)
        return self._mapping[k]


class GamblersRuin:
    def __init__(self, sim_duration, initial_capital, p, gambling_function=None):
        self._duration = sim_duration
        self._timevec = np.arange(1, sim_duration + 1)
        self._initial_capital = initial_capital
        self._kmf = KaplanMeierFitter()
        self._p = p
       
        if gambling_function is not None:
           self._gambling_function = gambling_function 

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

    @property
    def expectation(self):
        """
        Returns expected game trend of model.

        Uses vector calculations to predict average-case model trend
        where C(t) = C0 + C'(t)*t where C'(t) = 2p - 1.

        Returns:
        list: Array of expected values that can be combined with time vector
        to form a timeseries of the expected value.

        """
        return self.expectation

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
        Shows plot of simulated games.
  
        Displays simulated games as line plots with the option of showing the theoretical expected
        trend over time.
  
        Parameters:
        games (list): List of simulated games to be visualized
        show_theoretical (boolean): Whether to add the theoretical prediction to the plot.

        """
        num_games = len(games)
        plt.figure()

        for i in range(num_games):
            sns.lineplot(x=games[i]["time"], y=games[i]["capital"], alpha=0.1, color="C0")
        if show_theoretical:
            sns.lineplot(x=self._timevec, y=self.expectation, color="red", label="Predicted trend", linewidth=3)

        plt.xlabel("Time")
        plt.ylabel("Capital")
        plt.show()

    def survival_analysis(self, games):
        """
        Estimates survival function for a vector of simulated games. Values of p can vary.

        Uses the Kaplan-Meier estimator fitter from the lifelines package to compare the difference in

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
        plt.xlabel("Time elapsed (num gambles)")
        plt.ylabel("Proportion of games in play")
        plt.ylim(-0.05, 1.05)
        plt.show()

    def median_survival(self, games):
        """
        Calculates median of survival given a set of simulated games.

        Uses Kaplan-Meier estimate method to derive a survival function estimate and finds median of
        survival. Returns Pandas dataframe containing p value of game with survival median.

        Parameters:
        games (list): List of simulations for analysis.
        """
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
            temp = self.expectation - game["capital"]
            avg_err = np.average(temp)
            result = {"avg_error": avg_err, "ind_error": temp}
            err.append(pd.DataFrame(result))
        return err

if __name__ == '__main__':
    func = GamblingFunction(bin_vec=[0.25, 0.75, 1.25], payoff_vec=[2, 2, 2])
    print(func.get_value())
