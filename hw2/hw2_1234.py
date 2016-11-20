# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt

N_COINS = 1000
N_PROBES = 10000
N_FLIPS = 10
EPS_TOLERANCE = 1e-7

def flip_coins():
    heads = np.zeros(N_COINS, dtype=int)
    for flip in range(N_FLIPS):
        heads += np.random.random_sample(size=N_COINS) > 0.5
    return heads

def check_heads_probabilities_sum(heads_target):
    head_prob_sum = np.sum(heads_target, axis=1)
    prob_sum_test_passed = np.abs(head_prob_sum - np.ones(3)) < EPS_TOLERANCE
    assert np.all(prob_sum_test_passed)

def flip_coins_histogram():
    """
     Covers questions #1 and #2.
     Since expected value (mean) of getting a head for one coin is 0.5,
     only c1 and c_rand distributions satisfy Hoeffding inequality.
    """
    heads_target = np.zeros((3, N_FLIPS+1))
    fractions = np.zeros(3)
    for probe in range(N_PROBES):
        heads = flip_coins()
        c1 = 0
        c_rand = np.random.randint(0, N_COINS)
        c_min = np.argmin(heads)
        for coin_type, coin_id in enumerate([c1, c_rand, c_min]):
            coin_heads = heads[coin_id]
            fractions[coin_type] += coin_heads
            heads_target[coin_type, coin_heads] += 1
    heads_target /= N_PROBES
    fractions /= N_PROBES * N_FLIPS
    check_heads_probabilities_sum(heads_target)
    print("c1, c_rand, c_min fractions: ", fractions)

    coins = np.arange(0, N_FLIPS+1)
    plt.plot(coins, heads_target[0], 'r', label="c1")
    plt.plot(coins, heads_target[1], 'g', label="c_rand")
    plt.plot(coins, heads_target[2], 'b', label="c_min")
    plt.xlabel("#heads")
    plt.ylabel("P(#heads)")
    plt.title("Probability distribution of getting #heads out of 10 flips for each of 3 coins.")
    plt.legend()
    plt.show()

    return heads_target

def q3():
    """
     P(error) = P(hypothesis miss) * P(no noise in Y) + P(hypothesis hit) * P(noisy Y)
              = mu * lambda + (1 - u) * (1 - lambda)
    """
    pass

def q4():
    """
     P(error) = mu * (2 * lambda - 1) + 1 - lambda won't depend of mu when lambda = 0.5
     ==> P(error|lambda=0.5) = 0.5;
     ==> any model, that tries to fit a white noise, would give the same performance
         with probability of being wrong equals to P(error) = 0.5.
    """

if __name__ == "__main__":
    flip_coins_histogram()

