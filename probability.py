import enum, random
import matplotlib.pyplot as plt
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1 

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1 
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1 
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1 

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2 )/ (SQRT_TWO_PI * sigma))

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma))/ 2 


def inverse_normal_cdf(p: float, 
                       mu: float = 0, 
                       sigma: float = 1, 
                       tolerance: float = 0.00001) -> float:
    "Find approximate inverse using binary search"

    # if not standardized, compute standard and rescale
    if mu != 0 or sigma != 1:
        low_z = -10.0
        hi_z = 10.0
        while hi_z - low_z > tolerance:
            mid_z = (hi_z + low_z)/2
            mid_p = normal_cdf(mid_z)
            if mid_p < p:
                low_z = mid_z
            else:
                hi_z = mid_z
        return mid_z

def bernoulli_trial(p: float) -> float:
    """Returns 1 with probabiltiy p and 0 with probability 1 - p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) triasl"""
    return sum(bernoulli_trial(p) for _ in range  (n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n, p) and plots their histogram"""
    data = [binomial(n, p) for _ in range(num_points)]

    #use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v/ num_points for v in histogram.values()],
            0.8,
            color = '0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
            for i in xs]
    plt.plot(xs, ys)
    plt.title('Binomial Distribution vs. Normal Approximation')
    plt.show()

# binomial_histogram(0.75, 100, 10000)