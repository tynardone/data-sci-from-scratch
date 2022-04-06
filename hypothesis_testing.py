from typing import Tuple

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * n * (1 - p))
    return mu, sigma

from probability import normal_cdf

# the normal cdif is the probability the variable is below a threshold
normal_probability_below = normal_cdf

# Its above the threshold if it's not below the threshold 
def normal_probability_above(lo: float, 
                            mu: float = 0, 
                            sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo: float,
                                hi: float,
                                mu: float = 0,
                                sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between

from probability import inverse_normal_cdf

def import_upper_bound(probability: float,
                        mu: float = 0,
                        sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)