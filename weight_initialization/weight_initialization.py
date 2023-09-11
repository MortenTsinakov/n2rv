import numpy as np


def random_normal(fan_in, fan_out):
    mu = 0.0
    sigma = 0.05
    return np.random.normal(loc=mu,
                            scale=sigma,
                            size=(fan_in, fan_out) if fan_in else (fan_out))


def random_uniform(fan_in, fan_out):
    limit = 0.05
    return np.random.uniform(low=-limit,
                             high=limit,
                             size=(fan_in, fan_out) if fan_in else (fan_out))


def zeros(fan_in, fan_out):
    return np.zeros((fan_in, fan_out) if fan_in else (fan_out))


def ones(fan_in, fan_out):
    return np.ones((fan_in, fan_out) if fan_in else fan_out)


def xavier_normal(fan_in, fan_out):
    mu = 0.0
    sigma = (2 / (fan_in + fan_out)) ** 0.5
    return np.random.normal(loc=mu,
                            scale=sigma,
                            size=(fan_in, fan_out) if fan_in else (fan_out))


def xavier_uniform(fan_in, fan_out):
    limit = (6 / (fan_in + fan_out)) ** 0.5
    return np.random.uniform(low=-limit,
                             high=limit,
                             size=(fan_in, fan_out) if fan_in else (fan_out))


def he_normal(fan_in, fan_out):
    mu = 0.0
    sigma = (2 / fan_in) ** 0.5
    return np.random.normal(loc=mu,
                            scale=sigma,
                            size=(fan_in, fan_out) if fan_in else (fan_out))


def he_uniform(fan_in, fan_out):
    limit = (6 / fan_in) ** 0.5
    return np.random.uniform(low=-limit,
                             high=limit,
                             size=(fan_in, fan_out) if fan_in else (fan_out))


def get_weights(fan_in, fan_out, init_technique=None):
    if init_technique not in init_techniques:
        raise ValueError(
            "Requested weight initialization techinque " +
            f"doesn not excist: {init_technique}."
        )
    return init_techniques[init_technique](fan_in, fan_out)


init_techniques = {
    None: random_normal,
    'random_normal': random_normal,
    'random_uniform': random_uniform,
    'zeros': zeros,
    'ones': ones,
    'xavier_normal': xavier_normal,
    'xavier_uniform': xavier_uniform,
    'he_normal': he_normal,
    'he_uniform': he_uniform,
}
