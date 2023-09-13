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
    validate_get_weights_parameters(fan_in, fan_out, init_technique)
    return init_techniques[init_technique](fan_in, fan_out)


def validate_get_weights_parameters(fan_in, fan_out, init_technique):
    if fan_in is not None and\
       not isinstance(fan_in, int) and\
       not isinstance(fan_in, np.integer):
        raise TypeError(
            "Fan_in in weight_initialization.get_weights() should be " +
            f"either None or an integer. Got: {type(fan_in)}."
        )
    if not isinstance(fan_out, int) and\
       not isinstance(fan_out, np.integer):
        raise TypeError(
            "Fan_out in weight_initialization.get_weights() should be " +
            f"an integer. Got: {type(fan_out)}."
        )
    if isinstance(fan_in, int) and fan_in <= 0:
        raise ValueError(
            "Fan_in in weight_initialization.get_weights() has to be " +
            f"None or >0. Got fan_in={fan_in}."
        )
    if fan_out <= 0:
        raise ValueError(
            "Fan_out in weight_initialization.get_weights() has to be " +
            f">0. Got {fan_out}."
        )
    if init_technique not in init_techniques:
        raise ValueError(
            "Requested weight initialization techinque " +
            f"doesn not excist: {init_technique}."
        )


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
