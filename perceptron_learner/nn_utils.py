activationFunctions = {
    'sigmoid': lambda x: (1 / (1 + np.exp(-x))),
    'sigmoid_deriv': lambda x: activationFunctions['sigmoid'](x) * (1 - activationFunctions['sigmoid'](x)),

    'tanh': lambda x: np.tanh(x),
    'tanh_deriv': lambda x: (4 / ((np.exp(x) + np.exp(-x))**2)),

    'ReLu': lambda x: max(0, x),
    'ReLu_deriv': lambda x: 1 if x > 0 else 0
}


"""
Loss Functions
"""


def squared_loss(y_hat, y):
    return (y_hat - y) ** 2
