"""Constant-mean GARCH(1,1), fitted by likelihood to percentage returns."""

import numpy as np
from scipy.stats import norm, t
from arch import arch_model


def innovation_quantiles(alphas, distribution, nu=None):
    if distribution == "normal":
        return norm.ppf(alphas)
    if distribution != "t" or nu is None or nu <= 2:
        raise ValueError("Student-t innovations require nu > 2")
    return t.ppf(alphas, nu) * np.sqrt((nu - 2) / nu)


def fit_garch(returns_percent, distribution):
    model = arch_model(
        returns_percent,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist=distribution,
        rescale=False,
    )
    fit = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    if fit.convergence_flag != 0:
        fit = model.fit(
            disp="off", show_warning=False, tol=1e-7, options={"maxiter": 5000}
        )
    if fit.convergence_flag != 0:
        raise RuntimeError(
            f"GARCH fit did not converge: {fit.optimization_result.message}"
        )
    return fit


def forecast(X_train, y_train, x_next, config):
    fitted = fit_garch(100.0 * y_train, config.distribution)
    forecast = fitted.forecast(horizon=1, reindex=False)
    mu = forecast.mean.values[-1, 0]
    variance = forecast.residual_variance.values[-1, 0]
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid GARCH variance forecast")
    quantiles = innovation_quantiles(
        config.alphas, config.distribution, fitted.params.get("nu")
    )
    return np.asarray((mu + np.sqrt(variance) * quantiles) / 100.0)
