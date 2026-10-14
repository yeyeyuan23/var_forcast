"""第四步：评估的是分位数质量，而不是预测均值的 MSE。"""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2


def pinball_loss(y, q, alpha):
    error = np.asarray(y) - np.asarray(q)
    return np.maximum(alpha * error, (alpha - 1) * error)


def bernoulli_loglik(successes, total, probability):
    """xlogy(0, 0)=0，避免无突破或全突破时出现 0*log(0) 的 NaN。"""
    return float(
        xlogy(successes, probability) + xlogy(total - successes, 1 - probability)
    )


def coverage_tests(hits, alpha):
    """hits=1 表示真实收益率严格低于预测 VaR。

    Kupiec H0：突破概率为 alpha。
    Christoffersen independence H0：相邻突破指示相互独立（一阶检验）。
    Conditional coverage H0：覆盖正确且独立；LR_CC=LR_UC+LR_IND。
    渐近 p 值在稀疏尾部不可靠；无可估转移行时不输出伪造的 p 值。
    """
    hits = np.asarray(hits, dtype=bool)
    n, k = len(hits), int(hits.sum())
    if n < 2 or not 0 < alpha < 1:
        raise ValueError("检验至少需要两条观测，且 0 < alpha < 1")
    rate = k / n
    lr_uc = max(0.0, 2 * (bernoulli_loglik(k, n, rate) - bernoulli_loglik(k, n, alpha)))
    result = {
        "kupiec_lr": lr_uc,
        "kupiec_p": float(chi2.sf(lr_uc, 1)),
        "binomial_exact_p": float(binomtest(k, n, alpha).pvalue),
    }
    transitions = np.zeros((2, 2), dtype=int)
    np.add.at(transitions, (hits[:-1].astype(int), hits[1:].astype(int)), 1)
    n00, n01, n10, n11 = transitions.ravel()
    result.update(
        dict(zip(["n00", "n01", "n10", "n11"], map(int, transitions.ravel())))
    )
    notes = []
    if n * alpha < 20:
        notes.append("few expected tail events; asymptotic tests have limited power")
    if min(n00 + n01, n10 + n11) == 0:
        lr_ind = p_ind = lr_cc = p_cc = np.nan
        notes.append(
            "independence/conditional coverage undefined: unobserved transition row"
        )
    else:
        p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
        p = (n01 + n11) / (n - 1)
        unrestricted = bernoulli_loglik(n01, n00 + n01, p01) + bernoulli_loglik(
            n11, n10 + n11, p11
        )
        restricted = bernoulli_loglik(n01 + n11, n - 1, p)
        lr_ind = max(0.0, 2 * (unrestricted - restricted))
        p_ind = float(chi2.sf(lr_ind, 1))
        lr_cc = lr_uc + lr_ind
        p_cc = float(chi2.sf(lr_cc, 2))
        if transitions.min() < 5:
            notes.append("sparse transition cells; independence p-value is approximate")
    result.update(
        {
            "independence_lr": lr_ind,
            "independence_p": p_ind,
            "conditional_coverage_lr": lr_cc,
            "conditional_coverage_p": p_cc,
            "test_note": "; ".join(notes),
        }
    )
    return result


def evaluate(predictions):
    """所有方法必须在同一组日期、同一真实收益率上比较。"""
    if predictions.empty or predictions.duplicated(["date", "model", "alpha"]).any():
        raise ValueError("预测为空或存在重复预测")
    if not np.isfinite(predictions[["actual", "var"]].to_numpy()).all():
        raise ValueError("预测或实际收益率包含非有限值")
    if predictions.groupby("date")["actual"].nunique().max() != 1:
        raise ValueError("不同方法的真实收益率不一致")
    expected_alphas = set(predictions["alpha"])
    for _, model_rows in predictions.groupby("model"):
        if set(model_rows["alpha"]) != expected_alphas:
            raise ValueError("每个模型必须包含相同的分位数，不能遗漏整组预测")
    reference_dates = None
    records = []
    for (model, alpha), group in predictions.groupby(["model", "alpha"], sort=False):
        group = group.sort_values("date")
        dates = tuple(group["date"])
        if reference_dates is None:
            reference_dates = dates
        if dates != reference_dates:
            raise ValueError("所有方法和分位数必须在相同日期上评估")
        hits = (group["actual"] < group["var"]).to_numpy()
        records.append(
            {
                "model": model,
                "alpha": alpha,
                "n": len(group),
                "breaches": int(hits.sum()),
                "expected_breaches": len(group) * alpha,
                "failure_rate": hits.mean(),
                "coverage_gap": hits.mean() - alpha,
                "pinball_loss": pinball_loss(
                    group["actual"], group["var"], alpha
                ).mean(),
                "mean_var": group["var"].mean(),
                "raw_crossing_rate": group["raw_crossing"].mean(),
                "mean_fit_seconds": group["fit_seconds"].mean(),
                **coverage_tests(hits, alpha),
            }
        )
    return pd.DataFrame(records)
