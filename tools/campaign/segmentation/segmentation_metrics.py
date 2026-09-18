"""Channel-B metrics and preregistered envelopes for public segmentation evidence."""
import math
import numpy as np


def subject_scores(probability, reference):
    probability, reference = np.asarray(probability), np.asarray(reference)
    if probability.shape != reference.shape or probability.ndim != 4:
        raise ValueError("probabilities and masks must share [N,1,H,W] geometry")
    if probability.shape[1] != 1 or not np.isfinite(probability).all():
        raise ValueError("finite binary segmentation probabilities required")
    if not np.isin(reference, [0, 1]).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("declared binary masks and probabilities in [0,1] required")
    p, y = (probability >= 0.5).reshape(len(reference), -1), reference.reshape(len(reference), -1).astype(bool)
    intersection = (p & y).sum(1)
    sizes, union = p.sum(1) + y.sum(1), (p | y).sum(1)
    dice, iou = np.ones(len(reference)), np.ones(len(reference))
    np.divide(2 * intersection, sizes, out=dice, where=sizes != 0)
    np.divide(intersection, union, out=iou, where=union != 0)
    return dice, iou, y.any(1)


def metrics(probability, reference):
    dice, iou, positive = subject_scores(probability, reference)
    def score(index):
        return None if not np.any(index) else {"dice": float(dice[index].mean()), "iou": float(iou[index].mean())}
    return {"all": score(np.ones(len(dice), bool)),
            "foreground_positive": score(positive), "empty_reference": score(~positive)}


def trivial_masks(reference):
    reference = np.asarray(reference)
    if tuple(reference.shape[1:]) != (1, 128, 128):
        raise ValueError("trivial masks are pinned to [N,1,128,128]")
    yy, xx = np.mgrid[:128, :128]
    # Center at the geometric grid center; no data-dependent fitting.
    shapes = {"empty": np.zeros((128, 128)), "full": np.ones((128, 128)),
              "disk_r32": (xx - 63.5) ** 2 + (yy - 63.5) ** 2 <= 32 ** 2,
              "square_64": (np.abs(xx - 63.5) < 32) & (np.abs(yy - 63.5) < 32)}
    scores = {name: metrics(np.broadcast_to(mask, reference.shape), reference)
              for name, mask in shapes.items()}
    return {"scores": scores,
            "strongest_dice": max(value["all"]["dice"] for value in scores.values())}


def mean_interval(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("at least three finite independent public replicates required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    margin = float(t.ppf(.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci95": [mean - margin, mean + margin], "n_replicates": len(values)}


def envelopes(replicates, small_replicates=None):
    """Rows: seed/epsilon/n_train/n_per_site/central_dice/federated_dice/trivial_dice.

    Run on one cohort. Input omissions fail; never fill an unexecuted cell with
    synthetic zeros. A true flag means investigate, not a failed privacy proof.
    """
    by_epsilon = {}
    for epsilon in (1, 4, 8):
        rows = sorted((r for r in replicates if r["epsilon"] == epsilon), key=lambda r: r["seed"])
        if len(rows) < 3 or len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("each epsilon needs at least three distinct seeds")
        by_epsilon[epsilon] = rows
    seeds = [{r["seed"] for r in rows} for rows in by_epsilon.values()]
    if not all(value == seeds[0] for value in seeds[1:]):
        raise ValueError("epsilon comparisons require matched seed sets")
    gaps = {epsilon: mean_interval([r["central_dice"] - r["federated_dice"] for r in rows])
            for epsilon, rows in by_epsilon.items()}
    adjacent = []
    for previous, current in ((1, 4), (4, 8)):
        change = gaps[current]["mean"] - gaps[previous]["mean"]
        tolerance = max(gaps[previous]["sd"], gaps[current]["sd"])
        adjacent.append({"previous": previous, "next": current, "gap_increase": change,
                         "tolerance": tolerance, "flag": change > tolerance})
    high = by_epsilon[8]
    fed, trivial = np.mean([r["federated_dice"] for r in high]), np.mean([r["trivial_dice"] for r in high])
    floor = {"federated_mean_dice": float(fed), "strongest_trivial_mean_dice": float(trivial),
             "absolute_pass": bool(fed >= .50), "margin_pass": bool(fed >= trivial + .10)}
    floor["pass"] = floor["absolute_pass"] and floor["margin_pass"]
    near = [{"epsilon": r["epsilon"], "seed": r["seed"], "n_train": r["n_train"],
             "smallest_site_n": min(r["n_per_site"]),
             "flag": r["n_train"] * r["epsilon"] < 2000 and r["central_dice"] < .95
                     and abs(r["central_dice"] - r["federated_dice"]) < .005}
            for r in replicates]
    trend = {"status": "not_executed"}
    if small_replicates is not None:
        for epsilon in (1, 4, 8):
            small_seeds = [r["seed"] for r in small_replicates if r["epsilon"] == epsilon]
            if len(small_seeds) != len(set(small_seeds)) or set(small_seeds) != seeds[0]:
                raise ValueError("small-N trend requires the complete matched seed matrix")
        low = sorted((r for r in small_replicates if r["epsilon"] == 1), key=lambda r: r["seed"])
        high = sorted((r for r in small_replicates if r["epsilon"] == 8), key=lambda r: r["seed"])
        if [r["seed"] for r in low] != [r["seed"] for r in high]:
            raise ValueError("small-N trend requires matched seeds")
        sd_low = mean_interval([r["central_dice"] - r["federated_dice"] for r in low])["sd"]
        sd_high = mean_interval([r["central_dice"] - r["federated_dice"] for r in high])["sd"]
        trend = {"status": "executed", "sd_epsilon1": sd_low, "sd_epsilon8": sd_high,
                 "flag": sd_high > sd_low}
    return {"gap_sign": "pooled_nonprivate_minus_federated_dp", "gap_summary": gaps,
            "epsilon_envelope": adjacent, "utility_floor": floor,
            "small_n_noise_trend": trend, "near_central": near}
