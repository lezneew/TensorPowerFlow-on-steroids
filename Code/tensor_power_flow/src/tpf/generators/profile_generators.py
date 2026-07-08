# tensor_power_flow/src/tpf/generators/profile_generators.py
"""
Zeitreihen-Profilgeneratoren für PV-Erzeugung und Lasten.
"""
import numpy as np
from numpy.typing import NDArray
import pandapower as pp


def generate_pv_profile(
    n_pv: int, tau: int,
    profile_type: str = "daily_cosine",
    capacity_factor: float = 0.25,
    p_nom_mw: float | NDArray = 0.05,
    seed: int = 42,
) -> NDArray:
    """
    Erzeugt PV-Wirkleistungsprofile (n_pv × τ) in MW.

    profile_type:
      "constant"       — konstant = p_nom
      "daily_cosine"   — Tages-Cosinus, τ = Minuten (525600 = 1 Jahr @ 1min)
      "stochastic"     — Random walk um daily_cosine
      "step"           — Sprünge (Wolkendurchgang-artig)
    """
    rng = np.random.default_rng(seed)
    p_nom = np.atleast_1d(p_nom_mw)
    if p_nom.size == 1:
        p_nom = np.full(n_pv, p_nom.item())

    if profile_type == "constant":
        return np.tile(p_nom.reshape(-1, 1), (1, tau))

    if profile_type == "daily_cosine":
        # Annahme: 1 Zeitschritt = 1 Minute → 1440 Schritte/Tag
        t = np.arange(tau)
        day_frac = (t % 1440) / 1440.0   # [0, 1)
        # Cosinus zwischen Sonnenauf/-untergang (6-18 Uhr)
        solar = np.maximum(0.0, np.cos(np.pi * (day_frac - 0.5) * 2))
        # Skalierung: peak/mean = 1/capacity_factor → peak = p_nom/capacity_factor
        peak_factor = 1.0 / max(capacity_factor, 0.01) * (1.0 / (2/np.pi))
        profile = p_nom.reshape(-1, 1) * peak_factor * solar.reshape(1, -1)
        # Kleine PV-Variabilität zwischen Anlagen
        noise = 1.0 + 0.05 * rng.standard_normal((n_pv, tau))
        return np.maximum(0.0, profile * noise)

    if profile_type == "stochastic":
        base = generate_pv_profile(n_pv, tau, "daily_cosine",
                                    capacity_factor, p_nom_mw, seed)
        walk = np.cumsum(0.02 * rng.standard_normal((n_pv, tau)), axis=1)
        walk = np.clip(walk, -0.5, 0.5)
        return np.maximum(0.0, base * (1.0 + walk))

    if profile_type == "step":
        base = generate_pv_profile(n_pv, tau, "daily_cosine",
                                    capacity_factor, p_nom_mw, seed)
        # 5% der Zeitschritte: 30% Cloud dip für 10 min
        mask = np.zeros((n_pv, tau))
        n_events = int(0.005 * tau)
        for i in range(n_pv):
            starts = rng.integers(0, tau - 10, size=n_events)
            for s in starts:
                mask[i, s:s+10] = -0.7
        return np.maximum(0.0, base * (1.0 + mask))

    raise ValueError(f"Unbekannter profile_type: {profile_type}")


def generate_load_profile(
    net: pp.pandapowerNet, tau: int,
    profile_type: str = "daily_double_peak",
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """
    Erzeugt Lastprofile für alle Loads im Netz.

    Returns
    -------
    p_profile_mw : (n_loads, τ)
    q_profile_mvar : (n_loads, τ)
    """
    rng = np.random.default_rng(seed)
    n_loads = len(net.load)
    p_base = net.load["p_mw"].values.reshape(-1, 1)
    q_base = net.load["q_mvar"].values.reshape(-1, 1)

    if profile_type == "constant":
        lam = np.ones((1, tau))
    elif profile_type == "daily_double_peak":
        t = np.arange(tau)
        day_frac = (t % 1440) / 1440.0
        # Morgenspitze 7-9h, Abendspitze 18-21h
        morning = np.exp(-((day_frac - 8/24)**2) / (2 * (1/24)**2))
        evening = np.exp(-((day_frac - 19/24)**2) / (2 * (1.5/24)**2))
        base = 0.5 + 0.3 * morning + 0.5 * evening
        # Pro Last kleine Variabilität
        indiv = 1.0 + 0.1 * rng.standard_normal((n_loads, 1))
        noise = 1.0 + 0.05 * rng.standard_normal((n_loads, tau))
        lam = base.reshape(1, -1) * indiv * noise
    elif profile_type == "random":
        lam = rng.uniform(0.5, 1.5, size=(n_loads, tau))
    else:
        raise ValueError(f"Unbekannter profile_type: {profile_type}")

    p_profile = p_base * lam
    q_profile = q_base * lam
    return p_profile, q_profile