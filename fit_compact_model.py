#!/usr/bin/env python3
"""Compact-model parameter extraction from input.csv.

Install dependencies:
    pip install numpy scipy pandas

Run:
    python fit_compact_model.py

The program reads settings and numerical points from input.csv. Fitted
constants, equations, and errors are written to output.txt.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input.csv"
OUTPUT_FILE = BASE_DIR / "output.txt"

Q = 1.602176634e-19
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
EPS0 = 8.8541878128e-12
KB_EV = 8.617333262145e-5

def barrier_equation(
    thickness_nm: NDArray[np.float64],
    phi_b0: float,
    delta_phi_inf: float,
    lambda_ge: float,
) -> NDArray[np.float64]:
    return phi_b0 - delta_phi_inf * (1.0 - np.exp(-thickness_nm / lambda_ge))


def transport_energy(
    nd_cm3: NDArray[np.float64],
    temperature_k: float,
    tunneling_mass_ratio: float,
    relative_permittivity: float,
) -> NDArray[np.float64]:
    nd_m3 = nd_cm3 * 1.0e6
    effective_mass = tunneling_mass_ratio * M0
    permittivity = relative_permittivity * EPS0
    e00_j = Q * HBAR / 2.0 * np.sqrt(nd_m3 / (effective_mass * permittivity))
    e00_ev = e00_j / Q
    return e00_ev / np.tanh(e00_ev / (KB_EV * temperature_k))


def load_input() -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Load settings, barrier points, and transport points from input.csv."""

    data = pd.read_csv(INPUT_FILE)
    required_columns = {
        "section",
        "name",
        "value",
        "t_ge_nm",
        "phi_b_ev",
        "nd_cm3",
        "rho_c_ohm_cm2",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"input.csv is missing columns: {sorted(missing)}")

    setting_rows = data.loc[data["section"] == "setting", ["name", "value"]].dropna()
    settings = {
        str(row["name"]): float(row["value"])
        for _, row in setting_rows.iterrows()
    }
    required_settings = {
        "temperature_k",
        "reference_doping_cm3",
        "tunneling_mass_ratio",
        "relative_permittivity",
    }
    missing_settings = required_settings.difference(settings)
    if missing_settings:
        raise ValueError(f"input.csv is missing settings: {sorted(missing_settings)}")

    barrier_data = data.loc[
        data["section"] == "barrier", ["t_ge_nm", "phi_b_ev"]
    ].dropna()
    transport_data = data.loc[
        data["section"] == "transport",
        ["t_ge_nm", "nd_cm3", "rho_c_ohm_cm2"],
    ].dropna()
    barrier_data = barrier_data.apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    transport_data = transport_data.apply(pd.to_numeric, errors="raise").reset_index(drop=True)

    if len(barrier_data) < 3:
        raise ValueError("input.csv needs at least three barrier rows.")
    if len(transport_data) < 5:
        raise ValueError("input.csv needs at least five transport rows.")
    return settings, barrier_data, transport_data


def fit_barrier(barrier_data: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    thickness = barrier_data["t_ge_nm"].to_numpy(dtype=float)
    phi_b = barrier_data["phi_b_ev"].to_numpy(dtype=float)

    if np.unique(thickness).size < 3:
        raise ValueError("Three distinct thickness values are required.")
    if np.any(thickness < 0.0) or np.any(phi_b <= 0.0):
        raise ValueError("Thickness must be nonnegative and Phi_B must be positive.")

    phi_b0_guess = float(phi_b[np.argmin(thickness)])
    delta_guess = max(float(np.ptp(phi_b)) * 1.8, 1.0e-3)
    lambda_guess = max(float(np.ptp(thickness)), 0.1)

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        phi_b0, log_delta, log_lambda = parameters
        prediction = barrier_equation(
            thickness, phi_b0, math.exp(log_delta), math.exp(log_lambda)
        )
        return prediction - phi_b

    result = least_squares(
        residual,
        x0=np.array([phi_b0_guess, math.log(delta_guess), math.log(lambda_guess)]),
        bounds=(
            np.array([0.0, math.log(1.0e-12), math.log(1.0e-12)]),
            np.array([np.inf, math.log(1.0e3), math.log(1.0e6)]),
        ),
        method="trf",
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=100_000,
    )
    constants = np.array([result.x[0], math.exp(result.x[1]), math.exp(result.x[2])])
    prediction = barrier_equation(thickness, *constants)
    return constants, prediction


def fit_transport(
    settings: dict[str, float],
    transport_data: pd.DataFrame,
    barrier_constants: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    thickness = transport_data["t_ge_nm"].to_numpy(dtype=float)
    nd = transport_data["nd_cm3"].to_numpy(dtype=float)
    rho_c = transport_data["rho_c_ohm_cm2"].to_numpy(dtype=float)

    if np.any(thickness < 0.0) or np.any(nd <= 0.0) or np.any(rho_c <= 0.0):
        raise ValueError("Thickness must be nonnegative; ND and rho_c must be positive.")
    if np.unique(nd).size < 3:
        raise ValueError("At least three distinct ND values are required.")

    phi_b = barrier_equation(thickness, *barrier_constants)
    e0 = transport_energy(
        nd,
        settings["temperature_k"],
        settings["tunneling_mass_ratio"],
        settings["relative_permittivity"],
    )
    x = np.log(nd / settings["reference_doping_cm3"])
    observed_log_rho = np.log(rho_c)

    def predicted_log_rho(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        b0, b1, b2, log_s0, log_p = parameters
        s0 = math.exp(log_s0)
        p = math.exp(log_p)
        log_rho_f = b0 + b1 * x + b2 * x**2
        sensitivity = s0 * np.exp(-p * x)
        return log_rho_f + sensitivity * phi_b / e0

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return predicted_log_rho(parameters) - observed_log_rho

    initial = np.array(
        [float(np.median(observed_log_rho)) - 1.0, 0.0, 0.0, math.log(0.5), 0.0]
    )
    result = least_squares(
        residual,
        x0=initial,
        bounds=(
            np.array([-100.0, -100.0, -100.0, math.log(1.0e-12), math.log(1.0e-12)]),
            np.array([100.0, 100.0, 100.0, math.log(1.0e3), math.log(1.0e3)]),
        ),
        method="trf",
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=100_000,
    )
    constants = np.array(
        [result.x[0], result.x[1], result.x[2], math.exp(result.x[3]), math.exp(result.x[4])]
    )
    predicted_rho = np.exp(predicted_log_rho(result.x))
    return constants, phi_b, predicted_rho


def make_output(
    barrier_data: pd.DataFrame,
    barrier_constants: NDArray[np.float64],
    barrier_prediction: NDArray[np.float64],
    transport_data: pd.DataFrame,
    transport_constants: NDArray[np.float64],
    phi_b_used: NDArray[np.float64],
    predicted_rho: NDArray[np.float64],
    settings: dict[str, float],
) -> str:
    phi_b0, delta_phi_inf, lambda_ge = barrier_constants
    b0, b1, b2, s0, p = transport_constants

    barrier_error = barrier_prediction - barrier_data["phi_b_ev"].to_numpy(dtype=float)
    rho_observed = transport_data["rho_c_ohm_cm2"].to_numpy(dtype=float)
    relative_error = 100.0 * (predicted_rho / rho_observed - 1.0)
    log_residual = np.log(predicted_rho) - np.log(rho_observed)

    lines = [
        "COMPACT MODEL FIT OUTPUT",
        "=" * 78,
        "",
        "CONSTANTS",
        f"Phi_B0       = {phi_b0:.12g} eV",
        f"DeltaPhi_inf = {delta_phi_inf:.12g} eV",
        f"lambda       = {lambda_ge:.12g} nm",
        f"b0           = {b0:.12g}",
        f"b1           = {b1:.12g}",
        f"b2           = {b2:.12g}",
        f"S0           = {s0:.12g}",
        f"p            = {p:.12g}",
        "",
        "EQUATIONS",
        (
            f"Phi_B(t) = {phi_b0:.12g} - {delta_phi_inf:.12g} "
            f"* [1 - exp(-t/{lambda_ge:.12g})]"
        ),
        f"x = ln(ND/{settings['reference_doping_cm3']:.12g})",
        f"ln(rho_f) = {b0:.12g} + ({b1:.12g})*x + ({b2:.12g})*x^2",
        f"S(ND) = {s0:.12g} * exp(-{p:.12g}*x)",
        "rho_c = rho_f * exp[S(ND)*Phi_B/E0]",
        "",
        "FIT ERRORS",
        f"Barrier RMSE (eV)       = {np.sqrt(np.mean(barrier_error**2)):.12g}",
        f"log(rho_c) RMSE         = {np.sqrt(np.mean(log_residual**2)):.12g}",
        f"Maximum relative error = {np.max(np.abs(relative_error)):.12g} %",
        "",
        "POINT-BY-POINT TRANSPORT RESULTS",
        "t_nm, ND_cm^-3, Phi_B_eV, rho_observed, rho_predicted, error_percent",
    ]

    for index, row in transport_data.iterrows():
        lines.append(
            f"{row['t_ge_nm']:.9g}, {row['nd_cm3']:.9g}, {phi_b_used[index]:.9g}, "
            f"{row['rho_c_ohm_cm2']:.9g}, {predicted_rho[index]:.9g}, "
            f"{relative_error[index]:+.9g}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    print("COMPACT MODEL PARAMETER EXTRACTION")
    print(f"Reading: {INPUT_FILE}")

    settings, barrier_data, transport_data = load_input()

    barrier_constants, barrier_prediction = fit_barrier(barrier_data)
    transport_constants, phi_b_used, predicted_rho = fit_transport(
        settings, transport_data, barrier_constants
    )
    output = make_output(
        barrier_data,
        barrier_constants,
        barrier_prediction,
        transport_data,
        transport_constants,
        phi_b_used,
        predicted_rho,
        settings,
    )
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print("\n" + output)
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
