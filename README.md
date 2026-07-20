# ge-cap-contact-compact-model
Python tool for extracting thickness- and doping-dependent contact-resistivity compact-model parameters using bounded nonlinear least-squares fitting and CSV input.
Compact Model Parameter Extraction

A small Python tool for extracting thickness- and doping-dependent contact-resistivity compact-model parameters from numerical data.

The program reads model settings and data points from `input.csv`, performs bounded nonlinear least-squares fitting, and writes the fitted constants, equations, errors, and point-by-point predictions to `output.txt`.

## Scope

This tool is designed for datasets that follow the model equations described below. It is not a universal curve-fitting package and does not automatically select equations for unrelated applications.

The fitted constants are:

- `Phi_B0`: zero-thickness barrier
- `DeltaPhi_inf`: limiting barrier reduction
- `lambda`: characteristic thickness
- `b0`, `b1`, and `b2`: doping-dependent prefactor coefficients
- `S0` and `p`: barrier-sensitivity coefficients

## Model equations

### Barrier-thickness relation

The barrier is represented by

```text
Phi_B(t) = Phi_B0 - DeltaPhi_inf * [1 - exp(-t/lambda)]
```

where `t` is the layer thickness in nanometres.

### Doping coordinate

```text
x = ln(ND/N0)
```

where `ND` is the active donor concentration and `N0` is the reference concentration specified in `input.csv`.

### Resistivity prefactor

```text
ln(rho_f) = b0 + b1*x + b2*x^2
```

### Barrier sensitivity

```text
S(ND) = S0 * exp(-p*x)
```

### Specific contact resistivity

```text
rho_c = rho_f * exp[S(ND)*Phi_B/E0]
```

`E0` is calculated from the tunnelling energy using the temperature, tunnelling effective mass, relative permittivity, and donor concentration supplied in the input.

## Fitting method

The extraction is performed in two stages:

1. `Phi_B0`, `DeltaPhi_inf`, and `lambda` are fitted to the barrier-versus-thickness data.
2. `b0`, `b1`, `b2`, `S0`, and `p` are fitted simultaneously to the contact-resistivity data.

The second fit minimizes residuals in `ln(rho_c)` rather than raw `rho_c`. This is useful when resistivity spans multiple orders of magnitude and prevents the largest numerical values from dominating the optimization.

SciPy's bounded nonlinear least-squares solver is used. Parameters that must remain positive are optimized internally in logarithmic form.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy

Install the dependencies with:

```bash
pip install numpy pandas scipy
```

## Files

```text
fit_compact_model.py   Fitting algorithm
input.csv              Model settings and numerical input data
output.txt             Generated constants, equations, and errors
README.md              Documentation
```

## Input format

`input.csv` uses a single table with three section types.

### Settings

```csv
section,name,value,t_ge_nm,phi_b_ev,nd_cm3,rho_c_ohm_cm2
setting,temperature_k,300.0,,,,
setting,reference_doping_cm3,1e20,,,,
setting,tunneling_mass_ratio,0.19,,,,
setting,relative_permittivity,11.7,,,,
```

### Barrier data

Each `barrier` row supplies thickness and barrier height:

```csv
barrier,,,0.000,0.842,,
barrier,,,0.283,0.727,,
barrier,,,0.566,0.648,,
```

At least three distinct thickness values are required.

### Contact-resistivity data

Each `transport` row supplies thickness, donor concentration, and measured or simulated specific contact resistivity:

```csv
transport,,,0.000,,1e20,1.0208e-7
transport,,,0.283,,1e20,6.2870e-8
transport,,,0.566,,1e20,4.1830e-8
```

At least five transport rows and three distinct donor concentrations are required. Additional points are strongly recommended because the transport relation contains five fitted constants.

## Usage

Edit `input.csv`, then run:

```bash
python fit_compact_model.py
```

The program reads `input.csv` from the same directory and overwrites `output.txt` with the latest result.

## Output

`output.txt` contains:

- Extracted constants
- Equations with substituted numerical values
- Barrier-fit RMSE
- Log-resistivity RMSE
- Maximum relative resistivity error
- Observed and predicted values for every transport point

## Interpretation and extrapolation

The fitted equations can be evaluated at thicknesses and donor concentrations not explicitly present in the input data.

- Evaluation between supplied data points is interpolation.
- Evaluation outside the supplied thickness or doping range is extrapolation.

Extrapolated values are mathematical continuations of the selected equations. Their reliability generally decreases as the requested conditions move farther from the fitted range. Independent validation data should be used before treating extrapolated values as predictive.

An exact fit does not by itself establish physical validity. For example, three barrier points and three barrier constants leave no independent residual degrees of freedom. When possible, provide additional points, inspect residual errors, and validate the model against data not used during extraction.
