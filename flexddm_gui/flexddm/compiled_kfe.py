from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(function):
            return function

        return decorator


@njit(cache=True)
def solve_kfe_arrays(initial_density, mu_grid, sigma_values, dtbase, dx):
    nt = sigma_values.shape[0] - 1
    nx = initial_density.shape[0] - 1
    density = initial_density.copy()
    upper_pdf = np.zeros(nt + 1)
    lower_pdf = np.zeros(nt + 1)
    tol = 1.0e-11

    for time_index in range(1, nt + 1):
        density[0] = 0.0
        density[nx] = 0.0
        mass = 0.0
        for i in range(nx + 1):
            if density[i] > 0.0:
                mass += density[i]
        if mass * dx <= 0.0:
            break

        density = _adaptive_step(
            density,
            mu_grid[time_index - 1],
            mu_grid[time_index],
            sigma_values[time_index - 1],
            sigma_values[time_index],
            dtbase,
            dx,
        )

        sigma_new = sigma_values[time_index]
        upper_value = 0.5 * sigma_new * sigma_new / (dx * dx) * (
            3.0 * density[nx - 1] - 1.5 * density[nx - 2] + (1.0 / 3.0) * density[nx - 3]
        )
        lower_value = 0.5 * sigma_new * sigma_new / (dx * dx) * (
            3.0 * density[1] - 1.5 * density[2] + (1.0 / 3.0) * density[3]
        )
        upper_pdf[time_index] = upper_value if upper_value > 0.0 else 0.0
        lower_pdf[time_index] = lower_value if lower_value > 0.0 else 0.0

        acc = 0.0
        for j in range(time_index + 1):
            acc += upper_pdf[j] + lower_pdf[j]
        if acc * dtbase > 0.999:
            break

    scale = 0.0
    for i in range(nt + 1):
        scale += upper_pdf[i] + lower_pdf[i]
    scale *= dtbase
    if scale > 0.0 and np.isfinite(scale):
        for i in range(nt + 1):
            upper_pdf[i] /= scale
            lower_pdf[i] /= scale

    return upper_pdf, lower_pdf


@njit(cache=True)
def _adaptive_step(density, mu_old, mu_new, sigma_old, sigma_new, dtbase, dx):
    accepted = density.copy()
    elapsed = 0.0
    dt = dtbase
    nsteps = 0
    tol = 1.0e-11

    while elapsed < dtbase - tol:
        candidate = _crank_nicolson_substep(accepted, mu_old, mu_new, sigma_old, sigma_new, dt, dx)
        step_ok = True
        for value in candidate:
            if value < -tol or not np.isfinite(value):
                step_ok = False
                break

        if step_ok:
            for i in range(candidate.shape[0]):
                accepted[i] = candidate[i] if candidate[i] > 0.0 else 0.0
            accepted[0] = 0.0
            accepted[accepted.shape[0] - 1] = 0.0
            elapsed += dt
            dt = 2.0 * dt
            if dt > dtbase:
                dt = dtbase
            remaining = dtbase - elapsed
            if dt > remaining:
                dt = remaining
        else:
            dt *= 0.125

        nsteps += 1
        if nsteps > 1000:
            return accepted

    return accepted


@njit(cache=True)
def _crank_nicolson_substep(density, mu_old, mu_new, sigma_old, sigma_new, dt, dx):
    size = density.shape[0]
    nx = size - 1
    l_old = 0.5 * sigma_old * sigma_old / dx
    l_new = 0.5 * sigma_new * sigma_new / dx

    rhs = np.zeros(size)
    lower = np.zeros(size)
    main = np.ones(size)
    upper = np.zeros(size)

    for i in range(1, nx):
        rhs[i] = dx * density[i] - 0.5 * dt * (
            (-l_old - 0.5 * mu_old[i]) * density[i - 1]
            + (2.0 * l_old) * density[i]
            + (-l_old + 0.5 * mu_old[i]) * density[i + 1]
        )
        lower[i] = 0.5 * dt * (-l_new - 0.5 * mu_new[i])
        main[i] = dx + 0.5 * dt * (2.0 * l_new)
        upper[i] = 0.5 * dt * (-l_new + 0.5 * mu_new[i])

    return _solve_tridiagonal(lower, main, upper, rhs)


@njit(cache=True)
def _solve_tridiagonal(lower, main, upper, rhs):
    n = rhs.shape[0]
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    result = np.zeros(n)

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]
    for i in range(1, n):
        denominator = main[i] - lower[i] * c_prime[i - 1]
        if abs(denominator) < 1.0e-300:
            denominator = 1.0e-300
        c_prime[i] = upper[i] / denominator if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denominator

    result[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        result[i] = d_prime[i] - c_prime[i] * result[i + 1]

    return result
