from __future__ import annotations

import math

from flexddm import define_model


def gamma_pulse(time, shape, characteristic_time, peak_amplitude):
    time = max(time, 1e-6)
    shape = max(shape, 1.0001)
    characteristic_time = max(characteristic_time, 1e-6)
    base = (time * math.e) / ((shape - 1) * characteristic_time)
    return (
        peak_amplitude
        * math.exp(-(time / characteristic_time))
        * math.pow(base, shape - 1)
        * (((shape - 1) / time) - (1 / characteristic_time))
    )


def drift(t, condition, alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, collapse_rate, collapse_shape, collapse_amount):
    direction = 1 if condition == "0" else -1
    return mu_c + direction * gamma_pulse(t + tau, shape, characteristic_time, peak_amplitude)


def bound(t, condition, alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, collapse_rate, collapse_shape, collapse_amount):
    scaled_time = max(t, 0.0) / max(collapse_rate, 1e-6)
    collapse = collapse_amount * (1.0 - math.exp(-(scaled_time ** collapse_shape)))
    return max((alpha / 2) * (1.0 - collapse), 0.02)


process_model = define_model(
    name="DMC_Collapsing",
    conditions=("0", "1"),
    drift=drift,
    bound=bound,
    nondecision="tau",
)

model = process_model.to_simulation_model()
