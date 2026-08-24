#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

double normal_cdf(double z) {
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

template <typename View>
void require_1d(const View &view, const char *name) {
  if (view.ndim() != 1) {
    throw std::invalid_argument(std::string(name) + " must be one-dimensional.");
  }
}

template <typename View>
void require_2d(const View &view, const char *name) {
  if (view.ndim() != 2) {
    throw std::invalid_argument(std::string(name) + " must be two-dimensional.");
  }
}

} // namespace

py::tuple solve_transition(
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> edge_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> bound_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> drift_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> noise_array,
    double start,
    double dt) {
  auto grid = grid_array.unchecked<1>();
  auto edges = edge_array.unchecked<1>();
  auto bounds = bound_array.unchecked<1>();
  auto drift = drift_array.unchecked<2>();
  auto noise = noise_array.unchecked<2>();

  require_1d(grid, "grid");
  require_1d(edges, "edges");
  require_1d(bounds, "bounds");
  require_2d(drift, "drift");
  require_2d(noise, "noise");

  const py::ssize_t nx = grid.shape(0);
  const py::ssize_t nt = bounds.shape(0);
  if (nx < 5) {
    throw std::invalid_argument("grid must contain at least five points.");
  }
  if (edges.shape(0) != nx + 1) {
    throw std::invalid_argument("edges must have one more entry than grid.");
  }
  if (drift.shape(0) < nt - 1 || drift.shape(1) != nx) {
    throw std::invalid_argument("drift must have shape (len(bounds)-1, len(grid)).");
  }
  if (noise.shape(0) < nt - 1 || noise.shape(1) != nx) {
    throw std::invalid_argument("noise must have shape (len(bounds)-1, len(grid)).");
  }
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    throw std::invalid_argument("dt must be finite and positive.");
  }

  std::vector<double> state(static_cast<std::size_t>(nx), 0.0);
  std::vector<double> next(static_cast<std::size_t>(nx), 0.0);
  std::vector<double> upper_mass(static_cast<std::size_t>(nt), 0.0);
  std::vector<double> lower_mass(static_cast<std::size_t>(nt), 0.0);

  py::ssize_t start_index = 0;
  double best = std::numeric_limits<double>::infinity();
  for (py::ssize_t i = 0; i < nx; ++i) {
    const double distance = std::abs(grid(i) - start);
    if (distance < best) {
      best = distance;
      start_index = i;
    }
  }
  state[static_cast<std::size_t>(start_index)] = 1.0;

  const double sqrt_dt = std::sqrt(dt);
  double captured = 0.0;
  for (py::ssize_t t = 1; t < nt; ++t) {
    const double current_bound = bounds(t - 1);
    const double next_bound = bounds(t);
    if (!(current_bound > 0.0) || !(next_bound > 0.0) || !std::isfinite(current_bound) || !std::isfinite(next_bound)) {
      throw std::runtime_error("invalid boundary value.");
    }

    double alive = 0.0;
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (std::abs(grid(i)) >= current_bound) {
        state[static_cast<std::size_t>(i)] = 0.0;
      } else if (state[static_cast<std::size_t>(i)] > 0.0) {
        alive += state[static_cast<std::size_t>(i)];
      }
    }
    if (alive <= 0.0) {
      break;
    }

    std::fill(next.begin(), next.end(), 0.0);
    double upper_step = 0.0;
    double lower_step = 0.0;

    for (py::ssize_t j = 0; j < nx; ++j) {
      const double mass = state[static_cast<std::size_t>(j)];
      if (mass <= 0.0 || std::abs(grid(j)) >= current_bound) {
        continue;
      }

      const double local_noise = std::max(std::abs(noise(t - 1, j)), 1.0e-12);
      const double sigma = std::max(local_noise * sqrt_dt, 1.0e-12);
      const double mean = grid(j) + drift(t - 1, j) * dt;

      const double lower_cross = normal_cdf((-next_bound - mean) / sigma);
      const double upper_cross = 1.0 - normal_cdf((next_bound - mean) / sigma);
      lower_step += mass * std::max(0.0, lower_cross);
      upper_step += mass * std::max(0.0, upper_cross);

      for (py::ssize_t i = 0; i < nx; ++i) {
        if (std::abs(grid(i)) >= next_bound) {
          continue;
        }
        const double lo = std::max(edges(i), -next_bound);
        const double hi = std::min(edges(i + 1), next_bound);
        if (hi <= lo) {
          continue;
        }
        double probability = normal_cdf((hi - mean) / sigma) - normal_cdf((lo - mean) / sigma);
        if (std::isfinite(probability) && probability > 0.0) {
          next[static_cast<std::size_t>(i)] += mass * probability;
        }
      }
    }

    upper_mass[static_cast<std::size_t>(t)] = upper_step;
    lower_mass[static_cast<std::size_t>(t)] = lower_step;
    captured += upper_step + lower_step;
    state.swap(next);
    if (captured > 0.999) {
      break;
    }
  }

  py::array_t<double> upper_pdf(nt);
  py::array_t<double> lower_pdf(nt);
  auto upper = upper_pdf.mutable_unchecked<1>();
  auto lower = lower_pdf.mutable_unchecked<1>();

  double scale = 0.0;
  for (py::ssize_t i = 0; i < nt; ++i) {
    scale += upper_mass[static_cast<std::size_t>(i)] + lower_mass[static_cast<std::size_t>(i)];
  }
  if (scale > 0.0 && std::isfinite(scale)) {
    for (py::ssize_t i = 0; i < nt; ++i) {
      upper(i) = (upper_mass[static_cast<std::size_t>(i)] / dt) / scale;
      lower(i) = (lower_mass[static_cast<std::size_t>(i)] / dt) / scale;
    }
  } else {
    for (py::ssize_t i = 0; i < nt; ++i) {
      upper(i) = 0.0;
      lower(i) = 0.0;
    }
  }

  return py::make_tuple(upper_pdf, lower_pdf);
}

py::tuple solve_kfe(
    py::array_t<double, py::array::c_style | py::array::forcecast> initial_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> bound_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> drift_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> bound_derivative_array,
    py::object sigma_object,
    double dtbase,
    double dx) {
  auto initial = initial_array.unchecked<1>();
  auto x_vec = x_array.unchecked<1>();
  auto bounds = bound_array.unchecked<1>();
  auto drift = drift_array.unchecked<1>();
  auto dt_bound = bound_derivative_array.unchecked<1>();

  require_1d(initial, "initial");
  require_1d(x_vec, "x_vec");
  require_1d(bounds, "bounds");
  require_1d(drift, "drift");
  require_1d(dt_bound, "bound_derivative");

  const py::ssize_t size = initial.shape(0);
  const py::ssize_t nx = size - 1;
  const py::ssize_t nt = bounds.shape(0) - 1;
  if (size < 5) {
    throw std::invalid_argument("initial must contain at least five grid points.");
  }
  if (x_vec.shape(0) != size) {
    throw std::invalid_argument("x_vec must have the same length as initial.");
  }
  if (drift.shape(0) != nt + 1 || dt_bound.shape(0) != nt + 1) {
    throw std::invalid_argument("drift and bound_derivative must match bounds length.");
  }
  if (!(dtbase > 0.0) || !(dx > 0.0) || !std::isfinite(dtbase) || !std::isfinite(dx)) {
    throw std::invalid_argument("dt and dx must be finite and positive.");
  }

  std::vector<double> sigma_values(static_cast<std::size_t>(nt + 1), 0.0);
  if (py::isinstance<py::array>(sigma_object)) {
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma_array = sigma_object.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto sigma_view = sigma_array.unchecked<1>();
    require_1d(sigma_view, "sigma");
    if (sigma_view.shape(0) != nt + 1) {
      throw std::invalid_argument("sigma array must match bounds length.");
    }
    for (py::ssize_t i = 0; i <= nt; ++i) {
      sigma_values[static_cast<std::size_t>(i)] = sigma_view(i);
    }
  } else {
    const double sigma_scalar = sigma_object.cast<double>();
    for (py::ssize_t i = 0; i <= nt; ++i) {
      sigma_values[static_cast<std::size_t>(i)] = sigma_scalar;
    }
  }
  for (double sigma_value : sigma_values) {
    if (!(sigma_value > 0.0) || !std::isfinite(sigma_value)) {
      throw std::invalid_argument("sigma values must be finite and positive.");
    }
  }

  std::vector<double> xx(static_cast<std::size_t>(size), 0.0);
  std::vector<double> f(static_cast<std::size_t>(size), 0.0);
  std::vector<double> mu_old(static_cast<std::size_t>(size), 0.0);
  std::vector<double> mu_new(static_cast<std::size_t>(size), 0.0);
  std::vector<double> a(static_cast<std::size_t>(size), 0.0);
  std::vector<double> b(static_cast<std::size_t>(size), 0.0);
  std::vector<double> c(static_cast<std::size_t>(size), 0.0);
  std::vector<double> xxnew(static_cast<std::size_t>(size), 0.0);
  std::vector<double> upper_raw(static_cast<std::size_t>(nt + 1), 0.0);
  std::vector<double> lower_raw(static_cast<std::size_t>(nt + 1), 0.0);

  for (py::ssize_t i = 0; i < size; ++i) {
    xx[static_cast<std::size_t>(i)] = initial(i);
  }

  const double tol = 1.0e-11;
  double acc = 0.0;

  for (py::ssize_t n = 1; n <= nt; ++n) {
    const double j_old = bounds(n - 1);
    const double j_new = bounds(n);
    if (!(j_old > 0.0) || !(j_new > 0.0) || !std::isfinite(j_old) || !std::isfinite(j_new)) {
      throw std::runtime_error("invalid boundary value.");
    }

    const double sigma_old = sigma_values[static_cast<std::size_t>(n - 1)] / j_old;
    const double sigma_new = sigma_values[static_cast<std::size_t>(n)] / j_new;
    const double l_old = sigma_old * sigma_old / (2.0 * dx);
    const double l_new = sigma_new * sigma_new / (2.0 * dx);

    for (py::ssize_t i = 0; i < size; ++i) {
      mu_old[static_cast<std::size_t>(i)] = (drift(n - 1) - dt_bound(n - 1) * x_vec(i)) / j_old;
      mu_new[static_cast<std::size_t>(i)] = (drift(n) - dt_bound(n) * x_vec(i)) / j_new;
    }

    double step_elapsed = 0.0;
    double dt = dtbase;
    int nsteps = 0;

    while (step_elapsed < dtbase - tol) {
      f[0] = 0.0;
      f[static_cast<std::size_t>(nx)] = 0.0;

      for (py::ssize_t i = 1; i < nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        f[idx] = dx * xx[idx] - dt * 0.5 *
          ((-l_old - 0.5 * mu_old[idx]) * xx[idx - 1] +
           (2.0 * l_old) * xx[idx] +
           (-l_old + 0.5 * mu_old[idx]) * xx[idx + 1]);
      }

      for (py::ssize_t i = 0; i < size; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        a[idx] = dt * 0.5 * (-l_new - 0.5 * mu_new[idx]);
        b[idx] = dx + dt * 0.5 * (2.0 * l_new);
        c[idx] = dt * 0.5 * (-l_new + 0.5 * mu_new[idx]);
      }
      a[0] = 0.0;
      b[0] = 1.0;
      c[0] = 0.0;
      a[static_cast<std::size_t>(nx)] = 0.0;
      b[static_cast<std::size_t>(nx)] = 1.0;
      c[static_cast<std::size_t>(nx)] = 0.0;

      c[0] /= b[0];
      for (py::ssize_t i = 1; i < nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double denom = b[idx] - c[idx - 1] * a[idx];
        c[idx] /= denom;
      }
      f[0] /= b[0];
      for (py::ssize_t i = 1; i <= nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double denom = b[idx] - c[idx - 1] * a[idx];
        f[idx] = (f[idx] - f[idx - 1] * a[idx]) / denom;
      }

      bool step_ok = true;
      xxnew[static_cast<std::size_t>(nx)] = f[static_cast<std::size_t>(nx)];
      for (py::ssize_t i = nx - 1; i >= 0; --i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        xxnew[idx] = f[idx] - c[idx] * xxnew[idx + 1];
        if (xxnew[idx] < -tol || !std::isfinite(xxnew[idx])) {
          step_ok = false;
          break;
        }
      }

      if (step_ok) {
        for (py::ssize_t i = 0; i <= nx; ++i) {
          xx[static_cast<std::size_t>(i)] = std::max(0.0, xxnew[static_cast<std::size_t>(i)]);
        }
        xx[0] = 0.0;
        xx[static_cast<std::size_t>(nx)] = 0.0;
        step_elapsed += dt;
        dt = std::min(dtbase, 2.0 * dt);
        dt = std::min(dt, dtbase - step_elapsed);
      } else {
        dt *= 0.125;
      }

      if (nsteps > 1000) {
        throw std::runtime_error("Extreme parameters: number of adaptive KFE steps exceeded.");
      }
      ++nsteps;
    }

    double upper_value = 0.5 * sigma_new * sigma_new / (dx * dx) *
      (3.0 * xx[static_cast<std::size_t>(nx - 1)] -
       1.5 * xx[static_cast<std::size_t>(nx - 2)] +
       (1.0 / 3.0) * xx[static_cast<std::size_t>(nx - 3)]);
    double lower_value = 0.5 * sigma_new * sigma_new / (dx * dx) *
      (3.0 * xx[1] - 1.5 * xx[2] + (1.0 / 3.0) * xx[3]);

    if (upper_value < -tol) {
      upper_value = 0.5 * sigma_new * sigma_new / (dx * dx) * xx[static_cast<std::size_t>(nx - 1)];
    }
    if (lower_value < -tol) {
      lower_value = 0.5 * sigma_new * sigma_new / (dx * dx) * xx[1];
    }
    upper_raw[static_cast<std::size_t>(n)] = std::max(0.0, upper_value);
    lower_raw[static_cast<std::size_t>(n)] = std::max(0.0, lower_value);

    acc += upper_raw[static_cast<std::size_t>(n)] + lower_raw[static_cast<std::size_t>(n)];
    if (acc * dx * dtbase > 0.999) {
      break;
    }
  }

  py::array_t<double> upper_pdf(nt + 1);
  py::array_t<double> lower_pdf(nt + 1);
  auto upper = upper_pdf.mutable_unchecked<1>();
  auto lower = lower_pdf.mutable_unchecked<1>();

  double scale = 0.0;
  for (py::ssize_t i = 0; i <= nt; ++i) {
    scale += upper_raw[static_cast<std::size_t>(i)] + lower_raw[static_cast<std::size_t>(i)];
  }
  scale *= dtbase;

  if (scale > 0.0 && std::isfinite(scale)) {
    for (py::ssize_t i = 0; i <= nt; ++i) {
      upper(i) = upper_raw[static_cast<std::size_t>(i)] / scale;
      lower(i) = lower_raw[static_cast<std::size_t>(i)] / scale;
    }
  } else {
    for (py::ssize_t i = 0; i <= nt; ++i) {
      upper(i) = 0.0;
      lower(i) = 0.0;
    }
  }

  return py::make_tuple(upper_pdf, lower_pdf);
}

py::tuple simulate_scalar_process(
    py::array_t<double, py::array::c_style | py::array::forcecast> drift_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> bound_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> noise_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> start_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> nondecision_array,
    int n_trials,
    double dt,
    int seed,
    double max_time) {
  auto drift = drift_array.unchecked<2>();
  auto bound = bound_array.unchecked<2>();
  auto noise = noise_array.unchecked<2>();
  auto start = start_array.unchecked<1>();
  auto nondecision = nondecision_array.unchecked<1>();

  require_2d(drift, "drift");
  require_2d(bound, "bound");
  require_2d(noise, "noise");
  require_1d(start, "start");
  require_1d(nondecision, "nondecision");

  const py::ssize_t n_conditions = drift.shape(0);
  const py::ssize_t n_times = drift.shape(1);
  if (n_conditions <= 0 || n_times <= 0) {
    throw std::invalid_argument("drift must include at least one condition and one time point.");
  }
  if (bound.shape(0) != n_conditions || bound.shape(1) != n_times || noise.shape(0) != n_conditions || noise.shape(1) != n_times) {
    throw std::invalid_argument("drift, bound, and noise arrays must have matching shapes.");
  }
  if (start.shape(0) != n_conditions || nondecision.shape(0) != n_conditions) {
    throw std::invalid_argument("start and nondecision arrays must match the number of conditions.");
  }
  if (n_trials <= 0 || !(dt > 0.0) || !(max_time > 0.0)) {
    throw std::invalid_argument("n_trials, dt, and max_time must be positive.");
  }

  py::array_t<int> trial_array(n_trials);
  py::array_t<double> accuracy_array(n_trials);
  py::array_t<double> rt_array(n_trials);
  py::array_t<int> condition_array(n_trials);
  auto trial = trial_array.mutable_unchecked<1>();
  auto accuracy = accuracy_array.mutable_unchecked<1>();
  auto rt = rt_array.mutable_unchecked<1>();
  auto condition = condition_array.mutable_unchecked<1>();

  std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(seed));
  std::normal_distribution<double> normal(0.0, 1.0);
  const double sqrt_dt = std::sqrt(dt);

  for (int row = 0; row < n_trials; ++row) {
    const py::ssize_t cond = row % n_conditions;
    double evidence = start(cond);
    double time = 0.0;
    bool finished = false;

    trial(row) = row + 1;
    condition(row) = static_cast<int>(cond);

    while (time < max_time) {
      py::ssize_t time_index = static_cast<py::ssize_t>(std::floor(time / dt));
      if (time_index < 0) {
        time_index = 0;
      }
      if (time_index >= n_times) {
        time_index = n_times - 1;
      }
      const double boundary = std::abs(bound(cond, time_index));
      if (!(boundary > 0.0) || !std::isfinite(boundary)) {
        throw std::runtime_error("invalid boundary value.");
      }
      if (evidence >= boundary || evidence <= -boundary) {
        accuracy(row) = evidence >= boundary ? 1.0 : 0.0;
        rt(row) = time + nondecision(cond);
        finished = true;
        break;
      }
      const double step_noise = std::abs(noise(cond, time_index)) * sqrt_dt * normal(rng);
      evidence += drift(cond, time_index) * dt + step_noise;
      time += dt;
    }

    if (!finished) {
      accuracy(row) = evidence >= 0.0 ? 1.0 : 0.0;
      rt(row) = max_time + nondecision(cond);
    }
  }

  return py::make_tuple(trial_array, accuracy_array, rt_array, condition_array);
}

PYBIND11_MODULE(_cpp, module) {
  module.doc() = "Compiled numerical kernels for FlexDDM.";
  module.def("solve_transition", &solve_transition, py::arg("grid"), py::arg("edges"), py::arg("bounds"), py::arg("drift"), py::arg("noise"), py::arg("start"), py::arg("dt"));
  module.def("solve_kfe", &solve_kfe, py::arg("initial"), py::arg("x_vec"), py::arg("bounds"), py::arg("drift"), py::arg("bound_derivative"), py::arg("sigma"), py::arg("dt"), py::arg("dx"));
  module.def("simulate_scalar_process", &simulate_scalar_process, py::arg("drift"), py::arg("bound"), py::arg("noise"), py::arg("start"), py::arg("nondecision"), py::arg("n_trials"), py::arg("dt"), py::arg("seed"), py::arg("max_time") = 10.0);
}
