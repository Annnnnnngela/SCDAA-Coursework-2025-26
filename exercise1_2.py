import numpy as np
import torch
import matplotlib.pyplot as plt

from exercise1_1 import LQRRiccatiSolver


def simulate_cost_explicit(solver, x0, t0, N, n_paths, seed=12345):
    """
    Simulate the closed-loop LQR system under the optimal control using
    an explicit Euler scheme.

    The function returns the Monte Carlo estimate of the cost starting
    from (t0, x0).
    """
    rng = np.random.default_rng(seed)

    dt = (solver.T - t0) / N
    sqrt_dt = np.sqrt(dt)

    x0 = np.asarray(x0, dtype=float).reshape(2)
    X = np.tile(x0, (n_paths, 1))

    running_cost = np.zeros(n_paths)
    time_grid = np.linspace(t0, solver.T, N + 1)

    for n in range(N):
        tn = time_grid[n]

        # Interpolate S(t_n)
        S_tn = solver._interp_S(np.array([tn]))[0]

        # Optimal Markov control a(t_n, X_n) = -D^{-1} M^T S(t_n) X_n
        A = -solver.D_inv @ solver.M.T @ S_tn
        alpha = (A @ X.T).T

        # Running cost: x^T C x + a^T D a
        state_cost = np.einsum("bi,ij,bj->b", X, solver.C, X)
        control_cost = np.einsum("bi,ij,bj->b", alpha, solver.D, alpha)
        running_cost += (state_cost + control_cost) * dt

        # Brownian increment
        dW = rng.normal(size=(n_paths, 2)) * sqrt_dt
        diffusion = dW @ solver.sigma.T

        # Euler step
        drift = (X @ solver.H.T) + (alpha @ solver.M.T)
        X = X + drift * dt + diffusion

    terminal_cost = np.einsum("bi,ij,bj->b", X, solver.R, X)
    total_cost = running_cost + terminal_cost

    return total_cost.mean()


def exact_value_from_solver(solver, t0, x0):
    """
    Evaluate the benchmark value function from Exercise 1.1 at one point.
    """
    t_tensor = torch.tensor([t0], dtype=torch.float32)
    x_tensor = torch.tensor([x0], dtype=torch.float32)
    return solver.value_function(t_tensor, x_tensor).item()


def convergence_in_time_steps(solver, x0, t0=0.0, big_mc=100000):
    """
    Fix a large Monte Carlo sample size and vary the number of time steps.
    """
    N_values = [1, 10, 50, 100, 500, 1000, 5000]
    v_true = exact_value_from_solver(solver, t0, x0)

    estimates = []
    errors = []

    for N in N_values:
        est = simulate_cost_explicit(
            solver=solver,
            x0=x0,
            t0=t0,
            N=N,
            n_paths=big_mc,
            seed=12345
        )
        err = abs(est - v_true)

        estimates.append(est)
        errors.append(err)

        print(f"N = {N:5d} | estimate = {est:.8f} | true = {v_true:.8f} | error = {err:.8e}")

    return N_values, estimates, errors


def convergence_in_mc_samples(solver, x0, t0=0.0, big_N=5000):
    """
    Fix a large number of time steps and vary the Monte Carlo sample size.
    """
    M_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    v_true = exact_value_from_solver(solver, t0, x0)

    estimates = []
    errors = []

    for M in M_values:
        est = simulate_cost_explicit(
            solver=solver,
            x0=x0,
            t0=t0,
            N=big_N,
            n_paths=M,
            seed=12345
        )
        err = abs(est - v_true)

        estimates.append(est)
        errors.append(err)

        print(f"M = {M:6d} | estimate = {est:.8f} | true = {v_true:.8f} | error = {err:.8e}")

    return M_values, estimates, errors


def fit_loglog_slope(x, y):
    """
    Fit a straight line in log-log space and return the slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = y > 0
    coeffs = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)

    slope = coeffs[0]
    intercept = coeffs[1]
    return slope, intercept


def plot_loglog(x, y, xlabel, ylabel, title, filename):
    """
    Plot log-log error curves and save the figure.
    """
    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Use the same coefficient setup as in Exercise 1.1
    H = np.array([[[0.1, 0.0],
                  [0.0, 0.2]]])
    M = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    R = np.eye(2)
    sigma = 0.3 * np.eye(2)
    T = 1.0

    solver = LQRRiccatiSolver(H, M, C, D, R, sigma, T)

    # A fine grid is used so that Riccati/interpolation error is small enough
    riccati_grid = np.linspace(0.0, T, 5001)
    solver.solve_riccati(riccati_grid)

    # Initial point for the convergence study
    x0 = np.array([1.0, -1.0])
    t0 = 0.0

    print("\n--- Convergence with respect to time steps N ---")
    N_values, est_N, err_N = convergence_in_time_steps(
        solver=solver,
        x0=x0,
        t0=t0,
        big_mc=100000
    )

    slope_N, _ = fit_loglog_slope(N_values, err_N)
    print(f"Estimated slope with respect to N: {slope_N:.4f}")

    plot_loglog(
        N_values,
        err_N,
        xlabel="Number of time steps N",
        ylabel="Absolute error",
        title="Exercise 1.2: error vs time steps",
        filename="ex1_2_error_vs_timesteps.png"
    )

    print("\n--- Convergence with respect to Monte Carlo samples M ---")
    M_values, est_M, err_M = convergence_in_mc_samples(
        solver=solver,
        x0=x0,
        t0=t0,
        big_N=5000
    )

    slope_M, _ = fit_loglog_slope(M_values, err_M)
    print(f"Estimated slope with respect to M: {slope_M:.4f}")

    plot_loglog(
        M_values,
        err_M,
        xlabel="Number of Monte Carlo samples M",
        ylabel="Absolute error",
        title="Exercise 1.2: error vs Monte Carlo samples",
        filename="ex1_2_error_vs_mc_samples.png"
    )