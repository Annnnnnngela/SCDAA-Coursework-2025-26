import numpy as np
import torch
from scipy.integrate import solve_ivp


class LQRRiccatiSolver:
    """
    Solve the 2D LQR benchmark by first computing the matrix-valued Riccati ODE
    backward in time, and then using the resulting S(t) to evaluate both the
    value function and the optimal Markov control.

    The class is written so that later coursework parts can reuse the same
    benchmark solver as a source of supervised-learning labels and as a numerical
    reference when checking convergence.
    """

    def __init__(self, H, M, C, D, R, sigma, T):
        self.H = np.asarray(H, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.T = float(T)

        self.D_inv = np.linalg.inv(self.D)
        self.sigma_sigma_T = self.sigma @ self.sigma.T

        self.time_grid = None
        self.S_grid = None
        self.int_term_grid = None

    def _riccati_rhs(self, t, y):
        """
        The ODE is solved in vector form. We reshape back to a 2x2 matrix,
        apply the Riccati dynamics, and flatten again.
        """
        S = y.reshape(2, 2)
        dS = (
            S @ self.M @ self.D_inv @ self.M.T @ S
            - self.H.T @ S
            - S @ self.H
            - self.C
            )
        return dS.reshape(-1)

    def solve_riccati(self, time_grid):
        """
        Solve the Riccati ODE backward from T to 0 and store the result on the
        requested grid. The grid may be given as a numpy array or torch tensor.
        """
        if isinstance(time_grid, torch.Tensor):
            time_grid = time_grid.detach().cpu().numpy()

        time_grid = np.asarray(time_grid, dtype=float)
        time_grid = np.sort(time_grid)

        sol = solve_ivp(
            fun=self._riccati_rhs,
            t_span=(self.T, time_grid[0]),
            y0=self.R.reshape(-1),
            t_eval=time_grid[::-1],   # backward evaluation
            rtol=1e-9,
            atol=1e-11,
            method="RK45"
        )

        # Reorder back to increasing time
        S_backward = sol.y.T.reshape(-1, 2, 2)
        self.time_grid = sol.t[::-1]
        self.S_grid = S_backward[::-1]

        # Precompute integral term: \int_t^T tr(sigma sigma^T S(r)) dr
        g = np.array([
            np.trace(self.sigma_sigma_T @ S)
            for S in self.S_grid
        ])

        self.int_term_grid = np.zeros_like(g)
        # Compute backward cumulative trapezoid
        for i in range(len(self.time_grid) - 2, -1, -1):
            dt = self.time_grid[i + 1] - self.time_grid[i]
            self.int_term_grid[i] = (
                self.int_term_grid[i + 1]
                + 0.5 * dt * (g[i] + g[i + 1])
            )

    def _interp_S(self, t):
        """
        Interpolate each entry of S(t) separately so that we can evaluate the
        Riccati solution at arbitrary batch times.
        """
        t = np.asarray(t, dtype=float)
        out = np.zeros((len(t), 2, 2))
        for i in range(2):
            for j in range(2):
                out[:, i, j] = np.interp(
                    t, self.time_grid, self.S_grid[:, i, j]
                )
        return out

    def _interp_integral_term(self, t):
        """
        Interpolate the scalar correction term in the value function.
        """
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.time_grid, self.int_term_grid)

    def value_function(self, t_tensor, x_tensor):
        """
        Return a tensor of shape (batch_size, 1) containing v(t, x) for each
        batched pair (t, x).
        """
        t_np = t_tensor.detach().cpu().numpy().reshape(-1)
        x_np = x_tensor.detach().cpu().numpy().reshape(-1, 2)

        S_eval = self._interp_S(t_np)
        int_eval = self._interp_integral_term(t_np)

        quad = np.einsum("bi,bij,bj->b", x_np, S_eval, x_np)
        values = quad + int_eval

        return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def markov_control(self, t_tensor, x_tensor):
        """
        Return a tensor of shape (batch_size, 2) containing the optimal Markov
        control a(t, x) = -D^{-1} M^T S(t) x.
        """
        t_np = t_tensor.detach().cpu().numpy().reshape(-1)
        x_np = x_tensor.detach().cpu().numpy().reshape(-1, 2)

        S_eval = self._interp_S(t_np)
        controls = np.zeros((len(t_np), 2))

        for b in range(len(t_np)):
            controls[b] = -self.D_inv @ self.M.T @ S_eval[b] @ x_np[b]

        return torch.tensor(controls, dtype=torch.float32)
    

if __name__ == "__main__":
    import numpy as np
    import torch

    H = np.array([[0.1, 0.0],
                  [0.0, 0.2]])
    M = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    R = np.eye(2)
    sigma = 0.3 * np.eye(2)
    T = 1.0

    solver = LQRRiccatiSolver(H, M, C, D, R, sigma, T)

    grid = np.linspace(0.0, T, 1001)
    solver.solve_riccati(grid)

    t = torch.tensor([0.0, 0.5], dtype=torch.float32)
    x = torch.tensor([[[1.0, 0.0]],
                      [[0.5, -0.2]]], dtype=torch.float32)

    print("value:", solver.value_function(t, x))
    print("control:", solver.markov_control(t, x))