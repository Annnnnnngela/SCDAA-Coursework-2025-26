import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from exercise1_1 import LQRRiccatiSolver


# ============================================================
# 0. Reproducibility
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Problem setup
#    Same coefficient choice as exercise1_1.py
# ============================================================
@dataclass
class ProblemConfig:
    T: float = 1.0
    x_low: float = -3.0
    x_high: float = 3.0


def build_lqr_solver(T: float = 1.0) -> LQRRiccatiSolver:
    H = np.array([[0.1, 0.0],
                  [0.0, 0.2]])
    M = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    R = np.eye(2)
    sigma = 0.3 * np.eye(2)

    solver = LQRRiccatiSolver(H, M, C, D, R, sigma, T)

    # Fine grid so interpolation error is small
    riccati_grid = np.linspace(0.0, T, 5001)
    solver.solve_riccati(riccati_grid)

    return solver


# ============================================================
# 3. Sample training inputs
# ============================================================
def sample_inputs(batch_size: int, config: ProblemConfig):
    """
    Sample t uniformly on [0, T]
    and x uniformly on [-3, 3] x [-3, 3].
    """
    t = torch.rand(batch_size, 1, device=DEVICE) * config.T
    x = config.x_low + (config.x_high - config.x_low) * torch.rand(batch_size, 2, device=DEVICE)
    inputs = torch.cat([t, x], dim=1)
    return t, x, inputs


# ============================================================
# 4. DGM-like network for value function
#    Input: (t, x1, x2) -> output: scalar value
# ============================================================
class DGMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Uz = nn.Linear(input_dim, hidden_dim)
        self.Wz = nn.Linear(hidden_dim, hidden_dim)

        self.Ug = nn.Linear(input_dim, hidden_dim)
        self.Wg = nn.Linear(hidden_dim, hidden_dim)

        self.Ur = nn.Linear(input_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)

        self.Uh = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)

        self.activation = torch.tanh

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.Uz(x) + self.Wz(s))
        g = self.activation(self.Ug(x) + self.Wg(s))
        r = self.activation(self.Ur(x) + self.Wr(s))
        h = self.activation(self.Uh(x) + self.Wh(s * r))
        s_new = (1.0 - g) * h + z * s
        return s_new


class NetDGM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 100, num_layers: int = 3, output_dim: int = 1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.dgm_layers = nn.ModuleList([DGMLayer(input_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.activation(self.input_layer(x))
        for layer in self.dgm_layers:
            s = layer(x, s)
        out = self.output_layer(s)
        return out


# ============================================================
# 5. FFN for control
#    Input: (t, x1, x2) -> output: 2D control
# ============================================================
class FFN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 100, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 6. Training
# ============================================================
def train_value_network(
    model: nn.Module,
    solver: LQRRiccatiSolver,
    config: ProblemConfig,
    num_steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    print_every: int = 200,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    model.train()
    for step in range(1, num_steps + 1):
        t, x, inputs = sample_inputs(batch_size, config)

        # Ground truth from Exercise 1.1 solver
        target = solver.value_function(t, x).to(DEVICE)

        pred = model(inputs)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % print_every == 0:
            print(f"[Value] Step {step:5d} | Loss = {loss.item():.6e}")

    return loss_history


def train_control_network(
    model: nn.Module,
    solver: LQRRiccatiSolver,
    config: ProblemConfig,
    num_steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    print_every: int = 200,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    model.train()
    for step in range(1, num_steps + 1):
        t, x, inputs = sample_inputs(batch_size, config)

        # Ground truth from Exercise 1.1 solver
        target = solver.markov_control(t, x).to(DEVICE)

        pred = model(inputs)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % print_every == 0:
            print(f"[Control] Step {step:5d} | Loss = {loss.item():.6e}")

    return loss_history


# ============================================================
# 7. Evaluation
# ============================================================
@torch.no_grad()
def evaluate_value_network(model: nn.Module, solver: LQRRiccatiSolver, config: ProblemConfig, n_test: int = 5000):
    model.eval()
    t, x, inputs = sample_inputs(n_test, config)

    pred = model(inputs)
    target = solver.value_function(t, x).to(DEVICE)

    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    return mse, mae


@torch.no_grad()
def evaluate_control_network(model: nn.Module, solver: LQRRiccatiSolver, config: ProblemConfig, n_test: int = 5000):
    model.eval()
    t, x, inputs = sample_inputs(n_test, config)

    pred = model(inputs)
    target = solver.markov_control(t, x).to(DEVICE)

    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    return mse, mae


# ============================================================
# 8. Plotting
# ============================================================
def plot_loss(loss_history, title: str, filename: str):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# 9. prediction vs truth scatter(It's not required for coursework, but it shows more information.
# ============================================================
@torch.no_grad()
def make_value_scatter_plot(model, solver, config, n_test=1000, filename="value_scatter.png"):
    model.eval()
    t, x, inputs = sample_inputs(n_test, config)

    pred = model(inputs).cpu().numpy().reshape(-1)
    true = solver.value_function(t, x).cpu().numpy().reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=10, alpha=0.5)
    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("Value network: prediction vs truth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


@torch.no_grad()
def make_control_scatter_plot(model, solver, config, n_test=1000, filename_prefix="control_scatter"):
    model.eval()
    t, x, inputs = sample_inputs(n_test, config)

    pred = model(inputs).cpu().numpy()
    true = solver.markov_control(t, x).cpu().numpy()

    for k in range(2):
        plt.figure(figsize=(6, 6))
        plt.scatter(true[:, k], pred[:, k], s=10, alpha=0.5)
        mn = min(true[:, k].min(), pred[:, k].min())
        mx = max(true[:, k].max(), pred[:, k].max())
        plt.plot([mn, mx], [mn, mx], "--")
        plt.xlabel(f"True control component {k+1}")
        plt.ylabel(f"Predicted control component {k+1}")
        plt.title(f"Control network component {k+1}: prediction vs truth")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{k+1}.png", dpi=300)
        plt.close()


# ============================================================
# 10. Main
# ============================================================
def main():
    set_seed(42)

    config = ProblemConfig(T=1.0, x_low=-3.0, x_high=3.0)
    solver = build_lqr_solver(T=config.T)

    # ----------------------------
    # Train value network
    # ----------------------------
    value_model = NetDGM(input_dim=3, hidden_dim=100, num_layers=3, output_dim=1).to(DEVICE)
    value_losses = train_value_network(
        model=value_model,
        solver=solver,
        config=config,
        num_steps=3000,
        batch_size=512,
        lr=1e-3,
        print_every=200,
    )
    plot_loss(value_losses, "Training Loss - Value Network (LQR labels)", "value_loss_lqr.png")
    value_mse, value_mae = evaluate_value_network(value_model, solver, config)
    print("\nValue network evaluation:")
    print(f"Test MSE = {value_mse:.6e}")
    print(f"Test MAE = {value_mae:.6e}")
    make_value_scatter_plot(value_model, solver, config, n_test=1000, filename="value_scatter.png")

    # ----------------------------
    # Train control network
    # ----------------------------
    control_model = FFN(input_dim=3, hidden_dim=100, output_dim=2).to(DEVICE)
    control_losses = train_control_network(
        model=control_model,
        solver=solver,
        config=config,
        num_steps=3000,
        batch_size=512,
        lr=1e-3,
        print_every=200,
    )
    plot_loss(control_losses, "Training Loss - Control Network (LQR labels)", "control_loss_lqr.png")
    control_mse, control_mae = evaluate_control_network(control_model, solver, config)
    print("\nControl network evaluation:")
    print(f"Test MSE = {control_mse:.6e}")
    print(f"Test MAE = {control_mae:.6e}")
    make_control_scatter_plot(control_model, solver, config, n_test=1000, filename_prefix="control_scatter")

    # Save model weights
    torch.save(value_model.state_dict(), "value_model_lqr.pt")
    torch.save(control_model.state_dict(), "control_model_lqr.pt")

    print("\nDone.")
    print("Saved files:")
    print("  - value_loss_lqr.png")
    print("  - control_loss_lqr.png")
    print("  - value_scatter.png")
    print("  - control_scatter_1.png")
    print("  - control_scatter_2.png")
    print("  - value_model_lqr.pt")
    print("  - control_model_lqr.pt")


if __name__ == "__main__":
    main()