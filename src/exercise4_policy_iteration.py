import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from exercise1_1 import LQRRiccatiSolver
from exercise2_supervised_learning_lqr import NetDGM


# ============================================================
# 1. Device and problem setup
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_problem_matrices():
    H = torch.tensor([[0.1, 0.0],
                      [0.0, 0.2]], dtype=torch.float32, device=DEVICE)

    M = torch.eye(2, dtype=torch.float32, device=DEVICE)
    C = torch.eye(2, dtype=torch.float32, device=DEVICE)
    D = torch.eye(2, dtype=torch.float32, device=DEVICE)
    R = torch.eye(2, dtype=torch.float32, device=DEVICE)
    sigma = 0.3 * torch.eye(2, dtype=torch.float32, device=DEVICE)

    return H, M, C, D, R, sigma


# ============================================================
# 2. Sampling
# ============================================================
def sample_interior(batch_size, T, x_low, x_high, device):
    """
    Sample interior points (t, x) with
    t ~ Uniform[0, T], x ~ Uniform([x_low, x_high]^2).
    """
    t = torch.rand(batch_size, 1, device=device) * T
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return t, x


def sample_terminal(batch_size, T, x_low, x_high, device):
    """
    Sample terminal points (T, x).
    """
    t = torch.full((batch_size, 1), T, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return t, x


def terminal_value(x, R):
    """
    Terminal condition x^T R x.
    """
    Rx = x @ R.T
    return torch.sum(x * Rx, dim=1, keepdim=True)


# ============================================================
# 3. Policy network
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, x):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


# ============================================================
# 4. PDE loss with current policy
# ============================================================
def pde_loss_with_policy(
    value_net,
    policy_net,
    H, M, C, D, R, sigma,
    batch_interior,
    batch_terminal,
    T,
    x_low,
    x_high,
    lambda_terminal=1.0
):
    """
    Policy evaluation step:
    solve the linear PDE associated with the current policy a(t,x).
    """

    # ---------- Interior points ----------
    t, x = sample_interior(batch_interior, T, x_low, x_high, DEVICE)

    inp = torch.cat([t, x], dim=1)
    u = value_net(inp)

    # First derivatives
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=[t, x],
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )

    u_t = grad_u[0]
    u_x = grad_u[1]

    # Hessian entries
    u_x1 = u_x[:, 0:1]
    u_x2 = u_x[:, 1:2]

    grad_u_x1 = torch.autograd.grad(
        outputs=u_x1,
        inputs=x,
        grad_outputs=torch.ones_like(u_x1),
        create_graph=True
    )[0]

    grad_u_x2 = torch.autograd.grad(
        outputs=u_x2,
        inputs=x,
        grad_outputs=torch.ones_like(u_x2),
        create_graph=True
    )[0]

    u_xx_11 = grad_u_x1[:, 0:1]
    u_xx_12 = grad_u_x1[:, 1:2]
    u_xx_21 = grad_u_x2[:, 0:1]
    u_xx_22 = grad_u_x2[:, 1:2]

    # Hessian matrix
    hessian_u = torch.stack([
        torch.cat([u_xx_11, u_xx_12], dim=1),
        torch.cat([u_xx_21, u_xx_22], dim=1)
    ], dim=1)

    # Current policy
    a = policy_net(t, x)

    # PDE terms
    Hx = x @ H.T
    Ma = a @ M.T

    gradHx = torch.sum(u_x * Hx, dim=1, keepdim=True)
    gradMa = torch.sum(u_x * Ma, dim=1, keepdim=True)

    Cx = x @ C.T
    xCx = torch.sum(x * Cx, dim=1, keepdim=True)

    Da = a @ D.T
    aDa = torch.sum(a * Da, dim=1, keepdim=True)

    sigma_sigma_T = sigma @ sigma.T
    diff_term = 0.5 * torch.einsum("ij,bij->b", sigma_sigma_T, hessian_u).unsqueeze(1)

    residual = u_t + diff_term + gradHx + gradMa + xCx + aDa
    interior_loss = torch.mean(residual ** 2)

    # ---------- Terminal points ----------
    t_T, x_T = sample_terminal(batch_terminal, T, x_low, x_high, DEVICE)

    inp_T = torch.cat([t_T, x_T], dim=1)
    u_T = value_net(inp_T)

    target_T = terminal_value(x_T, R)
    terminal_loss = torch.mean((u_T - target_T) ** 2)

    total_loss = interior_loss + lambda_terminal * terminal_loss

    return total_loss, interior_loss, terminal_loss


# ============================================================
# 5. Hamiltonian loss for policy improvement
# ============================================================
def hamiltonian_loss(
    value_net,
    policy_net,
    H, M, C, D,
    batch_size,
    T,
    x_low,
    x_high
):
    """
    Policy improvement step:
    minimize the Hamiltonian with value network fixed.
    """
    t, x = sample_interior(batch_size, T, x_low, x_high, DEVICE)

    inp = torch.cat([t, x], dim=1)
    v = value_net(inp)

    grad_v = torch.autograd.grad(
        outputs=v,
        inputs=x,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    a = policy_net(t, x)

    Hx = x @ H.T
    Ma = a @ M.T

    gradHx = torch.sum(grad_v * Hx, dim=1, keepdim=True)
    gradMa = torch.sum(grad_v * Ma, dim=1, keepdim=True)

    Cx = x @ C.T
    xCx = torch.sum(x * Cx, dim=1, keepdim=True)

    Da = a @ D.T
    aDa = torch.sum(a * Da, dim=1, keepdim=True)

    ham = gradHx + gradMa + xCx + aDa
    return torch.mean(ham)


# ============================================================
# 6. Training routines
# ============================================================
def train_value_net(
    value_net,
    policy_net,
    H, M, C, D, R, sigma,
    epochs,
    lr,
    batch_interior,
    batch_terminal,
    T,
    x_low,
    x_high,
    lambda_terminal=1.0,
    print_every=200
):
    optimizer = optim.Adam(value_net.parameters(), lr=lr)

    history_total = []
    history_interior = []
    history_terminal = []

    value_net.train()
    policy_net.eval()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        total_loss, interior_loss, terminal_loss = pde_loss_with_policy(
            value_net=value_net,
            policy_net=policy_net,
            H=H, M=M, C=C, D=D, R=R, sigma=sigma,
            batch_interior=batch_interior,
            batch_terminal=batch_terminal,
            T=T,
            x_low=x_low,
            x_high=x_high,
            lambda_terminal=lambda_terminal
        )

        total_loss.backward()
        optimizer.step()

        history_total.append(total_loss.item())
        history_interior.append(interior_loss.item())
        history_terminal.append(terminal_loss.item())

        if epoch % print_every == 0:
            print(
                f"[Value Net] Epoch {epoch}/{epochs} | "
                f"Total: {total_loss.item():.6e} | "
                f"Interior: {interior_loss.item():.6e} | "
                f"Terminal: {terminal_loss.item():.6e}"
            )

    return value_net, history_total, history_interior, history_terminal


def train_policy_net(
    value_net,
    policy_net,
    H, M, C, D,
    epochs,
    lr,
    batch_size,
    T,
    x_low,
    x_high,
    print_every=200
):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    history_policy = []

    value_net.eval()
    policy_net.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss = hamiltonian_loss(
            value_net=value_net,
            policy_net=policy_net,
            H=H, M=M, C=C, D=D,
            batch_size=batch_size,
            T=T,
            x_low=x_low,
            x_high=x_high
        )

        loss.backward()
        optimizer.step()

        history_policy.append(loss.item())

        if epoch % print_every == 0:
            print(f"[Policy Net] Epoch {epoch}/{epochs} | Hamiltonian: {loss.item():.6e}")

    return policy_net, history_policy


# ============================================================
# 7. Benchmark comparison with Exercise 1.1
# ============================================================
def build_lqr_solver(H, M, C, D, R, sigma, T):
    solver = LQRRiccatiSolver(
        H.detach().cpu().numpy(),
        M.detach().cpu().numpy(),
        C.detach().cpu().numpy(),
        D.detach().cpu().numpy(),
        R.detach().cpu().numpy(),
        sigma.detach().cpu().numpy(),
        T
    )
    solver.solve_riccati(np.linspace(0.0, T, 5001))
    return solver


@torch.no_grad()
def evaluate_on_test_grid(
    value_net,
    policy_net,
    lqr_solver,
    n_test,
    T,
    x_low,
    x_high
):
    value_net.eval()
    policy_net.eval()

    t_test = torch.rand(n_test, 1, device=DEVICE) * T
    x_test = x_low + (x_high - x_low) * torch.rand(n_test, 2, device=DEVICE)

    v_pred = value_net(torch.cat([t_test, x_test], dim=1))
    a_pred = policy_net(t_test, x_test)

    t_test_lqr = t_test.squeeze(1)
    x_test_lqr = x_test

    v_true = lqr_solver.value_function(t_test_lqr, x_test_lqr).to(DEVICE)
    a_true = lqr_solver.markov_control(t_test_lqr, x_test_lqr).to(DEVICE)

    if v_true.dim() == 1:
        v_true = v_true.unsqueeze(1)
    if a_true.dim() == 3:
        a_true = a_true.squeeze(1)

    value_error = torch.mean(torch.abs(v_pred - v_true)).item()
    policy_error = torch.mean(torch.norm(a_pred - a_true, dim=1)).item()

    return value_error, policy_error


# ============================================================
# 8. Plotting
# ============================================================
def plot_value_losses(total_hist, interior_hist, terminal_hist, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(total_hist, label="Total loss")
    plt.plot(interior_hist, label="Interior loss")
    plt.plot(terminal_hist, label="Terminal loss")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Exercise 4.1: Value Network Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_policy_losses(policy_hist, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(policy_hist, label="Hamiltonian loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Exercise 4.1: Policy Network Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_outer_errors(errors, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Policy iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# ============================================================
# 9. Main
# ============================================================
def main():
    print("Using device:", DEVICE)

    # ---------- Problem setup ----------
    T = 1.0
    x_low, x_high = -3.0, 3.0

    H, M, C, D, R, sigma = build_problem_matrices()

    # ---------- Hyperparameters ----------
    batch_interior = 128
    batch_terminal = 128
    batch_hamiltonian = 128

    value_hidden_dim = 100
    policy_hidden_dim = 100

    n_policy_iter = 3
    value_epochs_per_iter = 400
    policy_epochs_per_iter = 200

    lr_value = 1e-3
    lr_policy = 1e-3
    lambda_terminal = 1.0

    n_test = 1000

    # ---------- Benchmark solver from Exercise 1.1 ----------
    solver = build_lqr_solver(H, M, C, D, R, sigma, T)

    # ---------- Networks ----------
    value_net = NetDGM(input_dim=3, hidden_dim=value_hidden_dim, num_layers=3, output_dim=1).to(DEVICE)
    policy_net = PolicyNet(input_dim=3, hidden_dim=policy_hidden_dim, output_dim=2).to(DEVICE)

    print("\nInitial value network:")
    print(value_net)
    print("\nInitial policy network:")
    print(policy_net)

    # ---------- Histories ----------
    value_loss_history_all = []
    value_interior_history_all = []
    value_terminal_history_all = []

    policy_loss_history_all = []

    outer_iter_value_error = []
    outer_iter_policy_error = []

    # ---------- Policy iteration loop ----------
    print("\nStarting policy iteration...")

    for k in range(1, n_policy_iter + 1):
        print("\n" + "=" * 60)
        print(f"Policy Iteration Step {k}/{n_policy_iter}")
        print("=" * 60)

        # Step 1: policy evaluation
        value_net, hist_total, hist_interior, hist_terminal = train_value_net(
            value_net=value_net,
            policy_net=policy_net,
            H=H, M=M, C=C, D=D, R=R, sigma=sigma,
            epochs=value_epochs_per_iter,
            lr=lr_value,
            batch_interior=batch_interior,
            batch_terminal=batch_terminal,
            T=T,
            x_low=x_low,
            x_high=x_high,
            lambda_terminal=lambda_terminal,
            print_every=200
        )

        value_loss_history_all.extend(hist_total)
        value_interior_history_all.extend(hist_interior)
        value_terminal_history_all.extend(hist_terminal)

        # Step 2: policy improvement
        policy_net, hist_policy = train_policy_net(
            value_net=value_net,
            policy_net=policy_net,
            H=H, M=M, C=C, D=D,
            epochs=policy_epochs_per_iter,
            lr=lr_policy,
            batch_size=batch_hamiltonian,
            T=T,
            x_low=x_low,
            x_high=x_high,
            print_every=200
        )

        policy_loss_history_all.extend(hist_policy)

        # Step 3: benchmark comparison
        value_err, policy_err = evaluate_on_test_grid(
            value_net=value_net,
            policy_net=policy_net,
            lqr_solver=solver,
            n_test=n_test,
            T=T,
            x_low=x_low,
            x_high=x_high
        )

        outer_iter_value_error.append(value_err)
        outer_iter_policy_error.append(policy_err)

        print(
            f"[Outer Iter {k}] "
            f"Mean value error: {value_err:.6e} | "
            f"Mean policy error: {policy_err:.6e}"
        )

    # ---------- Save models ----------
    torch.save(value_net.state_dict(), "exercise4_value_net.pt")
    torch.save(policy_net.state_dict(), "exercise4_policy_net.pt")

    # ---------- Plots ----------
    plot_value_losses(
        value_loss_history_all,
        value_interior_history_all,
        value_terminal_history_all,
        "exercise4_value_training_loss.png"
    )

    plot_policy_losses(
        policy_loss_history_all,
        "exercise4_policy_training_loss.png"
    )

    plot_outer_errors(
        outer_iter_value_error,
        ylabel="Mean absolute value error",
        title="Exercise 4.1: Value Error vs Policy Iteration",
        filename="exercise4_value_error_vs_iteration.png"
    )

    plot_outer_errors(
        outer_iter_policy_error,
        ylabel="Mean policy error",
        title="Exercise 4.1: Policy Error vs Policy Iteration",
        filename="exercise4_policy_error_vs_iteration.png"
    )

    # ---------- Final summary ----------
    print("\n" + "=" * 60)
    print("Final diagnostics")
    print("=" * 60)

    if len(outer_iter_value_error) > 0:
        print(f"Final mean value error:  {outer_iter_value_error[-1]:.6e}")

    if len(outer_iter_policy_error) > 0:
        print(f"Final mean policy error: {outer_iter_policy_error[-1]:.6e}")

    print(f"Total value training steps:  {len(value_loss_history_all)}")
    print(f"Total policy training steps: {len(policy_loss_history_all)}")

    print("\nSaved files:")
    print("  - exercise4_value_net.pt")
    print("  - exercise4_policy_net.pt")
    print("  - exercise4_value_training_loss.png")
    print("  - exercise4_policy_training_loss.png")
    print("  - exercise4_value_error_vs_iteration.png")
    print("  - exercise4_policy_error_vs_iteration.png")


if __name__ == "__main__":
    main()