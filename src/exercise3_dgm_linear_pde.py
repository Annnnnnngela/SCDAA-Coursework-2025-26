import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass

from exercise2_supervised_learning_lqr import NetDGM


# ============================================================
# 1. Problem configuration
# ============================================================
@dataclass
class ProblemConfig:
    T: float = 1.0
    x_low: float = -3.0
    x_high: float = 3.0


def sample_interior_points(batch_size, config, device):
    """
    Sample interior points (t, x) with
    t ~ Uniform[0, T], x ~ Uniform([x_low, x_high]^2).
    """
    t = torch.rand(batch_size, 1, device=device) * config.T
    x = config.x_low + (config.x_high - config.x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def sample_terminal_points(batch_size, config, device):
    """
    Sample terminal points (T, x) with x ~ Uniform([x_low, x_high]^2).
    """
    t = torch.full((batch_size, 1), config.T, device=device)
    x = config.x_low + (config.x_high - config.x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


# ============================================================
# 2. Linear PDE model under constant control alpha = (1,1)^T
# ============================================================
class LinearPDE2D:
    def __init__(self, model, H, M, C, D, R, sigma, config, device):
        self.model = model
        self.H = H.to(device).float()
        self.M = M.to(device).float()
        self.C = C.to(device).float()
        self.D = D.to(device).float()
        self.R = R.to(device).float()
        self.sigma = sigma.to(device).float()

        self.config = config
        self.T = float(config.T)
        self.device = device
        self.x_low = config.x_low
        self.x_high = config.x_high

        # Constant control alpha = (1, 1)^T
        self.alpha = torch.tensor([1.0, 1.0], device=device).float()

        # Sigma = sigma sigma^T
        self.Sigma = self.sigma @ self.sigma.T

    def terminal_target(self, x):
        """
        Terminal condition: u(T, x) = x^T R x
        """
        Rx = x @ self.R.T
        return torch.sum(x * Rx, dim=1, keepdim=True)

    def pde_residual(self, t, x):
        """
        Compute PDE residual:
        u_t + 0.5 tr((sigma sigma^T) Hessian u)
        + (grad u)^T Hx + (grad u)^T M alpha
        + x^T C x + alpha^T D alpha
        """
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)

        inp = torch.cat([t, x], dim=1)
        u = self.model(inp)

        # Time derivative
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Spatial gradient
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        ux1 = u_x[:, 0:1]
        ux2 = u_x[:, 1:2]

        # Second derivatives
        u_xx1 = torch.autograd.grad(
            outputs=ux1,
            inputs=x,
            grad_outputs=torch.ones_like(ux1),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx2 = torch.autograd.grad(
            outputs=ux2,
            inputs=x,
            grad_outputs=torch.ones_like(ux2),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx_11 = u_xx1[:, 0:1]
        u_xx_12 = u_xx1[:, 1:2]
        u_xx_21 = u_xx2[:, 0:1]
        u_xx_22 = u_xx2[:, 1:2]

        # Diffusion term: 0.5 * tr((sigma sigma^T) Hessian)
        diffusion = 0.5 * (
            self.Sigma[0, 0] * u_xx_11 +
            self.Sigma[0, 1] * u_xx_12 +
            self.Sigma[1, 0] * u_xx_21 +
            self.Sigma[1, 1] * u_xx_22
        )

        # Drift term (grad u)^T Hx
        Hx = x @ self.H.T
        drift1 = torch.sum(u_x * Hx, dim=1, keepdim=True)

        # Drift term (grad u)^T M alpha
        Malpha = (self.M @ self.alpha).view(1, 2).repeat(x.shape[0], 1)
        drift2 = torch.sum(u_x * Malpha, dim=1, keepdim=True)

        # State running cost x^T C x
        Cx = x @ self.C.T
        quad_x = torch.sum(x * Cx, dim=1, keepdim=True)

        # Constant control cost alpha^T D alpha
        alphaDa = (
            self.alpha.view(1, 2) @ self.D @ self.alpha.view(2, 1)
        ).view(1, 1).repeat(x.shape[0], 1)

        residual = u_t + diffusion + drift1 + drift2 + quad_x + alphaDa
        return residual

    def interior_loss(self, batch_size):
        t, x = sample_interior_points(batch_size, self.config, self.device)
        residual = self.pde_residual(t, x)
        return torch.mean(residual ** 2)

    def boundary_loss(self, batch_size):
        tT, x = sample_terminal_points(batch_size, self.config, self.device)
        inp = torch.cat([tT, x], dim=1)
        uT = self.model(inp)
        target = self.terminal_target(x)
        return torch.mean((uT - target) ** 2)

    def total_loss(self, batch_size, lambda_boundary=1.0):
        loss_interior = self.interior_loss(batch_size)
        loss_boundary = self.boundary_loss(batch_size)
        total = loss_interior + lambda_boundary * loss_boundary
        return total, loss_interior.detach(), loss_boundary.detach()


# ============================================================
# 3. Monte Carlo benchmark under constant control
# ============================================================
def mc_value_constant_control(
    x0, t0, T, H, M, C, D, R, sigma, alpha,
    n_steps=1000, n_mc=10000, seed=12345
):
    """
    Monte Carlo estimate of the linear PDE solution under fixed control alpha.
    """
    torch.manual_seed(seed)

    device = x0.device
    dt = (T - t0) / n_steps
    sqrt_dt = dt ** 0.5

    X = x0.repeat(n_mc, 1)
    alpha_row = alpha.view(1, 2).repeat(n_mc, 1)

    running_cost = torch.zeros(n_mc, 1, device=device)

    for _ in range(n_steps):
        state_cost = torch.sum((X @ C) * X, dim=1, keepdim=True)
        control_cost = torch.sum((alpha_row @ D) * alpha_row, dim=1, keepdim=True)
        running_cost += (state_cost + control_cost) * dt

        dW = sqrt_dt * torch.randn(n_mc, 2, device=device)
        drift = X @ H.T + alpha_row @ M.T
        X = X + drift * dt + dW @ sigma.T

    terminal_cost = torch.sum((X @ R) * X, dim=1, keepdim=True)
    total_cost = running_cost + terminal_cost

    return total_cost.mean().item()


@torch.no_grad()
def evaluate_against_mc(
    model, test_points, T, H, M, C, D, R, sigma, alpha, device,
    n_steps=1000, n_mc=5000
):
    """
    Compare the trained DGM solution against MC estimates
    at a list of fixed test points.
    """
    model.eval()

    abs_errors = []
    records = []

    for (t0, x1, x2) in test_points:
        inp = torch.tensor([[t0, x1, x2]], dtype=torch.float32, device=device)
        pred = model(inp).item()

        x0 = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
        mc_val = mc_value_constant_control(
            x0=x0,
            t0=t0,
            T=T,
            H=H,
            M=M,
            C=C,
            D=D,
            R=R,
            sigma=sigma,
            alpha=alpha,
            n_steps=n_steps,
            n_mc=n_mc
        )

        err = abs(pred - mc_val)
        abs_errors.append(err)

        records.append({
            "t": t0,
            "x1": x1,
            "x2": x2,
            "nn_value": pred,
            "mc_value": mc_val,
            "abs_error": err
        })

    mean_abs_error = float(np.mean(abs_errors))
    return mean_abs_error, records


# ============================================================
# 4. Trainer
# ============================================================
class Trainer:
    def __init__(self, net, pde_model, batch_size, device):
        self.net = net
        self.model = pde_model
        self.batch_size = batch_size
        self.device = device

        self.total_losses = []
        self.pde_losses = []
        self.terminal_losses = []

        self.mc_errors = []
        self.mc_check_steps = []

    def train(
        self, epochs, lr, print_every=100, mc_every=200,
        test_points=None, mc_n_steps=1000, mc_n_paths=5000,
        lambda_boundary=1.0
    ):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        avg_loss = 0.0

        best_mc_error = float("inf")
        best_state_dict = None

        for epoch in range(1, epochs + 1):
            self.net.train()
            optimizer.zero_grad()

            loss, loss_interior, loss_boundary = self.model.total_loss(
                self.batch_size,
                lambda_boundary=lambda_boundary
            )

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            self.total_losses.append(loss.item())
            self.pde_losses.append(loss_interior.item())
            self.terminal_losses.append(loss_boundary.item())

            if epoch % print_every == 0:
                mean_loss = avg_loss / print_every
                print(
                    f"Epoch {epoch:5d} | "
                    f"total loss = {mean_loss:.6e} | "
                    f"pde loss = {loss_interior.item():.6e} | "
                    f"terminal loss = {loss_boundary.item():.6e}"
                )
                avg_loss = 0.0

            if (test_points is not None) and (epoch % mc_every == 0):
                mean_mc_error, records = evaluate_against_mc(
                    model=self.net,
                    test_points=test_points,
                    T=self.model.T,
                    H=self.model.H,
                    M=self.model.M,
                    C=self.model.C,
                    D=self.model.D,
                    R=self.model.R,
                    sigma=self.model.sigma,
                    alpha=self.model.alpha,
                    device=self.device,
                    n_steps=mc_n_steps,
                    n_mc=mc_n_paths
                )

                self.mc_errors.append(mean_mc_error)
                self.mc_check_steps.append(epoch)

                print(f"           MC mean abs error = {mean_mc_error:.6e}")
                for rec in records:
                    print(rec)

                if mean_mc_error < best_mc_error:
                    best_mc_error = mean_mc_error
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.net.state_dict().items()
                    }

        if best_state_dict is not None:
            self.net.load_state_dict(best_state_dict)
            self.net.to(self.device)
            print(f"\nLoaded best model with MC error = {best_mc_error:.6e}")

    def save_model(self, filename="exercise3_dgm_model.pt"):
        torch.save(self.net.state_dict(), filename)

    def plot_training_loss(self, filename="exercise3_training_loss.png"):
        plt.figure(figsize=(8, 5))
        plt.plot(self.total_losses, label="Total Loss")
        plt.plot(self.pde_losses, label="PDE Loss")
        plt.plot(self.terminal_losses, label="Terminal Loss")
        plt.yscale("log")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Exercise 3.1: Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    def plot_mc_error(self, filename="exercise3_mc_error.png"):
        plt.figure(figsize=(8, 5))
        plt.plot(self.mc_check_steps, self.mc_errors, "-o", label="MC Error")
        plt.yscale("log")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Absolute Error")
        plt.title("Exercise 3.1: DGM Error against Monte Carlo")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()


# ============================================================
# 5. Optional visualisation
# ============================================================
class Demo:
    def __init__(self, net, pde_model, nx1, nx2, device):
        self.net = net
        self.pde_model = pde_model
        self.nx1 = nx1
        self.nx2 = nx2
        self.device = device

        self.x1_range = np.linspace(pde_model.x_low, pde_model.x_high, nx1)
        self.x2_range = np.linspace(pde_model.x_low, pde_model.x_high, nx2)

    def get_solution_slice(self, t_value=0.0):
        est_solution = np.zeros((self.nx1, self.nx2))

        for i, x1 in enumerate(self.x1_range):
            for j, x2 in enumerate(self.x2_range):
                inp = torch.tensor([[t_value, x1, x2]], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    val = self.net(inp).cpu().numpy().item()
                est_solution[i, j] = val

        return est_solution

    def compare_with_mc(self, test_points, n_steps=1000, n_mc=5000):
        results = []

        for (t, x1, x2) in test_points:
            inp = torch.tensor([[t, x1, x2]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                pred = self.net(inp).item()

            x0 = torch.tensor([[x1, x2]], dtype=torch.float32, device=self.device)

            mc = mc_value_constant_control(
                x0=x0,
                t0=t,
                T=self.pde_model.T,
                H=self.pde_model.H,
                M=self.pde_model.M,
                C=self.pde_model.C,
                D=self.pde_model.D,
                R=self.pde_model.R,
                sigma=self.pde_model.sigma,
                alpha=self.pde_model.alpha,
                n_steps=n_steps,
                n_mc=n_mc
            )

            results.append({
                "t": t,
                "x1": x1,
                "x2": x2,
                "nn_value": pred,
                "mc_value": mc,
                "abs_error": abs(pred - mc)
            })

        return results


# ============================================================
# 6. Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Matrices from Exercise 1.1
    H = torch.tensor([[0.1, 0.0],
                      [0.0, 0.2]], dtype=torch.float32)

    M = torch.eye(2, dtype=torch.float32)
    C = torch.eye(2, dtype=torch.float32)
    D = torch.eye(2, dtype=torch.float32)
    R = torch.eye(2, dtype=torch.float32)
    sigma = 0.3 * torch.eye(2, dtype=torch.float32)

    config = ProblemConfig(T=1.0, x_low=-3.0, x_high=3.0)

    test_points = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.2, 1.0, -1.0),
        (0.5, 2.0, 0.5),
    ]

    # DGM network
    net = NetDGM(input_dim=3, hidden_dim=100, num_layers=3, output_dim=1).to(device)
    print(net)

    # PDE model
    pde_model = LinearPDE2D(
        model=net,
        H=H,
        M=M,
        C=C,
        D=D,
        R=R,
        sigma=sigma,
        config=config,
        device=device
    )

    # Trainer
    trainer = Trainer(
        net=net,
        pde_model=pde_model,
        batch_size=2**10,
        device=device
    )

    # Train
    trainer.train(
        epochs=5000,
        lr=1e-3,
        print_every=100,
        mc_every=100,
        test_points=test_points,
        mc_n_steps=1000,
        mc_n_paths=5000,
        lambda_boundary=1.0
    )

    # Save and plot
    trainer.save_model("exercise3_dgm_model.pt")
    trainer.plot_training_loss("exercise3_training_loss.png")
    trainer.plot_mc_error("exercise3_mc_error.png")

    # Final fixed-point MC comparison
    demo = Demo(net=net, pde_model=pde_model, nx1=80, nx2=80, device=device)
    results = demo.compare_with_mc(
        test_points=test_points,
        n_steps=1000,
        n_mc=5000
    )

    print("\nFinal comparison with Monte Carlo:")
    for rec in results:
        print(rec)

    print("\nSaved files:")
    print("  - exercise3_dgm_model.pt")
    print("  - exercise3_training_loss.png")
    print("  - exercise3_mc_error.png")


if __name__ == "__main__":
    main()