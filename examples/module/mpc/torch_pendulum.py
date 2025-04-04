# %%
import torch
import pypose as pp
import numpy as np

# import gymnasium as gym
import time

import torch._dynamo
from skrl_examples.environments.pytorch_pendulum_env import PendulumEnvTorch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

seed = 10


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_batch, n_state, n_ctrl, T = 20, 2, 1, 10
dt = 0.1
g = 10.0
time_ = torch.arange(0, T, device=device) * dt
current_u = torch.sin(time_).unsqueeze(1).unsqueeze(0)
current_u = current_u.repeat(n_batch, 1, 1)


def plot_pendulum(state: np.ndarray, fig: plt.Figure, axes=plt.Axes):
    """
    Plot the pendulum state.
    Parameters:
        state (torch.Tensor): The state of the pendulum.
        axes (plt.Axes): The axes to plot on.
    """

    # the pendulum is a 2D line
    # the state is [theta]
    lines = []
    # Create a line representing the pendulum
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if i * axes.shape[1] + j >= state.shape[0]:
                break
            ax = axes[i, j]
            ax.cla()
            ax.set_title(f"states for batch {i * n_cols+ j}")
            ax.set_xlim(0, iterations)
            ax.set_xlabel("iteration")
            ax.set_ylabel("angle")
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"Pendulum {i * n_cols+ j}")
            ax.set_ylabel("x")
            ax.set_xlabel("y")
            ax.grid()
            # Create a line representing the pendulum
            line = ax.plot([], [], "o-", lw=2)[0]  # Line and bob
            lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return (*lines,)

    def update(frame):
        # Update the pendulum position
        for i, line in enumerate(lines):
            x = -np.sin(state[i, frame])
            y = np.cos(state[i, frame])
            line.set_data([0, x], [0, y])
        return (*lines,)

    # Create an animation
    ani = animation.FuncAnimation(
        fig, update, frames=state.shape[1], init_func=init, blit=False, interval=100
    )
    return ani


def timed(fn):
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    torch.cuda.synchronize()
    return result, start - end


# %%
class Pendulum(pp.module.NLS):
    def __init__(self, dt, length, m, g=10.0, use_custom_jacobians=True):
        super().__init__(use_custom_jacobians=use_custom_jacobians)
        self.dt = dt
        self.length = length
        self.m = m
        self.g = g

    def state_transition(self, state, input, t=None):
        """
        Vectorized state transition function for batch operations.

        Parameters:
            state (torch.Tensor): Tensor of shape (batch_size, 2)
            representing the current state [theta , theta_dot].

            input (torch.Tensor): Tensor of shape (batch_size, 1)
            representing the input torque.
            t (torch.Tensor or None): Optional time variable
            (not used in this implementation).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 2) representing
            the next state [theta , theta_dot].
        """
        state_ = state.clone()
        input_ = input.clone()

        theta = state_[..., 0:1]

        thetaAcc = (
            3.0 * self.g / (2.0 * self.length) * torch.sin(theta)
            + 3.0 / (self.m * self.length**2) * input_
        )
        theta_dot = state_[..., 1:2]

        theta_dot = theta_dot + thetaAcc * self.dt

        theta = theta + theta_dot * self.dt
        return torch.cat(
            (
                theta,
                theta_dot,
            ),
            dim=-1,
        )

    def observation(self, state, input, t=None):
        # Convert the state to Cartesian coordinates

        return torch.cat(
            (
                self.length * torch.cos(state[..., 0:1]),
                self.length * torch.sin(state[..., 0:1]),
                state[..., 1:2],
            ),
            dim=-1,
        )

    def custom_jacobians(self, state, input, t=None):
        """
        Custom Jacobian function for the state transition.
        Parameters:
            state (torch.Tensor): Tensor of shape (batch_size, 2)
            representing the current state [theta, theta_dot].
            input (torch.Tensor): Tensor of shape (batch_size, 1)
            representing the input torque.
            t (torch.Tensor or None): Optional time variable
            (not used in this implementation).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors
            representing the Jacobian matrices (A, B, C, D).
        """
        # Compute the Jacobian matrices A and B
        A = torch.zeros(
            (state.shape[0], state.shape[1], state.shape[1]), device=state.device
        )
        A[..., 0, 1] = 1
        A[..., 1, 0] = 3.0 * self.g / (2.0 * self.length) * torch.cos(state[..., 0])

        Ad = A * self.dt + torch.eye(
            state.shape[1], device=state.device
        )  # Add the identity matrix to the diagonal
        B = torch.zeros(
            (state.shape[0], state.shape[1], input.shape[1]), device=state.device
        )
        B[..., 1, 0] = 3.0 / (self.m * self.length**2)
        Bd = B * self.dt
        C = torch.zeros(
            (state.shape[0], state.shape[1] + 1, state.shape[1]), device=state.device
        )
        C[..., 0, 0:1] = -self.length * torch.sin(state[..., 0:1])
        C[..., 1, 0:1] = self.length * torch.cos(state[..., 0:1])
        C[..., 2, 1:2] = 1.0
        D = torch.zeros(
            (state.shape[0], state.shape[1] + 1, input.shape[1]), device=state.device
        )
        return Ad, Bd, C, D


def cost_fn(trajectory):
    return (
        # trajectory[..., 0] ** 2
        (1.0 - torch.cos(trajectory[..., 0])) ** 2
        + (torch.sin(trajectory[..., 0])) ** 2
        + 0.1 * trajectory[..., 1] ** 2
        + 0.001 * trajectory[..., 2] ** 2
    )


env = PendulumEnvTorch(num_envs=n_batch, device=device)

goal_weights = torch.tensor(
    [10.0, 0.1], device=device
)  # penalize the angular velocity more
goal_state = torch.tensor(
    [0.0, 0.0], device=device
)  # pendulum in the upright position, not moving
ctrl_penalty = 1e-3

q = torch.cat(
    [
        goal_weights,
        ctrl_penalty * torch.ones(n_ctrl, device=device),
    ]
)

p = (
    -torch.sqrt(q[:n_state]) * goal_state
)  # we want the pendulum to be upright and not moving
p = torch.cat([p, torch.zeros(n_ctrl, device=device)])
exp = dict(
    Q=torch.tile(torch.diag(q), (n_batch, T, 1, 1)),
    p=torch.tile(p, (n_batch, T, 1)),
    len=torch.tensor(1.0).to(device),
    m=torch.tensor(1.0).to(device),
)

torch.manual_seed(seed)
u_lower = torch.tile(torch.tensor(env.action_space.low, device=device), (T, n_ctrl))
u_upper = torch.tile(torch.tensor(env.action_space.high, device=device), (T, n_ctrl))
solver_exp = Pendulum(dt, exp["len"], exp["m"], g)

obs, _ = env.reset()
# # set the initial state to an upright pendulum
# u_init = current_u.to(device)
u_init = None

obs, _ = env.reset(seed=seed)
mpc_opt = pp.module.MPC(
    solver_exp.to(device),
    T,
    u_lower=u_lower,
    u_upper=u_upper,
    max_linesearch_iter=5,
    max_qp_iter=4,
    qp_decay=0.2,
).to(device)

iterations = 200
costs = torch.zeros((n_batch, iterations), device=device)
rewards = torch.zeros((n_batch, iterations), device=device)
states = torch.zeros((n_batch, iterations), device=device)
velocities = torch.zeros((n_batch, iterations), device=device)
actions = torch.zeros((n_batch, iterations), device=device)
for i in range(iterations):
    obs_ = obs.clone().detach().to(device)
    x_init = torch.cat(
        (
            torch.atan2(obs_[..., 1:2], obs_[..., 0:1]),
            obs_[..., 2:3],
        ),
        dim=-1,
    )

    (x_true, u_true, cost), time_ms = timed(
        lambda: mpc_opt(
            x_init,
            # (exp["Q"], exp["p"]),
            cost_fn=cost_fn,
            dt=dt,
            # u_init=u_init,
        )
    )

    print(f"compiled evaluation time: {time_ms} ms")
    # print the top 10 functions that take up the most time
    action = u_true[..., 0, :].detach()
    obs, reward, truncated, terminated, _ = env.step(action)

    cost2 = (
        torch.pow((x_true[..., :T, 0]), 2)
        + 0.1 * x_true[..., :T, 1] ** 2
        + 0.001 * u_true[..., 0] ** 2
    ).sum(dim=-1)
    cost1 = (
        torch.pow(torch.sin(x_true[..., :T, 0]), 2)
        + torch.pow(1.0 - torch.cos(x_true[..., :T, 0]), 2)
        + 0.1 * x_true[..., :T, 1] ** 2
        + 0.001 * u_true[..., 0] ** 2
    ).sum(dim=-1)

    next_predicted_state = solver_exp.observation(
        x_true[..., 1, :].detach(), u_true[..., 1, :].detach()
    )
    print(f"obs - x_true1 {obs - next_predicted_state}")
    costs[:, i] = cost
    rewards[:, i] = reward.squeeze()
    states[:, i] = x_true[..., 0, 0]
    velocities[:, i] = x_true[..., 0, 1]
    actions[:, i] = action.squeeze(1)

    # env.render()
    u_init = u_true[..., 1:, :].detach()
    u_init = torch.cat(
        (
            u_init,
            u_true[..., -1:, :],
        ),
        dim=1,
    )


# plot the cost and reward
import matplotlib.pyplot as plt

num_plots = n_batch
fig, axs = plt.subplots(num_plots, 2, figsize=(10, 5))
axs = np.atleast_2d(axs)

for i in range(n_batch):
    axs[i, 0].plot(costs[i].cpu().numpy(), label="cost")
    axs[i, 1].plot(rewards[i].cpu().numpy(), label="reward")
    axs[i, 0].set_title(f"costs for batch {i}")
    axs[i, 1].set_title(f"rewards for batch {i}")
    axs[i, 0].legend()
    axs[i, 1].legend()
axs[0, 0].set_xlabel("iteration")
axs[0, 1].set_xlabel("iteration")
axs[0, 0].set_ylabel("cost")
axs[0, 1].set_ylabel("reward")
fig.suptitle("Costs and Rewards")

# plot the states as an animation
plt.savefig("costs_rewards.png")
n_cols = min(5, num_plots)

n_rows = num_plots // n_cols + num_plots % n_cols
fig, axs = plt.subplots(
    nrows=n_rows, ncols=n_cols, figsize=(10, 5), sharex=True, sharey=True
)
axs = np.atleast_2d(axs)
for i in range(n_rows):
    for j in range(n_cols):
        if i * n_cols + j >= num_plots:
            break
        axs[i, j].cla()
        axs[i, j].plot(velocities[i * n_cols + j, :].cpu().numpy(), label="velocities")
        axs[i, j].plot(actions[i * n_cols + j, :].cpu().numpy(), label="actions")
        axs[i, j].plot(states[i * n_cols + j, :].cpu().numpy(), label="states")
        axs[i, j].set_title(f"states for batch {i * n_cols+ j}")
        axs[i, j].set_ylim(-10, 10)
        axs[i, j].set_xlim(0, iterations)
        axs[i, j].set_xlabel("iteration")
        axs[i, j].legend()
fig.suptitle("States")
plt.savefig("states.png")

fig, axs = plt.subplots(
    nrows=n_rows, ncols=n_cols, figsize=(10, 5), sharex=True, sharey=True
)
axs = np.atleast_2d(axs)

ani = plot_pendulum(
    states.cpu().numpy(), fig, axs
)  # pass the figure and axes to the function

ani.save("pendulums.gif", writer="imagemagick")
# save the figure
env.close()
