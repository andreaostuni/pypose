# %%
import torch
import pypose as pp
import numpy as np
import gymnasium as gym
import time

import torch._dynamo

torch._dynamo.config.cache_size_limit = 64  # Increase cache limit from 8 to 64
from skrl_examples.environments.pytorch_pendulum_env import PendulumEnvTorch


# ENV_NAME = "Pendulum-v1"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_batch, n_state, n_ctrl, T = 1, 3, 1, 5
dt = 0.2
g = 10.0
time_ = torch.arange(0, T, device=device) * dt
current_u = torch.sin(time_).unsqueeze(1).unsqueeze(0)
current_u = current_u.repeat(n_batch, 1, 1)


def timed(fn):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    torch.cuda.synchronize()
    # print(f"elapsed time: {end - start}")
    return result, start - end


# %%
class Pendulum(pp.module.NLS):
    def __init__(self, dt, length, m, g=10.0):
        super().__init__()
        self.dt = dt
        self.length = length
        self.m = m
        self.g = g

    def state_transition(self, state, input, t=None):
        """
        Vectorized state transition function for batch operations.

        Parameters:
            state (torch.Tensor): Tensor of shape (batch_size, 3)
            representing the current state [x, y, theta_dot].

            input (torch.Tensor): Tensor of shape (batch_size, 1)
            representing the input torque.
            t (torch.Tensor or None): Optional time variable
            (not used in this implementation).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 3) representing
            the next state [x, y, theta_dot].
        """
        state_ = state.clone()
        input_ = input.clone()

        theta = torch.atan2(state_[..., 1:2], state_[..., 0:1])

        thetaAcc = (
            3.0 * self.g / (2.0 * self.length) * torch.sin(theta)
            + 3.0 / (self.m * self.length**2) * input_
        )

        _dstate = torch.cat(
            (
                state_[..., 2:3] + self.dt * thetaAcc,
                thetaAcc,
            ),
            dim=-1,
        )

        theta = theta + _dstate[..., 0:1] * self.dt
        return torch.cat(
            (
                self.length * torch.cos(theta),
                self.length * torch.sin(theta),
                state_[..., 2:3] + _dstate[..., 1:2] * self.dt,
            ),
            dim=-1,
        )

    def observation(self, state, input, t=None):
        return state


def cost_fn(trajectory):
    return (
        # torch.norm(torch.atan2(trajectory[..., 1], trajectory[..., 0])) ** 2
        torch.norm(trajectory[..., 1]) ** 2
        + torch.norm(1.0 - trajectory[..., 0]) ** 2
        + 0.1 * trajectory[..., 2] ** 2
        + 0.001 * trajectory[..., 3] ** 2
    )


# %%
# env = gym.make_vec(ENV_NAME, render_mode="human", num_envs=n_batch)
env = PendulumEnvTorch(num_envs=n_batch, device=device)
# env = gym.make_vec(ENV_NAME, num_envs=n_batch)
# %%
# expert
goal_weights = torch.tensor(
    [1.0, 1.0, 0.1], device=device
)  # penalize the angular velocity more
goal_state = torch.tensor(
    [1.0, 0.0, 0.0], device=device
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

torch.manual_seed(0)
u_lower = torch.tile(torch.tensor(env.action_space.low, device=device), (T, n_ctrl))
u_upper = torch.tile(torch.tensor(env.action_space.high, device=device), (T, n_ctrl))
solver_exp = Pendulum(dt, exp["len"], exp["m"], g)

# compile the MPC model

mpc_exp = pp.module.MPC(
    solver_exp.to(device),
    T,
    u_lower=u_lower,
    u_upper=u_upper,
    max_linesearch_iter=2,
    max_qp_iter=4,
    qp_decay=0.2,
).to(device)

# %%
env.action_space

# %%
# Interact with the environment using the MPC model

# obs, _ = env.reset()
obs, _ = env.reset()
# set the initial state to an upright pendulum
u_init = current_u.to(device)

# mpc_opt = torch.compile(
#     pp.module.MPC(
#         solver_exp.to(device),
#         T,
#         u_lower=u_lower,
#         u_upper=u_upper,
#         max_linesearch_iter=2,
#         max_qp_iter=4,
#         qp_decay=0.2,
#     ).to(device),
#     mode="reduce-overhead",
# )
obs, _ = env.reset()
mpc_opt = pp.module.MPC(
    solver_exp.to(device),
    T,
    u_lower=u_lower,
    u_upper=u_upper,
    max_linesearch_iter=2,
    max_qp_iter=4,
    qp_decay=0.2,
).to(device)
for i in range(100):
    x_init = obs.clone().detach().to(device)

    (x_true, u_true, cost), time_ms = timed(
        lambda: mpc_opt(
            x_init,
            # (exp["Q"], exp["p"]),
            cost_fn=cost_fn,
            dt=dt,
            u_init=u_init,
        )
    )

    print(f"compiled evaluation time: {time_ms} ms")
    # print the top 10 functions that take up the most time
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    action = u_true[0, 0:1].detach()
    # print(f"action: {action}")
    # action.reshape(1, -1)
    # action = 0 if action < 0 else 1
    obs, reward, truncated, terminated, _ = env.step(action)
    env.render()
    u_init = u_true
    # u_init = torch.cat(
    #     (
    #         u_init,
    #         u_init[:, -1:, :],
    #     ),
    #     dim=1,
    # ).to(device)
    if terminated or truncated:
        state = env.reset()


env.close()
