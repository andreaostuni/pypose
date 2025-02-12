# %%
import torch
import pypose as pp

import gymnasium as gym

from torch.profiler import profile, record_function, ProfilerActivity
import multiprocessing

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    multiprocessing.set_start_method("spawn", force=True)

    ENV_NAME = "Pendulum-v1"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n_batch, n_state, n_ctrl, T = 2, 3, 1, 10
    dt = 0.05
    g = 10.0
    time_ = torch.arange(0, T, device=device) * dt
    current_u = torch.sin(time_).unsqueeze(1).unsqueeze(0)
    current_u = current_u.repeat(n_batch, 1, 1)

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

    # %%
    # env = gym.make_vec(ENV_NAME, render_mode="human", num_envs=n_batch)
    env = gym.make_vec(ENV_NAME, num_envs=n_batch)
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
    u_upper = torch.tile(
        torch.tensor(env.action_space.high, device=device), (T, n_ctrl)
    )
    solver_exp = Pendulum(dt, exp["len"], exp["m"], g)
    mpc_exp = pp.module.MPC(
        solver_exp,
        # exp["Q"],
        # exp["p"],
        T,
        u_lower=u_lower,
        u_upper=u_upper,
        max_linesearch_iter=2,
        max_qp_iter=4,
        qp_decay=0.2,
    )

    # %%
    env.action_space

    # %%
    # Interact with the environment using the MPC model

    # obs, _ = env.reset()
    obs, _ = env.reset()
    # set the initial state to an upright pendulum

    for i in range(100):
        x_init = torch.tensor(obs, dtype=torch.float32, device=device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        ) as prof:
            # with record_function("MPC_Execution"):
            x_true, u_true, cost = mpc_exp(
                x_init,
                exp["Q"],
                exp["p"],
                dt,
                u_init=current_u,
            )
        # print the top 10 functions that take up the most time
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        action = u_true[0, 0].detach().cpu().numpy()
        # print(f"action: {action}")
        action.reshape([1, 1])
        # action = 0 if action < 0 else 1
        obs, reward, truncated, terminated, _ = env.step(action)
        # env.render()
        # print(f'step {i} reward: {reward}')

        # print(f"step {i} reward: {reward} action: {action}")

        # print(f"x_true: {x_true} vs x_init: {x_init} vs obs: {obs}")
        # print(f"next predicted state: {x_true[:,1]}")
        # print(
        #     f"next state: {torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)}"
        # )
        # print(f"u_true: {u_true}")
        current_u = u_true
        if terminated or truncated:
            state = env.reset()

    env.close()
