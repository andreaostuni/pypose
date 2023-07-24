import torch
import pypose as pp


n_state, n_ctrl = 4, 2
n_sc = n_state+n_ctrl
T = 5
n_batch = 1

expert_seed = 333
torch.manual_seed(expert_seed)

alpha = 0.2

expert = dict(
    Q = torch.tile(torch.eye(n_sc), (n_batch, T, 1, 1)),
    p = torch.tile(torch.randn(n_sc), (n_batch, T, 1)),
    A = torch.eye(n_state) \
        + 0.2 * torch.randn(n_state, n_state),
    B = torch.randn(n_state, n_ctrl),)


x_init = torch.randn(n_batch,n_state)
u_lower, u_upper = None, None
u_init = None

C = torch.eye(n_state)
D = torch.zeros(n_state, n_ctrl)
c1 = torch.zeros(n_state)
c2 = torch.zeros(n_state)
dt = 1
stepper = pp.utils.ReduceToBason(steps=1, verbose=False)

lti = pp.module.LTI(expert['A'], expert['B'], C, D, c1, c2)
mpc = pp.module.MPC(lti, expert['Q'], expert['p'], T, stepper=stepper)
x_true, u_true, cost_true = mpc.forward(dt, x_init)

print(x_true)

u_lower = torch.tile(torch.randn(n_ctrl), (n_batch, T, 1))
u_upper = torch.tile(torch.randn(n_ctrl), (n_batch, T, 1))

lti = pp.module.LTI(expert['A'], expert['B'], C, D, c1, c2)
mpc = pp.module.MPC(lti, expert['Q'], expert['p'], T, stepper=stepper)
x_true, u_true, cost_true = mpc.forward(dt, x_init, u_lower=u_lower, u_upper=u_upper)

print(x_true)
