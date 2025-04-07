import torch
from torch import nn
from .. import bmv, bvmv
from .dynamics import runsys
from typing import Optional, Tuple, Union, Callable, Dict
from torch.linalg import cholesky, vecdot
from pypose.utils.qp_solver import solve_qp
from torch.func import vmap, jacrev, hessian


class LQR(nn.Module):
    r"""
    Linear Quadratic Regulator (LQR) with Dynamic Programming.

    Args:
        system (:obj:`instance`): The system to be soved by LQR.
        Q (:obj:`Tensor`): The weight matrix of the quadratic term.
        p (:obj:`Tensor`): The weight vector of the first-order term.
        T (:obj:`int`): Time steps of system.

    A discrete-time linear system can be described as:

    .. math::
        \begin{align*}
            \mathbf{x}_{t+1} &= \mathbf{A}_t\mathbf{x}_t + \mathbf{B}_t\mathbf{u}_t
                                                         + \mathbf{c}_{1t}          \\
            \mathbf{y}_t &= \mathbf{C}_t\mathbf{x}_t + \mathbf{D}_t\mathbf{u}_t
                                                     + \mathbf{c}_{2t}              \\
        \end{align*}

    where :math:`\mathbf{x}`, :math:`\mathbf{u}` are the state and input of the linear
    system; :math:`\mathbf{y}` is the observation of the linear system; :math:`\mathbf{A}`
    and :math:`\mathbf{B}` are the state matrix and input matrix of the linear system;
    :math:`\mathbf{C}`, :math:`\mathbf{D}` are the output matrix and observation matrix
    of the linear system; :math:`\mathbf{c}_{1}`, :math:`\mathbf{c}_{2}` are the constant
    input and constant output of the linear system. The subscript :math:`\cdot_{t}`
    denotes the time step.

    LQR finds the optimal nominal trajectory :math:`\mathbf{\tau}_{1:T}^*` =
    :math:`\begin{Bmatrix} \mathbf{x}_t, \mathbf{u}_t \end{Bmatrix}_{1:T}`
    for the linear system of the optimization problem:

    .. math::
        \begin{align*}
          \mathbf{\tau}_{1:T}^* = \mathop{\arg\min}\limits_{\tau_{1:T}}
            &\sum\limits_t\frac{1}{2}
          \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t
            + \mathbf{p}_t^\top\mathbf{\tau}_t \\
          \mathrm{s.t.} \quad \mathbf{x}_1 &= \mathbf{x}_{\text{init}}, \\
          \mathbf{x}_{t+1} &= \mathbf{F}_t\mathbf{\tau}_t + \mathbf{c}_{1t} \\
        \end{align*}

    where :math:`\mathbf{\tau}_t` = :math:`\begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t
    \end{bmatrix}`, :math:`\mathbf{F}_t` = :math:`\begin{bmatrix} \mathbf{A}_t &
    \mathbf{B}_t \end{bmatrix}`.

    One way to solve the LQR problem is to use the dynamic programming, where the process
    can be summarised as a backward and a forward recursion.

    - The backward recursion.

      For :math:`t` = :math:`T` to 1:

    .. math::
        \begin{align*}
            \mathbf{Q}_t &= \mathbf{Q}_t +
                \mathbf{F}_t^\top\mathbf{V}_{t+1} \mathbf{F}_t \\
            \mathbf{q}_t &= \mathbf{p}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}
                            \mathbf{c}_{1t} + \mathbf{F}_t^\top\mathbf{v}_{t+1}  \\
            \mathbf{K}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                        \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} \\
            \mathbf{k}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                        \mathbf{q}_{\mathbf{u}_t}  \\
            \mathbf{V}_t &= \mathbf{Q}_{\mathbf{x}_t, \mathbf{x}_t}
                + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{K}_t
                + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t}
                + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}
                    \mathbf{K}_t \\
            \mathbf{v}_t &= \mathbf{q}_{\mathbf{x}_t}
                + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{k}_t
                + \mathbf{K}_t^\top\mathbf{q}_{\mathbf{u}_t}
                + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t,
                    \mathbf{u}_t}\mathbf{k}_t  \\
        \end{align*}

    - The forward recursion.

      For :math:`t` = 1 to :math:`T`:

        .. math::
            \begin{align*}
                \mathbf{u}_t &= \mathbf{K}_t\mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{x}_{t+1} &= \mathbf{A}_t\mathbf{x}_t + \mathbf{B}_t\mathbf{u}_t
                                                                + \mathbf{c}_{1t} \\
            \end{align*}

    Then quadratic costs of the system over the time horizon:

    .. math::
        \mathbf{c} \left( \mathbf{\tau}_t \right) = \frac{1}{2}
        \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t
            + \mathbf{p}_t^\top\mathbf{\tau}_t

    For the **non-linear system**, sometimes people want to solve MPC problem with
    **iterative LQR**. A discrete-time non-linear system can be described as:

    .. math::
        \begin{aligned}
            \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
            \mathbf{y}_{t} &= \mathbf{g}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
        \end{aligned}

    We can do a linear approximation at current point :math:`\chi^*=(\mathbf{x}^*,
    \mathbf{u}^*, t^*)` along a trajectory with small perturbation
    :math:`\chi=(\mathbf{x}^*+\delta\mathbf{x}, \mathbf{u}^* +\delta\mathbf{u}, t^*)`
    near :math:`\chi^*` for both dynamics and cost:

    .. math::
            \begin{aligned}
            \mathbf{f}(\mathbf{x}, \mathbf{u}, t^*) &\approx \mathbf{f}(\mathbf{x}^*,
                \mathbf{u}^*, t^*) +  \left.\frac{\partial \mathbf{f}}{\partial\mathbf{x}}
                \right|_{\chi^*} \delta \mathbf{x} + \left. \frac{\partial \mathbf{f}}
                {\partial \mathbf{u}} \right|_{\chi^*} \delta \mathbf{u} \\
            &= \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*) + \mathbf{A} \delta \mathbf{x}
                + \mathbf{B} \delta \mathbf{u} \\
            \delta \mathbf{x}_{t+1} &= \mathbf{A}_t \delta \mathbf{x}_t + \mathbf{B}_t
                \delta \mathbf{u}_t \\
            &= \mathbf{F}_t \delta \mathbf{\tau}_t \\
            \mathbf{c} \left( \mathbf{\tau}, t^* \right) &\approx
                \mathbf{c} \left( \mathbf{\tau}^*, t^* \right) + \frac{1}{2} \delta
                \mathbf{\tau}^\top\nabla^2_{\mathbf{\tau}}\mathbf{c}\left(\mathbf{\tau}^*,
                t^* \right) \delta \mathbf{\tau} + \nabla_{\mathbf{\tau}}
                \mathbf{c} \left( \mathbf{\tau}^*, t^* \right)^\top \delta \mathbf{\tau}\\
            \bar{\mathbf{c}} \left( \delta \mathbf{\tau} \right) &= \frac{1}{2} \delta
                \mathbf{\tau}_t^\top \bar{\mathbf{Q}}_t \delta \mathbf{\tau}_t +
                \bar{\mathbf{p}}_t^\top \delta \mathbf{\tau}_t \\
            \end{aligned}

    where :math:`\delta \mathbf{\tau}_t` = :math:`\begin{bmatrix} \delta \mathbf{x}_t \\
    \delta \mathbf{u}_t \end{bmatrix}`, :math:`\mathbf{F}_t` = :math:`\begin{bmatrix}
    \mathbf{A}_t & \mathbf{B}_t \end{bmatrix}`, :math:`\bar{\mathbf{Q}}_t = \mathbf{Q}_t`,
    :math:`\bar{\mathbf{p}}_t` = :math:`\mathbf{Q}_t \mathbf{\tau}^*_t + \mathbf{p}_t`.

    Then, LQR can be performed on a linear quadractic problem with
    :math:`\delta \mathbf{\tau}_t`, :math:`\mathbf{F}_t`,
    :math:`\bar{\mathbf{Q}}_t` and :math:`\bar{\mathbf{p}}_t`.

    - The backward recursion.

      For :math:`t` = :math:`T` to 1:

    .. math::
        \begin{align*}
            \mathbf{Q}_t &= \bar{\mathbf{Q}}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}
                                \mathbf{F}_t \\
            \mathbf{q}_t &= \bar{\mathbf{p}}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}  \\
            \mathbf{K}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}^{-1}
                                \mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{x}_t} \\
            \mathbf{k}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}^{-1}
                                \mathbf{q}_{\delta \mathbf{u}_t}         \\
            \mathbf{V}_t &= \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{x}_t}
                + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{K}_t
                + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{x}_t}
                + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}
                    \mathbf{K}_t \\
            \mathbf{v}_t &= \mathbf{q}_{\delta \mathbf{x}_t}
                + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{k}_t
                + \mathbf{K}_t^\top\mathbf{q}_{\delta \mathbf{u}_t}
                + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t,
                    \delta \mathbf{u}_t}\mathbf{k}_t   \\
        \end{align*}

    Note:
        Because we made a linear approximation, here :math:`\bar{\mathbf{p}}_t` leads to a
        difference with :math:`{\mathbf{q}}_t` and :math:`{\mathbf{k}}_t` relative to the
        linear backward recursion, and this change will be compensated back by
        :math:`\mathbf{u}_t^*` in the forward recursion.

    - The forward recursion.

      For :math:`t` = 1 to :math:`T`:

        .. math::
            \begin{align*}
                \delta \mathbf{u}_t &= \mathbf{K}_t \delta \mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{u}_t &= \delta \mathbf{u}_t + \mathbf{u}_t^* \\
                \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t) \\
            \end{align*}

    Then quadratic costs of the system over the time horizon:

        .. math::
            \mathbf{c} \left( \mathbf{\tau}_t \right) = \frac{1}{2}
            \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t
                + \mathbf{p}_t^\top\mathbf{\tau}_t

    Note:
        The discrete-time system to be solved by LQR can be either linear time-invariant
        (:meth:`LTI`) system, or linear time-varying (:meth:`LTV`) system. For non-linear
        system, one can approximate it as a linear system via Taylor expansion, using
        iterative LQR algorithm for MPC. Here we provide a unified general format for the
        implementation.

    From the learning perspective, this can be interpreted as a module with unknown
    parameters :math:`\begin{Bmatrix} \mathbf{Q}, \mathbf{p}, \mathbf{F}, \mathbf{f}
    \end{Bmatrix}`, which can be integrated into an end-to-end learning system.

    Note:
        The implementation of LQR is based on page 24-32 of the slides:

        * `Optimal Control and Planning <https://tinyurl.com/y5ck36vw>`_.

        The implementation of iterative LQR is based on Eq. (1)~(19) of this paper:

        * Li Weiwei, and Emanuel Todorov, `Iterative linear quadratic regulator design for
          nonlinear biological movement systems <https://tinyurl.com/bdma36s6>`_, ICINCO
          (1), 2004.

    Example:
        >>> torch.manual_seed(0)
        >>> n_batch, T = 2, 5
        >>> n_state, n_ctrl = 4, 3
        >>> n_sc = n_state + n_ctrl
        >>> Q = torch.randn(n_batch, T, n_sc, n_sc)
        >>> Q = torch.matmul(Q.mT, Q)
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> r = 0.2 * torch.randn(n_state, n_state)
        >>> A = torch.tile(torch.eye(n_state) + r, (n_batch, 1, 1))
        >>> B = torch.randn(n_batch, n_state, n_ctrl)
        >>> C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
        >>> D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
        >>> c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
        >>> c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
        >>> x_init = torch.randn(n_batch, n_state)
        >>> u_traj = torch.zeros(n_batch, T, n_ctrl, device=device)
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> dt = 1
        >>> LQR = pp.module.LQR(lti, Q, p, T)
        >>> x, u, cost = LQR(x_init, dt)
        >>> print("x = ", x)
        >>> print("u = ", u)
        x = tensor([[[-0.2633, -0.3466,  2.3803, -0.0423],
                     [ 0.1849, -1.3884,  1.0898, -1.6229],
                     [ 1.2138, -0.7161,  0.2954, -0.6819],
                     [ 1.4840, -1.1249, -1.0302,  0.9805],
                     [-0.3477, -1.7063,  4.6494,  2.6780],
                     [ 7.2346,  4.9958, 17.9926, -7.7881]],
                    [[-0.9744,  0.4976,  0.0603, -0.5258],
                     [-0.6356,  0.0539,  0.7264, -0.5048],
                     [-0.2275, -0.1649,  0.3872, -0.4614],
                     [ 0.2697, -0.3577,  0.0999, -0.4594],
                     [ 0.3916, -2.0832,  0.0701, -0.5407],
                     [ 1.0404, -1.3799, -2.0913, -0.1459]]])
        u = tensor([[[ 1.0405,  0.1586, -0.1282],
                     [-1.4845, -0.5745,  0.2523],
                     [-0.6322, -0.3281, -0.3620],
                     [-1.6768,  2.4054, -0.1047],
                     [-1.7948,  3.5269,  9.0703]],
                    [[-0.1795,  0.9153,  1.7066],
                     [ 0.0814,  0.4004,  0.7114],
                     [ 0.0436,  0.5782,  1.0127],
                     [-0.3017, -0.2897,  0.7251],
                     [-0.0728,  0.7290, -0.3117]]])
    """

    def __init__(
        self,
        system: nn.Module,
        T: int,
        u_lower: Optional[torch.Tensor] = None,
        u_upper: Optional[torch.Tensor] = None,
        du: Optional[torch.Tensor] = None,
        action_dim: Optional[int] = None,
        max_linesearch_iter: int = 10,
        linesearch_decay: float = 0.5,
        max_qp_iter: int = 10,
        qp_decay: float = 0.2,
        gamma: float = 1e-1,
    ):
        super().__init__()
        self.system = system
        self.T = T
        self.x_traj = None
        self.u_traj = None
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.du = du

        self.action_dim = action_dim
        if self.action_dim is not None:
            assert self.action_dim == u_lower.size(-1)
        else:
            assert (
                u_lower is not None and u_upper is not None
            ), "action_dim or u bounds must be provided"
            self.action_dim = u_lower.size(-1)

        self.max_linesearch_iter = max_linesearch_iter
        self.linesearch_decay = linesearch_decay
        self.max_qp_iter = max_qp_iter
        self.qp_decay = qp_decay
        self.gamma = gamma
        self.dargs = None

    def forward(
        self,
        x_init: torch.Tensor,
        cost_fn: Union[Tuple[torch.Tensor, torch.Tensor], Callable],
        dt: float = 1.0,
        cost_kwargs: Optional[Dict] = None,
        u_traj: Optional[torch.Tensor] = None,
        old_cost: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Performs LQR for the discrete system.

        Args:
            x_init (:obj:`Tensor`): The initial state of the system.
            dt (:obj:`float`): The interval (:math:`\delta t`) between two time steps.
                Default: `1.0`.
            cost_fn (:obj:`Tuple`): The cost function of the system.
                can be a tuple of Q, p or a callable function.
            cost_kwargs (:obj:`Dict`, optinal): The additional arguments for the cost function.
                Default: ``None``. (only used when cost_fn is a callable function)
            u_traj (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.
            u_lower (:obj:`Tensor`, optinal): The lower bounds on the controls.
                Default: ``None``.
            u_upper (:obj:`Tensor`, optinal): The upper bounds on the controls.
                Default: ``None``.
            du (:obj:`int`, optinal): The amount each component of the controls
                is allowed to change in each LQR iteration. Default: ``None``.

            For the line search in the forward recursion:
            old_cost (:obj:`Tensor`, optinal): The old cost of the system. Default: ``None``.

        Returns:
            List of :obj:`Tensor`: A list of tensors including the solved state sequence
            :math:`\mathbf{x}`, the solved input sequence :math:`\mathbf{u}`, and the
            associated quadratic costs :math:`\mathbf{c}` over the time horizon.
        """
        if self.dargs is None:
            self.dargs = {"dtype": x_init.dtype, "device": x_init.device}
            # self.n_batch = p.shape[:-2]
        if cost_kwargs is None and callable(cost_fn):
            cost_kwargs = {"args": (), "in_dims": ()}

        K, k = self.lqr_backward(
            x_init=x_init,
            cost_fn=cost_fn,
            cost_kwargs=cost_kwargs,
            dt=dt,
            u_traj=u_traj,
        )

        # instead of using Q, p, we use the cost function

        x, u, cost, du_norm = self.lqr_forward(
            x_init,
            cost_fn=cost_fn,
            cost_kwargs=cost_kwargs,
            K=K,
            k=k,
            old_cost=old_cost,
        )
        return x, u, cost, du_norm

    # @torch.compile
    def lqr_backward(
        self,
        x_init: torch.Tensor,
        cost_fn: Union[Tuple[torch.Tensor, torch.Tensor], Callable],
        dt: int,
        u_traj: torch.Tensor = None,
        cost_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the backward recursion of the LQR algorithm.

        Args:
            x_init (:obj:`Tensor`): The initial state of the system.
            cost_fn (:obj:`Tuple`): The cost function of the system.
                can be a tuple of Q, p or a callable function.
            dt (:obj:`float`): The interval (:math:`\delta t`) between two time steps.
            u_traj (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.
            cost_kwargs (:obj:`Dict`, optinal): The additional arguments for the cost function.
                Default: ``None``. (only used when cost_fn is a callable function)

        Returns:
            Tuple of :obj:`Tensor`: A tuple of tensors including the feedback gain
            :math:`\mathbf{K}` and the feedforward term :math:`\mathbf{k}`.
        """

        ns, nc, n_batch = x_init.size(-1), self.action_dim, x_init.size(0)
        nsc = ns + nc
        prev_kt = None

        if u_traj is None:
            self.u_traj = torch.zeros((n_batch, self.T, nc), **self.dargs)
        else:
            self.u_traj = u_traj

        self.x_traj = x_init.unsqueeze(-2).repeat((1, self.T, 1))

        self.x_traj = runsys(self.system, self.T, self.x_traj, self.u_traj)

        K = torch.zeros((n_batch, self.T, nc, ns), **self.dargs)
        k = torch.zeros((n_batch, self.T, nc), **self.dargs)

        V = torch.zeros((n_batch, self.T, nsc, nsc), **self.dargs)
        v = torch.zeros((n_batch, self.T, nsc), **self.dargs)

        xut = torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1)
        # Compute the Q, p for the cost function
        if isinstance(cost_fn, Tuple):
            Q, p = cost_fn
            p = bmv(Q, xut) + p
        else:
            assert cost_kwargs is not None, "cost_kwargs must be provided."
            Q, p = self.linearize_cost(cost_fn, cost_kwargs, xut)

        for t in range(self.T - 1, -1, -1):
            if t == self.T - 1:
                Qt = Q[..., t, :, :]
                qt = p[..., t, :]
            else:
                self.system.set_refpoint(
                    state=self.x_traj[..., t, :],
                    input=self.u_traj[..., t, :],
                    t=torch.tensor(
                        t * dt, device=self.x_traj.device, dtype=self.x_traj.dtype
                    ),
                )
                A = self.system.A
                B = self.system.B

                F = torch.cat((A, B), dim=-1)
                Qt = Q[..., t, :, :] + F.mT @ V @ F
                qt = p[..., t, :] + bmv(F.mT, v)

            Qt_, qt_ = Qt.clone(), qt.clone()

            Qxx, Qxu = Qt_[..., :ns, :ns], Qt_[..., :ns, ns:]
            Qux, Quu = Qt_[..., ns:, :ns], Qt_[..., ns:, ns:]
            qx, qu = qt_[..., :ns], qt_[..., ns:]

            qu_ = qu.clone()
            qx_ = qx.clone()
            Quu_ = Quu.clone()
            Qux_ = Qux.clone()
            Qxu_ = Qxu.clone()
            Qxx_ = Qxx.clone()

            if self.u_lower is None or self.u_upper is None:
                # L = cholesky(Quu_)
                L, _ = torch.linalg.cholesky_ex(Quu_)
                Kt = -torch.cholesky_solve(Qux_, L)
                K[..., t, :, :] = Kt
                kt = -torch.cholesky_solve(qu.unsqueeze(-1), L).squeeze(-1)
                k[..., t, :] = kt
            else:
                lb = self.u_lower[t, :].unsqueeze(0) - self.u_traj[..., t, :]
                ub = self.u_upper[t, :].unsqueeze(0) - self.u_traj[..., t, :]
                if self.du is not None:
                    lb = torch.max(lb, -self.du)
                    ub = torch.min(ub, self.du)

                # The following code is to find the optimal variation
                # of the control input delta_u for the current time step

                # argmin(0.5 * delta_u' * Quu * delta_u + qu' * delta_u)
                # solved for delta_u

                # prev_kt is the initial guess for the QP solver
                # prev_kt is the delta_u from the previous iteration

                kt, Qt_uu_free_LLT, I_free = solve_qp(
                    Quu_,
                    qu_,
                    lb,
                    ub,
                    x_init=prev_kt,
                    n_iter=self.max_qp_iter,
                    decay=self.qp_decay,
                    gamma=self.gamma,
                )

                prev_kt = kt
                Qux_ = Qux_.clone()
                Qux_[..., I_free.logical_not().unsqueeze(-1).repeat(1, 1, ns)] = 0
                Kt = -torch.cholesky_solve(Qux_, Qt_uu_free_LLT)
                K[..., t, :, :] = Kt
                k[..., t, :] = kt
                qu_[..., ~I_free] = 0
            V = Qxx_ + Qxu_ @ Kt + Kt.mT @ Qux_ + Kt.mT @ Quu_ @ Kt
            v = qx_ + bmv(Qxu_, kt) + bmv(Kt.mT, qu_) + bmv(Kt.mT @ Quu_, kt)

        return K, k

    # @torch.compile
    def lqr_forward(
        self,
        x_init,
        cost_fn,
        cost_kwargs,
        K,
        k,
        old_cost=None,
    ):

        assert x_init.device == K.device == k.device
        assert x_init.dtype == K.dtype == k.dtype
        assert x_init.ndim == 2, "Shape not compatible."

        ns, nc, n_batch = (
            self.x_traj.size(-1),
            self.u_traj.size(-1),
            self.x_traj.size(0),
        )

        u = torch.zeros((n_batch, self.T, nc), **self.dargs)
        delta_u = torch.zeros((n_batch, self.T, nc), **self.dargs)

        x = torch.zeros((n_batch, self.T + 1, ns), **self.dargs)
        xt = x[..., 0, :] = x_init
        self.system.reset()
        alphas = torch.ones(n_batch, **self.dargs)
        old_cost_ = (
            torch.full((n_batch,), float("inf"), **self.dargs)
            if old_cost is None
            else old_cost
        )
        if old_cost is None:
            if isinstance(cost_fn, Tuple):
                Q, p = cost_fn
                old_cost_ = 0.5 * bvmv(
                    torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1),
                    Q,
                    torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1),
                ).sum(dim=-1) + vecdot(
                    torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1),
                    p,
                ).sum(
                    dim=-1
                )
            else:
                # use vmap to compute the cost function over the time horizon
                old_cost_ = (
                    vmap(
                        cost_fn,
                        in_dims=(
                            1,
                            *[None for _ in range(len(cost_kwargs["in_dims"]))],
                        ),
                    )(
                        torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1),
                        *cost_kwargs["args"],
                    )
                    .movedim(1, 0)
                    .sum(dim=-1)
                )
        for i in range(self.max_linesearch_iter):
            cost = torch.zeros(n_batch, **self.dargs)
            for t in range(self.T):
                Kt, kt = K[..., t, :, :], k[..., t, :]
                delta_xt = xt - self.x_traj[..., t, :]
                delta_u[..., t, :] = bmv(Kt, delta_xt) + torch.diag(alphas).mm(kt)
                u[..., t, :] = ut = delta_u[..., t, :] + self.u_traj[..., t, :]

                if self.u_lower is not None and self.u_upper is not None:
                    lb = self.u_lower[t, :].unsqueeze(0)
                    ub = self.u_upper[t, :].unsqueeze(0)
                    if self.du is not None:
                        lb = torch.max(lb, -self.du)
                        ub = torch.min(ub, self.du)
                    u[..., t, :] = ut = torch.clamp(ut, lb, ub)

                xut = torch.cat((xt, ut), dim=-1)
                x[..., t + 1, :] = xt = self.system(xt, ut)[0]
                if isinstance(cost_fn, Tuple):
                    Q, p = cost_fn
                    cost = (
                        cost
                        + 0.5 * bvmv(xut, Q[..., t, :, :], xut)
                        + vecdot(xut, p[..., t, :])
                    )
                else:
                    cost = cost + cost_fn(xut, *cost_kwargs["args"])

            if torch.all(cost <= old_cost_):
                break
            alphas[~(cost <= old_cost_)] = (
                alphas[~(cost <= old_cost_)] * self.linesearch_decay
            )
            x[..., 0, :] = xt = x_init

        # get the full norm of the update in the control inputs over different iterations
        du_norm = torch.norm((u - self.u_traj).view(-1, nc * self.T), dim=-1)
        return x, u, cost, du_norm

    # @torch.compile
    def linearize_cost(
        self, cost_fn: Callable, cost_kwargs: Dict, xut: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linearizes the cost function.

        Args:
            cost_fn (:obj:`Callable`): The cost function of the system.
            cost_kwargs (:obj:`Dict`): The additional arguments for the cost function.
            xut (:obj:`Tensor`): The concatenated tensor of state and input.

        Returns:
            Tuple of :obj:`Tensor`: A tuple of tensors including the Q and p for the cost function.
        """

        # the xut is the concatenated tensor of state and input over the time horizon
        # xut [n_batch, T, n_state + n_ctrl]

        p = (
            vmap(
                vmap(jacrev(cost_fn, argnums=0), in_dims=(0, *cost_kwargs["in_dims"])),
                in_dims=(1, *[None for _ in range(len(cost_kwargs["in_dims"]))]),
            )(xut, *cost_kwargs["args"])
            .movedim(1, 0)
            .squeeze(2)
        )
        Q = (
            vmap(
                vmap(hessian(cost_fn, argnums=0), in_dims=(0, *cost_kwargs["in_dims"])),
                in_dims=(1, *[None for _ in range(len(cost_kwargs["in_dims"]))]),
            )(xut, *cost_kwargs["args"])
            .movedim(1, 0)
            .squeeze(2)
        ) * 0.5

        # the last term in the time horizon is the terminal cost
        # Q[..., -1, -self.action_dim :, -self.action_dim :] = (
        #     Q[..., -1, -self.action_dim :, -self.action_dim :] * 10.0
        # )
        # p[..., -1, -self.action_dim :] = p[..., -1, -self.action_dim :] * 10.0
        # Q[..., -1, -self.action_dim :, -self.action_dim :] = 1e-8
        # p[..., -1, -self.action_dim :] = 1e-8
        Q = Q + 1e-8 * torch.eye(Q.size(-1), **self.dargs)
        return Q, p
