import torch
from torch import nn
from .. import bmv, bvmv
from .dynamics import runsys
from typing import Optional, Tuple
from torch.linalg import cholesky, vecdot


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
        Q: torch.Tensor,
        p: torch.Tensor,
        T: int,
        max_linesearch_iter: int = 10,
        linesearch_decay: float = 0.5,
        max_qp_iter: int = 10,
        qp_decay: float = 0.2,
        gamma: float = 1e-1,
    ):
        super().__init__()
        self.system = system
        self.Q, self.p, self.T = Q, p, T
        self.x_traj = None
        self.u_traj = None
        self.max_linesearch_iter = max_linesearch_iter
        self.linesearch_decay = linesearch_decay
        self.max_qp_iter = max_qp_iter
        self.qp_decay = qp_decay
        self.gamma = gamma

        if self.Q.ndim == 3:
            self.Q = torch.tile(self.Q.unsqueeze(-3), (1, self.T, 1, 1))

        if self.p.ndim == 2:
            self.p = torch.tile(self.p.unsqueeze(-2), (1, self.T, 1))

        self.n_batch = self.p.shape[:-2]

        assert self.Q.shape[:-1] == self.p.shape, "Shape not compatible."
        assert self.Q.size(-1) == self.Q.size(-2), "Shape not compatible."
        assert self.Q.ndim == 4 or self.p.ndim == 3, "Shape not compatible."
        assert self.Q.device == self.p.device, "Device not compatible."
        assert self.Q.dtype == self.p.dtype, "Tensor data type not compatible."
        self.dargs = {"dtype": self.p.dtype, "device": self.p.device}

    def forward(
        self,
        x_init: torch.Tensor,
        dt: int = 1,
        u_traj: Optional[torch.Tensor] = None,
        u_lower: Optional[torch.Tensor] = None,
        u_upper: Optional[torch.Tensor] = None,
        du: Optional[torch.Tensor] = None,
        old_cost: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Performs LQR for the discrete system.

        Args:
            x_init (:obj:`Tensor`): The initial state of the system.
            dt (:obj:`int`): The interval (:math:`\delta t`) between two time steps.
                Default: `1`.
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
        K, k = self.lqr_backward(x_init, dt, u_traj, u_lower, u_upper, du)

        x, u, cost = self.lqr_forward(
            x_init,
            K,
            k,
            u_lower,
            u_upper,
            du,
            old_cost=old_cost,
        )
        return x, u, cost

    def lqr_backward(
        self,
        x_init: torch.Tensor,
        dt: int,
        u_traj: torch.Tensor = None,
        u_lower: Optional[torch.Tensor] = None,
        u_upper: Optional[torch.Tensor] = None,
        du: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the backward recursion of the LQR algorithm.

        Args:
            x_init (:obj:`Tensor`): The initial state of the system.
            dt (:obj:`int`): The interval (:math:`\delta t`) between two time steps.
            u_traj (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.
            u_lower (:obj:`Tensor`, optinal): The lower bounds on the controls.
                Default: ``None``.
            u_upper (:obj:`Tensor`, optinal): The upper bounds on the controls.
                Default: ``None``.
            du (:obj:`int`, optinal): The amount each component of the controls
                is allowed to change in each LQR iteration. Default: ``None``.

        Returns:
            Tuple of :obj:`Tensor`: A tuple of tensors including the feedback gain
            :math:`\mathbf{K}` and the feedforward term :math:`\mathbf{k}`.
        """

        ns, nsc = x_init.size(-1), self.p.size(-1)
        nc = nsc - ns
        prev_kt = None

        if u_traj is None:
            self.u_traj = torch.zeros(self.n_batch + (self.T, nc), **self.dargs)
        else:
            self.u_traj = u_traj

        self.x_traj = x_init.unsqueeze(-2).repeat((1, self.T, 1))

        self.x_traj = runsys(self.system, self.T, self.x_traj, self.u_traj)

        K = torch.zeros(self.n_batch + (self.T, nc, ns), **self.dargs)
        k = torch.zeros(self.n_batch + (self.T, nc), **self.dargs)

        V = torch.zeros(self.n_batch + (self.T, nsc, nsc), **self.dargs)
        v = torch.zeros(self.n_batch + (self.T, nsc), **self.dargs)

        xut = torch.cat((self.x_traj[..., : self.T, :], self.u_traj), dim=-1)
        p = bmv(self.Q, xut) + self.p

        for t in range(self.T - 1, -1, -1):
            if t == self.T - 1:
                Qt = self.Q[..., t, :, :]
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
                if A.ndim == 4:
                    A = torch.stack([A[i, :, i, :] for i in range(A.shape[0])])
                    B = torch.stack([B[i, :, i, :] for i in range(B.shape[0])])

                F = torch.cat((A, B), dim=-1)
                Qt = self.Q[..., t, :, :] + F.mT @ V @ F
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

            if u_lower is None and u_upper is None:

                L = cholesky(Quu_)
                Kt = -torch.cholesky_solve(Qux_, L)
                K[..., t, :, :] = Kt
                kt = -torch.cholesky_solve(qu.unsqueeze(-1), L).squeeze(-1)
                k[..., t, :] = kt
            else:
                lb = u_lower[..., t, :] - u_traj[..., t, :]
                ub = u_upper[..., t, :] - u_traj[..., t, :]
                if du is not None:
                    lb = torch.max(lb, -du)
                    ub = torch.min(ub, du)

                # The following code is to find the optimal variation
                # of the control input delta_u for the current time step

                # argmin(0.5 * delta_u' * Quu * delta_u + qu' * delta_u)
                # solved for delta_u

                # prev_kt is the initial guess for the QP solver
                # prev_kt is the delta_u from the previous iteration

                kt, Qt_uu_free_LU, I_free = self.solve_qp(
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
                Qux_[..., I_free.logical_not().repeat(Qux_.size()[-2:])] = 0
                Kt = -torch.linalg.lu_solve(*Qt_uu_free_LU, Qux_)
                K[..., t, :, :] = Kt
                k[..., t, :] = kt
            V = Qxx_ + Qxu_ @ Kt + Kt.mT @ Qux_ + Kt.mT @ Quu_ @ Kt
            v = qx_ + bmv(Qxu_, kt) + bmv(Kt.mT, qu_) + bmv(Kt.mT @ Quu_, kt)

        return K, k

    def lqr_forward(
        self,
        x_init,
        K,
        k,
        u_lower=None,
        u_upper=None,
        du=None,
        old_cost=None,
    ):

        assert x_init.device == K.device == k.device
        assert x_init.dtype == K.dtype == k.dtype
        assert x_init.ndim == 2, "Shape not compatible."

        ns, nc = self.x_traj.size(-1), self.u_traj.size(-1)

        u = torch.zeros(self.n_batch + (self.T, nc), **self.dargs)
        delta_u = torch.zeros(self.n_batch + (self.T, nc), **self.dargs)
        cost = torch.zeros(self.n_batch, **self.dargs)
        x = torch.zeros(self.n_batch + (self.T + 1, ns), **self.dargs)
        xt = x[..., 0, :] = x_init
        self.system.systime = 0
        alphas = torch.ones(self.n_batch, **self.dargs)
        old_cost_ = torch.zeros_like(cost) if old_cost is None else old_cost

        for i in range(self.max_linesearch_iter):
            for t in range(self.T):
                Kt, kt = K[..., t, :, :], k[..., t, :]
                delta_xt = xt - self.x_traj[..., t, :]
                delta_u[..., t, :] = bmv(Kt, delta_xt) + torch.diag(alphas).mm(kt)
                u[..., t, :] = ut = delta_u[..., t, :] + self.u_traj[..., t, :]

                if u_lower is not None and u_upper is not None:
                    lb = u_lower[..., t, :]
                    ub = u_upper[..., t, :]
                    if du is not None:
                        lb = torch.max(lb, -du)
                        ub = torch.min(ub, du)
                    u[..., t, :] = ut = torch.clamp(ut, lb, ub)

                xut = torch.cat((xt, ut), dim=-1)
                x[..., t + 1, :] = xt = self.system(xt, ut)[0]
                cost = (
                    cost
                    + 0.5 * bvmv(xut, self.Q[..., t, :, :], xut)
                    + vecdot(xut, self.p[..., t, :])
                )

            if torch.all(cost <= old_cost_):
                break
            alphas[~(cost <= old_cost_)] = (
                alphas[~(cost <= old_cost_)] * self.linesearch_decay
            )
            x[..., 0, :] = xt = x_init

        print("Number of line search iterations: ", i + 1)
        return x, u, cost

    def solve_qp(
        self,
        H: torch.Tensor,
        q: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        n_iter: int = 10,
        decay: float = 0.1,
        gamma: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Solves the quadratic programming problem with box constraints.

        .. math::
            \begin{align*}
                \min &\quad 0.5 x'Hx + q'x \\
                \text{s.t.} &\quad \text{lower} \leq x \leq \text{upper}
            \end{align*}

        Args:
            H (:obj:`Tensor`): The symmetric Hessian matrix.
            q (:obj:`Tensor`): The linear term.
            lower (:obj:`Tensor`): The lower bound.
            upper (:obj:`Tensor`): The upper bound.
            x_init (:obj:`Tensor`, optional): The initial guess. Default: ``None``.
            n_iter (:obj:`int`, optional): The number of iterations. Default: ``10``.

        Returns:
            Tuple of :obj:`Tensor`: A tuple of tensors including the solution :math:`x`
            and the number of iterations.
        """

        n_batch, n, _ = H.size()  # n is the number of control inputs
        pnqp_I = 1e-8 * torch.eye(n).type_as(H).expand_as(
            H
        )  # small value to avoid numerical issues

        def obj(x: torch.Tensor) -> torch.Tensor:
            """Objective function for the QP problem
            :math:`0.5 x'Hx + q'x`."""
            return 0.5 * bvmv(x, H, x) + vecdot(q, x)

        if x_init is None:
            # If we don't have an initial guess, we can use the following
            # formula to get an initial guess.
            # x_init = - H^-1 @ q
            H_lu = torch.linalg.lu_factor(H)
            x_init = -torch.linalg.lu_solve(*H_lu, q.unsqueeze(-1)).squeeze(-1)
        else:
            x_init = x_init.clone()  # Don't over-write the original x_init.

        x = torch.clamp(x_init, lower, upper)
        dx = torch.zeros_like(x)

        # Iteratively solve the QP problem

        for i in range(n_iter):
            # Compute the gradient of the objective function
            grad = bmv(H, x) + q  # Gradient of the objective function

            # find the constrained set and the free set of the control inputs
            I_constrained = ((x == lower) & (grad > 0)) | ((x == upper) & (grad < 0))
            I_free = ~I_constrained

            grad_ = grad.clone()
            grad_[..., I_constrained] = 0  # it is equivalent to
            # grad_free = q_free + H_free_free @ x_free + H_free_constrained @ x_constrained

            H_ = H.clone()
            # compute outher product of the free set
            I_Free = I_free.unsqueeze(-1) * I_free.unsqueeze(-2)
            # compute the negation
            Constrained = I_Free.logical_not()

            H_[..., Constrained] = 0

            H_ = (
                H_ + pnqp_I
            )  # add a small value to the diagonal to avoid numerical issues

            # TODO instead of computing the inverse of the Hessian, we can solve the linear system
            # dx_free = -H_free_free^-1 @ (q_free + H_free_constrained @ x_constrained) - x_free

            H_lu_Free = torch.linalg.lu_factor(H_[..., [I_Free]])

            # if the free set is empty we can't solve the linear system
            if I_free.sum() == 0:
                dx_free = torch.zeros_like(x[..., I_free])
            else:
                dx_free = -torch.linalg.lu_solve(
                    *H_lu_Free, grad_[..., I_free].unsqueeze(-1)
                ).squeeze(-1)

            dx[..., I_free] = dx_free
            dx[..., I_constrained] = 0
            # Compute the descent direction

            H_lu_ = torch.linalg.lu_factor(H_)

            # J is a mask that indicates the elements in the batch that are not zero
            J = (
                torch.norm(dx, dim=-1) > 1e-4
            )  # Check if the norm of the descent direction is greater than 1e-4

            # compute the number of environments that have a non-zero descent direction
            n_J = J.sum()

            if n_J == 0:
                return x, H_lu_, I_free

            # Do a line search to find the step size that minimizes the objective function
            alpha = torch.ones(n_batch).type_as(x)

            max_armijo = gamma
            for j in range(10):
                # line search to find the step size that minimizes the objective function
                x_new = torch.clamp(x + torch.diag(alpha).mm(dx), lower, upper)
                armijos = (gamma + 1e-6) * torch.ones(n_batch).type_as(
                    x
                )  # initialize the armijo condition
                # we update the armijo condition only if the environment has a non-zero descent direction
                armijos[J] = (obj(x) - obj(x_new))[J] / vecdot(grad, x - x_new)[J]
                I_arm = armijos <= gamma
                alpha[I_arm] = alpha[I_arm] * decay
                max_armijo = torch.max(armijos)
                if max_armijo > gamma:
                    break
                print("max_armijo = ", max_armijo)

            x = x_new

        print("The number of iterations is: ", i)
        print(
            "[WARNING] The solution is not optimal. The number of components that are not zero is: ",
            n_J,
        )
        return x, H_lu_, I_free
