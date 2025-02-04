import torch
from typing import Tuple, Optional
from pypose import bmv, bvmv
from torch.linalg import cholesky, vecdot


def solve_qp(
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
        This uses the projected Newton method to solve the QP problem.

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
        # H_lu = torch.linalg.lu_factor(H)
        # x_init = -torch.linalg.lu_solve(*H_lu, q.unsqueeze(-1)).squeeze(-1)
        H_LLT = torch.linalg.cholesky(H)
        x_init = -torch.cholesky_solve(q.unsqueeze(-1), H_LLT).squeeze(-1)
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

        H_ = H_ + pnqp_I  # add a small value to the diagonal to avoid numerical issues

        # TODO instead of computing the inverse of the Hessian, we can solve the linear system
        # dx_free = -H_free_free^-1 @ (q_free + H_free_constrained @ x_constrained) - x_free

        # H_lu_Free = torch.linalg.lu_factor(H_[..., [I_Free])

        H_LLT_ = torch.linalg.cholesky(
            H_
        )  # if the free set is empty we can't solve the linear system
        if I_free.sum() == 0:
            dx = torch.zeros_like(x)
            n_J = 0
        else:
            # dx_free = -torch.linalg.lu_solve(
            #     *H_lu_Free, grad_[..., I_free].unsqueeze(-1)
            # ).squeeze(-1)
            # dx = -torch.linalg.lu_solve(*H_lu_, grad.unsqueeze(-1)).squeeze(-1)
            dx = -torch.cholesky_solve(grad.unsqueeze(-1), H_LLT_).squeeze(-1)
            dx[..., I_constrained] = 0
            # J is a mask that indicates the elements in the batch that are not zero
            # J = (
            #     torch.norm(dx, dim=-1) > 1e-4
            # )  # Check if the norm of the descent direction is greater than 1e-4
            # infinity norm
            J = torch.norm(grad[..., I_free], dim=-1, p=float("inf")) > 1e-4

            # compute the number of environments that have a non-zero descent direction
            n_J = J.sum()
            # Compute the descent direction

        if n_J == 0:
            return x, H_LLT_, I_free

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

    print("The number of qp iterations is: ", i)
    print(
        "[WARNING] The solution is not optimal. The number of components that are not zero is: ",
        n_J,
    )
    return x, H_LLT_, I_free
