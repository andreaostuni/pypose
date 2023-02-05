import torch, argparse
from pypose.module import PF
from bicycle import Bicycle, bicycle_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--N", type=int, default=1000, help='The number of particle.')
    args = parser.parse_args()

    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation

    input = torch.randn(T, M, device=args.device) * 0.1 + \
            torch.tensor([1, 0], device=args.device)
    state = torch.zeros(T, N, device=args.device)  # true states
    est = torch.randn(T, N, device=args.device) * p  # estimation
    obs = torch.zeros(T, N, device=args.device)  # observation
    P = torch.eye(N, device=args.device).repeat(T, 1, 1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=args.device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=args.device) * r ** 2  # covariance of observation

    bicycle = Bicycle()
    filter = PF(bicycle, Q, R, particle_number=args.N).to(args.device)

    for i in range(T - 1):
        w = q * torch.randn(N, device=args.device)
        v = r * torch.randn(N, device=args.device)
        state[i + 1], obs[i] = bicycle(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = filter(est[i], obs[i] + v, input[i], P[i])

    bicycle_plot('PF', state, est, P)
