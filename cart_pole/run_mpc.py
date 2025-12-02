# # run_provided_cartpole_mpc.py
# import numpy as np
# import matplotlib.pyplot as plt
# from ex_cart_pole import CartPoleProvidedEnv
# from mpc_qp import solve_mpc_qp
#
# def get_cost():
#     Q = np.diag([80.0, 2.0, 4.0, 0.2])   # theta, x, theta_dot, x_dot
#     R = np.array([[0.05]])
#     P = Q
#     x_ref = np.zeros(4)
#     return {"Q": Q, "R": R, "P": P, "x_ref": x_ref}
#
# def get_cons(info):
#     th_lim = np.deg2rad(float(info.get("angle_limit_deg", 12.0)))
#     x_lim  = float(info.get("pos_limit", 2.4))
#     u_max  = float(info.get("u_max", 10.0))
#     x_min = np.array([-th_lim, -x_lim, -np.inf, -np.inf])
#     x_max = np.array([+th_lim, +x_lim, +np.inf, +np.inf])
#     return {"x_min": x_min, "x_max": x_max, "u_min": -u_max, "u_max": +u_max}
#
# def vec_from_obs(obs):
#     return np.array([obs["theta"], obs["x"], obs["theta_dot"], obs["x_dot"]], dtype=float)
#
# def main():
#     env = CartPoleProvidedEnv(dt=0.02, horizon=2000,
#                               M=1.0, m=1.0, l=1.0, g=9.8,
#                               u_max=10.0)
#     obs, info = env.reset(seed=0)
#     # ---- larger initial condition ----
#     theta0 = np.deg2rad(10.0)   # try 10 deg (safe under 12-deg limit)
#     x0     = 0.5                # meters
#     thd0   = 0.0
#     xd0    = 0.0
#     env.state = np.array([theta0, x0, thd0, xd0], dtype=float)
#     obs = env._obs()  # refresh the observation dict
#     params = dict(M=env.M, m=env.m, l=env.l, g=env.g)
#
#     N = 50; dt = env.dt; T = 400
#     xs, thetas, us = [], [], []
#     warm = None
#
#     for _ in range(T):
#         x0 = vec_from_obs(obs)
#         u, sol = solve_mpc_qp(x0, N=N, dt=dt, params=params,
#                               cost=get_cost(), cons=get_cons(info), warm=warm)
#         u = float(np.clip(u, -env.u_max, env.u_max))
#         obs, _, done, info = env.step({"force": u})
#
#         xs.append(obs["x"]); thetas.append(obs["theta"]); us.append(u)
#
#         if sol.get("U") is not None:
#             U, X = sol["U"], sol["X"]
#             warm = {"U": np.hstack([U[:, 1:], U[:, -1:]]),
#                     "X": np.hstack([X[:, 1:], X[:, -1:]])}
#         else:
#             warm = None
#         if done: break
#
#     t = np.arange(len(xs))*dt
#     fig, ax = plt.subplots(3,1,figsize=(9,7),sharex=True)
#     ax[0].plot(t, xs);     ax[0].axhline(+info["pos_limit"], ls="--"); ax[0].axhline(-info["pos_limit"], ls="--"); ax[0].set_ylabel("x (m)"); ax[0].set_title("Cart position")
#     ax[1].plot(t, np.degrees(thetas)); ax[1].axhline(+info["angle_limit_deg"], ls="--"); ax[1].axhline(-info["angle_limit_deg"], ls="--"); ax[1].set_ylabel("theta (deg)"); ax[1].set_title("Pole angle")
#     ax[2].plot(t, us);     ax[2].axhline(+info["u_max"], ls="--"); ax[2].axhline(-info["u_max"], ls="--"); ax[2].set_ylabel("u (N)"); ax[2].set_xlabel("time (s)"); ax[2].set_title("Control input")
#     fig.tight_layout(); plt.show()
#
# if __name__ == "__main__":
#     main()

# run_provided_cartpole_sampling.py
import numpy as np
import matplotlib.pyplot as plt
from ex_cart_pole import CartPoleProvidedEnv
from mpc_sampling import solve_mpc_sampling


def get_cost():
    Q = np.diag([80.0, 2.0, 4.0, 0.2])  # theta, x, theta_dot, x_dot
    R = np.array([[0.05]])
    P = Q
    x_ref = np.zeros(4)
    return {"Q": Q, "R": R, "P": P, "x_ref": x_ref}


def get_cons(info):
    th_lim = np.deg2rad(float(info.get("angle_limit_deg", 12.0)))
    x_lim = float(info.get("pos_limit", 2.4))
    u_max = float(info.get("u_max", 10.0))
    x_min = np.array([-th_lim, -x_lim, -np.inf, -np.inf])
    x_max = np.array([+th_lim, +x_lim, +np.inf, +np.inf])
    return {"x_min": x_min, "x_max": x_max, "u_min": -u_max, "u_max": +u_max}


def vec_from_obs(obs):
    return np.array([obs["theta"], obs["x"], obs["theta_dot"], obs["x_dot"]], dtype=float)


def main():
    env = CartPoleProvidedEnv(dt=0.02, horizon=2000,
                              M=1.0, m=1.0, l=1.0, g=9.8,
                              u_max=10.0)
    obs, info = env.reset(seed=0)
    # larger initial condition
    env.state = np.array([np.deg2rad(10.0), 0.5, 0.0, 0.0], dtype=float)
    obs = env._obs()
    params = dict(M=env.M, m=env.m, l=env.l, g=env.g)

    N = 5
    dt = env.dt
    T = 100
    xs, thetas, us = [], [], []
    warm = None

    for _ in range(T):
        x0 = vec_from_obs(obs)
        u, sol = solve_mpc_sampling(
            x0, N=N, dt=dt, params=params,
            cost=get_cost(), cons=get_cons(info),
            warm=warm,
            K=512, Ne=64, iters=4, init_std=3.0, block=2, seed=None, alpha=1.0
        )
        u = float(np.clip(u, -env.u_max, env.u_max))
        obs, _, done, info = env.step({"force": u})

        xs.append(obs["x"]);
        thetas.append(obs["theta"]);
        us.append(u)

        # carry warm start forward
        warm = sol.get("warm_next", None)
        if done: break

    t = np.arange(len(xs)) * dt
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, xs);
    ax[0].axhline(+info["pos_limit"], ls="--");
    ax[0].axhline(-info["pos_limit"], ls="--");
    ax[0].set_ylabel("x (m)");
    ax[0].set_title("Cart position")
    ax[1].plot(t, np.degrees(thetas));
    ax[1].axhline(+info["angle_limit_deg"], ls="--");
    ax[1].axhline(-info["angle_limit_deg"], ls="--");
    ax[1].set_ylabel("theta (deg)");
    ax[1].set_title("Pole angle")
    ax[2].plot(t, us);
    ax[2].axhline(+info["u_max"], ls="--");
    ax[2].axhline(-info["u_max"], ls="--");
    ax[2].set_ylabel("u (N)");
    ax[2].set_xlabel("time (s)");
    ax[2].set_title("Control input")
    fig.tight_layout();
    plt.show()


if __name__ == "__main__":
    main()

