"""
Design a self-contained cost function by cost_fn for CEM-based sampling MPC
"""
import numpy as np

def double_integrator_discretized(dt: float):
    A = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=float)
    B = np.array([[0.5*dt**2],
                  [dt]], dtype=float)
    c = np.zeros(2, dtype=float)
    return A, B, c

def cost_fn(x, u, u_prev=None, terminal=False):
    """Self-contained double-integrator cost.
    x: array-like state
    u: control (float)
    u_prev: previous control (float or None)
    terminal: if True, compute terminal-state-only cost
    Returns: (cost: float, u_clipped: float)
    """

    # # constants (edit as desired)
    # dt = 0.02
    # u_min, u_max = -2.0, 2.0
    # Q = np.diag([1.0, 0.5])      # running state weights
    # R = 0.01                     # running control weight
    # QT = np.diag([5.0, 2.0])     # terminal state weights (usually larger)
    #
    # x = np.asarray(x, dtype=float).ravel()
    # p = x[0] if x.size > 0 else 0.0
    # v = x[1] if x.size > 1 else 0.0
    #
    # u_clipped = float(np.clip(float(u), u_min, u_max))
    #
    # if terminal:
    #     # terminal cost: penalize final state only (no control penalties here)
    #     x2 = np.array([p, v], dtype=float)
    #     cost = float(x2 @ QT @ x2)
    #     return cost, u_clipped
    #
    # # running cost: state + small control effort (+ optional smoothness)
    # x2 = np.array([p, v], dtype=float)
    # cost = float(x2 @ Q @ x2 + R * (u_clipped ** 2))
    #
    # # optional smoothness term
    # if u_prev is not None:
    #     du = u_clipped - float(u_prev)
    #     cost += 0.001 * (du ** 2)
    #
    # return cost, u_clipped