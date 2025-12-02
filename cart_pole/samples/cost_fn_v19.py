import numpy as np

def double_integrator_discretized(dt: float = 0.01):
    A = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=float)
    B = np.array([[0.5 * dt ** 2],
                  [dt]], dtype=float)
    c = np.zeros(2, dtype=float)
    return A, B, c


def cost_fn(x, u, u_prev=None, terminal=False):
    A, B, c = double_integrator_discretized(0.02)
    Q = np.diag([1.0, 0.5])
    R = 0.01
    QT = np.diag([5.0, 2.0])
    x = np.asarray(x, dtype=float).ravel()
    p = x[0]
    v = x[1]
    u_clipped = float(np.clip(float(u), -2.0, 2.0))
    if terminal:
        x2 = np.array([p, v], dtype=float)
        cost = float(x2 @ QT @ x2)
        return cost, u_clipped
    x2 = np.array([p, v], dtype=float)
    cost = float(x2 @ Q @ x2 + R * (u_clipped ** 2))
    if u_prev is not None:
        du = u_clipped - float(u_prev)
        cost += 0.001 * (du ** 2)
    return cost, u_clipped