```python
def cost_fn(x, u, u_prev=None, terminal=False):
    """
    Self-contained double-integrator cost.
    x: array-like state
    u: control (float)
    u_prev: previous control (float or None)
    terminal: if True, compute terminal-state-only cost
    Returns: (cost: float, u_clipped: float)
    """
    if u_prev is None:
        u_prev = 0.0
    dt = 1.0  # fixed time step
    f = 0.5 * (x + u_prev)  # drift at x
    G = u  # input map at x
    cost = 0.5 * (f**2 + G**2)  # cost function
    if terminal:
        cost = cost - dt * (f**2 + G**2)  # terminal-state-only cost
    u_clipped = np.clip(u, -1.0, 1.0)  # clip control to [-1, 1]
    return cost, u_clipped
```