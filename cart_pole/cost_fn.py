import numpy as np

def cost_fn(x, u, u_prev=None, terminal=False):
    """
    Self-contained double-integrator cost.
    
    Args:
    x (list): [position, velocity] at time t
    u (float): control input
    u_prev (float, optional): previous control input. Defaults to None.
    terminal (bool, optional): whether the end of the simulation is reached. Defaults to False.
    
    Returns:
    (float, float): cost and clipped control input.
    """
    # Calculate the derivative of the cost function
    dxdt = x[1]
    dvdt = x[0]
    
    # Calculate the cost function
    cost = 0.5 * np.linalg.norm(dxdt)**2 + 0.5 * np.linalg.norm(dvdt)**2
    
    # Clip the control input to the range [-2, 2]
    u_clipped = np.clip(u, -2, 2)
    
    # If the end of the simulation is reached, return the cost and clipped control input
    if terminal:
        return cost, u_clipped
    else:
        return cost, u_clipped