import numpy as np
import matplotlib.pyplot as plt

# Define parameters
xF = 1.0
delta = 0.001

# Outer solution: u_outer(x) = 1 - x, for 0 <= x <= 1.
def u_outer(x):
    return 1 - x

# Composite solution: u_comp(x) = (1 - x)*[1 - tanh((x-1)/delta)]/2.
def u_comp(x):
    return (1 - x) * 0.5 * (1 - np.tanh((x - xF) / delta))

# Define x values
x = np.linspace(0, 1.2, 400)

# Compute solutions
u_outer_vals = u_outer(x)
u_comp_vals = u_comp(x)

# Plotting
plt.figure(figsize=(8,5))
plt.plot(x, u_outer_vals, 'r--', label='Outer solution: $1-x$')
plt.plot(x, np.abs(u_comp_vals)**.25, 'b-', lw=2, label='Composite solution')
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Composite (matched) solution for 1d Radiation Diffusion')
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.xlim(0,1.2)
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# Parameters
T_s = 1.0       # Source temperature (in units such that T_s = 1)
xF = 1.0        # Front position (Marshak wave front)
x_star = 0.95   # Matching point between outer and inner solutions
delta = 0.05    # Boundary layer thickness in the inner region

# Outer solution: valid for x <= x_star
def T_outer(x):
    # For x between 0 and xF, outer solution (Marshak-like) is:
    # T_outer(x) = T_s*(1 - x/xF)^(1/4)
    return T_s * (1 - x/xF)**(1/4)

# Inner solution: valid for x >= x_star.
# We assume at x = x_star the temperature is T_outer(x_star)
A = T_outer(x_star)
def T_inner(x):
    return A * np.exp(-(x - x_star)/delta)

# Composite solution: piecewise definition
def T_comp(x):
    return np.where(x <= x_star, T_outer(x), T_inner(x))

# Define x grid for plotting (from 0 to 1.2)
x = np.linspace(0, 1.2, 400)
T_outer_vals = T_outer(x)
T_inner_vals = T_inner(x)
T_comp_vals = T_comp(x)

# Plot the results
plt.figure(figsize=(8,5))
plt.plot(x, T_outer_vals, 'r--', label='Outer solution')
plt.plot(x, T_inner_vals, 'g:', label='Inner solution')
plt.plot(x, T_comp_vals, 'b-', lw=2, label='Composite solution')
plt.xlabel('$x$')
plt.ylabel('$T(x)$')
plt.title('Composite solution via matched asymptotics\nfor a 1d non-equilibrium radiation diffusion model')
plt.legend()
plt.grid(True)
plt.xlim(0, 1.2)
plt.ylim(-0.05, 1.05)
plt.show()
