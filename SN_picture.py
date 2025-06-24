import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def s4_level_symmetric():
    """
    Returns direction vectors (Omega_x, Omega_y, Omega_z) and
    their associated weights for a 3D S4 level-symmetric set.
    
    This set has 24 total directions (6 per octant) in one common convention.
    Each direction is reflected into all octants, with the same weight.
    """
    # In a level-symmetric S4 set, there are two types of direction cosines:
    #   1) Those with all components the same (e.g., a, a, a)
    #   2) Those with two components equal and one different (e.g., b, b, c)
    #
    # For S4, the typical numeric values (in one standard reference) are:
    #   a = 0.3500212
    #   b = 0.8688903
    #   c = 0.3500212  (same as 'a', but in different positions)
    #
    # The total number of directions is 24, but we can first define them in
    # the "first octant" (x>0,y>0,z>0) and then replicate to all 8 octants.
    
    # Hard-coded direction cosines (first octant) for S4:
    #    1) triple (a, a, a)
    #    2) permutations of (a, b, 0)
    #    3) permutations of (b, a, 0)
    #
    # However, references differ on the exact sets used. Here is one:
    
    # Single triple (all equal):
    a = 0.3500212
    b = 0.8688903
    
    # Directions in the first octant (x, y, z > 0).
    # We'll store them, then replicate to all octants.
    # 
    # The standard S4 set typically has 8 unique directions in the first octant:
    #   1) (a, a, a)
    #   2) (a, b, 0)
    #   3) (b, a, 0)
    #   4) (a, 0, b)
    #   5) (b, 0, a)
    #   6) (0, a, b)
    #   7) (0, b, a)
    #   8) (something with 0, 0, 1)? Actually that belongs to S2 or higher expansions...
    #
    # Different references can define these sets slightly differently. 
    # We'll define a commonly cited set below:
    
    first_octant = [
        ( a,  a,  a),
        ( a,  b,  0),
        ( b,  a,  0),
        ( a,  0,  b),
        ( b,  0,  a),
        ( 0,  a,  b),
        ( 0,  b,  a),
    ]
    
    # We replicate each direction into all octants. If (x,y,z) is in the first octant,
    # then we also want (+/- x, +/- y, +/- z) for all sign combinations.
    
    directions = []
    for (x, y, z) in first_octant:
        for sx in [+1, -1]:
            for sy in [+1, -1]:
                for sz in [+1, -1]:
                    directions.append((sx*x, sy*y, sz*z))
    
    directions = np.array(directions)
    
    # Weights:
    #
    # Often, each direction in an S4 set has the same weight, 
    # because of the level-symmetric structure. The standard total number 
    # of directions is 24 or 32 (depending on the variant). 
    #
    # We'll just assign an equal weight that sums to 4*pi (the total solid angle)
    # or 2*pi if you prefer half-range, etc. Typically for transport in all directions
    # we want 4*pi. 
    # 
    # Let's say we have N directions total. The sum of all weights = 4*pi. 
    # 
    # So weight per direction = 4*pi / N.
    
    N = len(directions)
    w = np.full(N, 4.0 * np.pi / N)
    
    return directions, w

def plot_sn_directions_3d(directions, weights):
    """
    Create a 3D scatter plot of discrete ordinates on the unit sphere,
    coloring by weight.
    
    directions: (N, 3) array of direction cosines
    weights: (N,) array of weights
    """
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize each direction to lie on the unit sphere for plotting
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_dirs = directions / norms
    
    # We will color the points by weight. 
    # For a standard level-symmetric S4 set, the weights might all be equal,
    # but let's do it anyway.
    sc = ax.scatter(
        unit_dirs[:,0], unit_dirs[:,1], unit_dirs[:,2], 
        c=weights, cmap='plasma', s=40, alpha=0.8
    )
    plt.colorbar(sc, ax=ax, label="Quadrature Weight")
    
    # Also draw a light-wire sphere for reference
    # Parametric sphere:
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("3D SN Directions (example: S4 Level-Symmetric)")
    plt.tight_layout()
    plt.show()

def main():
    # 1) Get directions and weights for a simple S4 set
    directions, weights = s4_level_symmetric()
    
    # 2) Plot them in 3D
    plot_sn_directions_3d(directions, weights)

if __name__ == "__main__":
    main()
