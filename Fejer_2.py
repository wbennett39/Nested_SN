import numpy as np

def fejer2_paper(n):
    """
    Implements Fejér's second rule as described in:
      Dumont-Le Brazidec & Peter (2018), 
      "Strongly nested 1D interpolatory quadrature..."
      Section 3.1, eq. (20).
    
    Parameters
    ----------
    n : int
        Number of interior nodes (i.e. excluding endpoints).
    
    Returns
    -------
    x : ndarray of shape (n,)
        The n quadrature nodes, x_k = cos(k*pi/(n+1)), k=1..n
    w : ndarray of shape (n,)
        The corresponding quadrature weights.
    """
    # Indices k = 1..n
    k = np.arange(1, n+1)
    # Angles theta_k = k*pi/(n+1)
    theta = k * np.pi / (n+1)
    
    # Quadrature nodes
    x = np.cos(theta)
    
    # Quadrature weights
    w = np.zeros(n)
    M = (n+1) // 2  # upper limit in the sine-sum
    for i in range(n):
        s = 0.0
        for ell in range(1, M+1):
            s += np.sin((2*ell - 1)*theta[i]) / (2*ell - 1)
        w[i] = (4.0 / (n+1)) * np.sin(theta[i]) * s
    
    return x, w

def main():
    # Example usage: compute and print nodes & weights for n=5
    n = 5
    x, w = fejer2_paper(n)
    
    print(f"Fejér's Second Rule (n={n})")
    print("---------------------------")
    print("Nodes (x_k):")
    print(x)
    print("\nWeights (w_k):")
    print(w)
    print(f"\nSum of the weights = {np.sum(w)}")
    print("(Should be very close to 2.)")

if __name__ == "__main__":
    main()
