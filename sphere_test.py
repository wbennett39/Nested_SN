import numpy as np
from transport_sn import solve
import matplotlib.pyplot as plt
import scipy

def run_problem(cells = 10, N_ang = 32):
    # psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = 8, N_ang = 2, left_edge = 'cold', right_edge = 'source1', IC = 'cold', source = 'off',
    #             opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 1, tol = 1e-16, source_strength = 100.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e8, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 4, ang_diff_type = 'diamond')
    # psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'source1', IC = 'cold', source = 'off',
                # opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 10.0, sigma_t = 10.0,  strength = [0.0,1], maxits = 1e4, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 8, ang_diff_type = 'SH')
    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-13, source_strength = 1.0, sigma_a = 0.9, sigma_s = 0.1, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e5, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 2, ang_diff_type = 'diamond')  
    psib2, phib2, cell_centersb2, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-13, source_strength = 1.0, sigma_a = 0.9, sigma_s = 0.1, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e5, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 5, ang_diff_type = 'SHDPN')            
    
    L = 1/3/100
    plt.ion()
    plt.figure('scalar flux')
    plt.plot(cell_centersb, phib)
    plt.plot(cell_centersb2, phib2)
    # plt.plot(cell_centersb, (1/np.sinh(1/L))*np.sinh(cell_centersb/L)/cell_centersb, 'k-')
    plt.show()
    plt.figure('reflecting condition')
    plt.plot(musb[1:], psib2[1:, 0])
    plt.plot(musb[1:], np.flip(psib2[1:, 0]))
    plt.show()

run_problem(cells = 10)

# run_problem(cells = 50)
# run_problem(cells = 100)