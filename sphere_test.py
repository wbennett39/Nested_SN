import numpy as np
from transport_sn import solve
import matplotlib.pyplot as plt
import scipy

def run_problem(cells = 10, N_ang = 8, L =10):
    # psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = 8, N_ang = 2, left_edge = 'cold', right_edge = 'source1', IC = 'cold', source = 'off',
    #             opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 1, tol = 1e-16, source_strength = 100.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e8, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 4, ang_diff_type = 'diamond')
    # psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'source1', IC = 'cold', source = 'off',
                # opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 10.0, sigma_t = 10.0,  strength = [0.0,1], maxits = 1e4, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 8, ang_diff_type = 'SH')
    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = L, tol = 1e-13, source_strength = 1.0, sigma_a = 0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e3, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 1, ang_diff_type = 'diamond')  
    psib2, phib2, cell_centersb2, musb2, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = L, tol = 1e-13, source_strength = 1.0, sigma_a = 0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e3, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = int(N_ang), ang_diff_type = 'SH')            
    
    L = 1/3/100
    plt.ion()
    plt.figure('scalar flux')
    plt.plot(cell_centersb, phib, label = 'DD')
    plt.plot(cell_centersb2, phib2, label = 'SH')
    plt.legend()
    # plt.plot(cell_centersb, (1/np.sinh(1/L))*np.sinh(cell_centersb/L)/cell_centersb, 'k-')
    plt.show()
    plt.figure('reflecting condition')
    plt.plot(musb2[:], psib2[:, 0])
    plt.plot(musb2[:], np.flip(psib2[:, 0]))
    plt.plot(musb[:], np.flip(psib[:, 0]), 'k-')
    plt.show()

# run_problem(cells = 5)

run_problem(cells = 20)
# run_problem(cells = 100)
def RMSE(l1, l2):
    temp = (l1-l2)**2
    temp2 = np.mean(temp)
    temp3 = np.sqrt(temp2)
    return temp3
def convergence(cells = 10):
    RMSE_diamond = []
    RMSE_SH = []
    
    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = 128, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-13, source_strength = 1.0, sigma_a = 0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e4, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 1, ang_diff_type = 'diamond')  
    N_ang_list = [2, 4, 8, 16]
    for iang in N_ang_list:
        psiD, phiD, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = iang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-13, source_strength = 1.0, sigma_a = 0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e4, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = 1, ang_diff_type = 'diamond')  
        psib2, phib2, cell_centersb2, musb2, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = iang, left_edge = 'cold', right_edge = 'cold', IC = 'cold', source = 'volume',
                opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 10, tol = 1e-13, source_strength = 1.0, sigma_a = 0, sigma_s = 1.0, sigma_t = 1.0,  strength = [0.0,1], maxits = 1e4, input_source = np.array([0.0]), quad_type='gauss_legendre', geometry = 'sphere', N_psi_moments = int(iang), ang_diff_type = 'SH')            
        RMSE_diamond.append(RMSE(phiD, phib))
        RMSE_SH.append(RMSE(phib2, phib))
    

    plt.loglog(N_ang_list, RMSE_diamond, '-^', label = 'Diamond differencing')
    plt.loglog(N_ang_list, RMSE_SH, '-o', label = 'Legendre')
    plt.legend()
    plt.show()


# convergence(cells = 10)