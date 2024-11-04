import numpy as np
import matplotlib.pyplot as plt
from transport_sn import solve
import tqdm
# This notebook will do standard convergence tests comparing cc and Gauss quadrature for a 1d steady problem

def RMSE(l1,l2):
    return np.sqrt(np.mean((l1-l2)**2))

def perform_convergence():
    N_ang_list = np.array([2,6,16,46,136, 406])
    cc_err = np.zeros((3, N_ang_list.size))
    gauss_err = np.zeros((3, N_ang_list.size))
    N_cells = 100
    psib, phib, cell_centersb, musb, tableaub, Jb = solve(N_cells = N_cells, N_ang = 512, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type='gauss')

    for iang, ang in tqdm.tqdm(enumerate(N_ang_list)):
        psicc, phicc, cell_centerscc, muscc, tableaucc, Jcc = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]))
        
        psig, phig, cell_centersg, musg, tableaug, Jg = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type = 'gauss')
        
        gauss_err[0,iang] = RMSE(phig, phib)
        cc_err[0, iang] = RMSE(phicc, phib)
        gauss_err[1,iang] = RMSE(Jg[0], Jb[0])
        gauss_err[2,iang] = RMSE(Jg[1], Jb[1])
        cc_err[0, iang] = RMSE(phicc, phib)
        cc_err[1, iang] = RMSE(Jb[0], Jcc[0])
        cc_err[2, iang] = RMSE(Jb[1], Jcc[1])


    plt.figure('Scalar flux')
    plt.loglog(N_ang_list, cc_err[0], '-^', mfc = 'none')
    plt.loglog(N_ang_list, gauss_err[0], '-o', mfc = 'none')
    plt.show()

    plt.figure('J')
    plt.loglog(N_ang_list, cc_err[1], 'r-^', mfc = 'none',  label = 'left')
    plt.loglog(N_ang_list, gauss_err[1], 'b-o', mfc = 'none',  label = 'left')
    plt.loglog(N_ang_list, cc_err[2], 'g-^', mfc = 'none',  label = 'right')
    plt.loglog(N_ang_list, gauss_err[2], 'p-o', mfc = 'none', label = 'right')
    plt.show()

