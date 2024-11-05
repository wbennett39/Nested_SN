import numpy as np
import matplotlib.pyplot as plt
from transport_sn import solve
from sn_transport_functions import convergence_estimator, reaction_rate
import tqdm
from show_loglog import show_loglog
# This notebook will do standard convergence tests comparing cc and Gauss quadrature for a 1d steady problem

def RMSE(l1,l2):
    return np.sqrt(np.mean((l1-l2)**2))

def perform_convergence(method = 'difference'):
    N_cells = 100
    N_ang_bench = 512
    # method = 'difference'
    N_ang_list = np.array([2,6,16,46, 136, 406])
    J_list = np.zeros(N_ang_list.size)
    cc_err = np.zeros((3, N_ang_list.size))
    gauss_err = np.zeros((3, N_ang_list.size))
    phi_cc_true = np.zeros((N_ang_list.size, N_cells))
    
    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = N_cells, N_ang = N_ang_bench, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type='gauss')

    for iang, ang in tqdm.tqdm(enumerate(N_ang_list)):
        psicc, phicc, cell_centerscc, muscc, tableaucc, Jcc, tableauJcc, sigmas = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = True, laststep = True,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]))
        
        psig, phig, cell_centersg, musg, tableaug, Jg, tableauJg, sigmas = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type = 'gauss')
        
        phi_cc_true[iang,:] = phicc
        J_list[iang] = Jcc[1]
        gauss_err[0,iang] = RMSE(phig, phib)
        cc_err[0, iang] = RMSE(phicc, phib)
        gauss_err[1,iang] = RMSE(Jg[0], Jb[0])
        gauss_err[2,iang] = RMSE(Jg[1], Jb[1])
        cc_err[0, iang] = RMSE(phicc, phib)
        cc_err[1, iang] = RMSE(Jb[0], Jcc[0])
        cc_err[2, iang] = RMSE(Jb[1], Jcc[1])

    phi_err_estimate = np.zeros((N_ang_list.size, N_cells))
    err_estimate = np.zeros(N_ang_list.size)
    J_err_estimate = np.zeros(N_ang_list.size)
    reaction_rate_bench = reaction_rate(cell_centersb, phib, sigmas[1], -0.5, 0.5)
    for ang in range(2,N_ang_list.size):
        target_estimate = np.zeros(N_cells)
        J_err_estimate[ang] = convergence_estimator(N_ang_list[0:ang], tableauJcc[-1][1:, 1][0:ang], method = method, target=N_ang_bench)
        # J_err_estimate[ang] = convergence_estimator(N_ang_list[0:ang], J_list[0:ang], method = method, target=N_ang_bench)
        for ix in range(N_cells):
            target_estimate[ix] = convergence_estimator(N_ang_list[0:ang], tableaucc[ix][1:, 1][0:ang], method = method, target = N_ang_bench)
            # target_estimate[ix] = convergence_estimator(N_ang_list[0:ang], phi_cc_true[0:ang, ix], method = method)
            phi_err_estimate[ang, ix] = target_estimate[ix]
            # print(target_estimate[ix], phib[ix])
        err_estimate[ang] = RMSE(target_estimate, target_estimate*0)
    print(tableauJcc[-1][1:, 1][0:], 'J tab')
    print(cc_err[2], 'cc err J')
    plt.figure('Scalar flux')
    plt.loglog(N_ang_list, cc_err[0], 'b-^', mfc = 'none')
    plt.loglog(N_ang_list, gauss_err[0], 'r-o', mfc = 'none')
    # plt.loglog(N_ang_list[2:], err_estimate[2:], '-s', mfc = 'none')
    print(err_estimate)
    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel('RMSE')
    show_loglog(f'flux_converge_method={method}', 1, N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)
    # plt.savefig(f'flux_converge_method={method}.pdf')
    plt.show()

    plt.figure('J')
    # plt.loglog(N_ang_list, cc_err[1], 'r--^', mfc = 'none',  label = 'left')
    # plt.loglog(N_ang_list, gauss_err[1], 'b--o', mfc = 'none',  label = 'left')
    plt.loglog(N_ang_list, cc_err[2], 'b-^', mfc = 'none',  label = 'right')
    plt.loglog(N_ang_list, gauss_err[2], 'r-o', mfc = 'none', label = 'right')
    plt.loglog(N_ang_list[2:], J_err_estimate[2:], 'k--')
    # plt.legend()
    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel(r'$|J^+_b-J^+_N|$', fontsize = 16)
    show_loglog(f'J_converge_method={method}', 1, N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)
    # plt.savefig(f'J_converge_method={method}.pdf')
    plt.show()


    plt.figure('error vs x')
    plt.plot(cell_centersb, phi_err_estimate[-1,:])
    plt.plot(cell_centersb, np.abs(phicc - phib), '--')

    plt.show()

def estimate_error(ang_list, tableau):
    err_estimate = np.zeros(ang_list.size)
    for ia in range(2, ang_list.size):
        # print(tableau[1:, 1][0:ia])
        err_estimate[ia] = convergence_estimator(ang_list[0:ia], tableau[1:, 1][0:ia])
    return err_estimate
