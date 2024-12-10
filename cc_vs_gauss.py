import numpy as np
import matplotlib.pyplot as plt
from transport_sn import solve
from sn_transport_functions import convergence_estimator, reaction_rate, cc_quad, wynn_epsilon_algorithm
from functions import quadrature
import tqdm
from show_loglog import show_loglog
from show import show
from matplotlib import ticker
# This notebook will do standard convergence tests comparing cc and Gauss quadrature for a 1d steady problem

def RMSE(l1,l2):
    return np.sqrt(np.mean((l1-l2)**2))
def spatial_converge():
    N_cells_list = np.array([10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 10000, 50000, 500000, 1000000 ])
    reaction_list = np.zeros(N_cells_list.size)
    J_list = np.zeros(N_cells_list.size)
    N_ang_bench = 16
    tol = 1e-15
    opacity = '3_material'
    for k, cells, in enumerate(N_cells_list):
        dx = 5/cells
        psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = cells, N_ang = N_ang_bench, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
                opacity_function = opacity, wynn_epsilon = False, laststep = False,  L = 5.0, tol = tol, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type='gauss')
        reaction_rate_bench = reaction_rate(cell_centersb, phib, sigmas[0], -0.5, 0.5)
        print('RR', reaction_rate_bench, 'dx = ', dx)
        print('J+', Jb[1], 'dx = ', dx)
        print('tol =', tol)
        reaction_list[k] = reaction_rate_bench
        J_list[k] = Jb[1]
        print(abs(reaction_list[k]-reaction_list[k-1]), 'reaction rate diff')
        print(abs(J_list[k] - J_list[k-1]), 'J diff')
    
    plt.loglog(N_cells_list, np.abs(reaction_list))
    plt.show()
    print(np.abs(reaction_list[-2] - reaction_list[-1]), 'tolerance RR')
    print(np.abs(J_list[-2] - J_list[-1]), 'tolerance RR')





def perform_convergence():
    N_cells = 500000
    print(5/N_cells, 'dx')
    # N_cells = 1500
    N_ang_bench = 1024
    # method = 'difference'
    N_ang_list = np.array([2,6,16,46, 136, 406])
    J_list = np.zeros(N_ang_list.size)
    cc_err = np.zeros((3, N_ang_list.size))
    gauss_err = np.zeros((3, N_ang_list.size))
    gaussl_err = np.zeros((3, N_ang_list.size))
    phi_cc_true = np.zeros((N_ang_list.size, N_cells))
    reaction_rate_cc = np.zeros(N_ang_list.size)
    reaction_rate_gauss = np.zeros(N_ang_list.size)
    reaction_rate_gauss_l = np.zeros(N_ang_list.size)
    reaction_rate_tableau = np.zeros((N_ang_list.size+1, N_ang_list.size-1))
    err_estimate = np.zeros(N_ang_list.size)
    J_err_estimate_diff = np.zeros(N_ang_list.size)
    J_err_estimate_lr = np.zeros(N_ang_list.size)
    reaction_err_estimate_diff = np.zeros(N_ang_list.size)
    reaction_rate_estimate_lr = np.zeros(N_ang_list.size)
    J_err_estimate_rich = np.zeros(N_ang_list.size)
    reaction_rate_estimate_rich = np.zeros(N_ang_list.size)
    reaction_rate_estimate_wynn = np.zeros(N_ang_list.size)
    J_err_estimate_wynn = np.zeros(N_ang_list.size)
    reaction_rate_nested = np.zeros(N_ang_list.size)
    opacity = '3_material'
    #prime numba
    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = 50, N_ang = 16, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = opacity, wynn_epsilon = False, laststep = False,  L = 5.0, tol = 5e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e6, input_source = np.array([0.0]), quad_type='gauss')
    reaction_rate_bench = reaction_rate(cell_centersb, phib, sigmas[0], -0.5, 0.5)


    psib, phib, cell_centersb, musb, tableaub, Jb, tableauJb, sigmas = solve(N_cells = N_cells, N_ang = N_ang_bench, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = opacity, wynn_epsilon = False, laststep = False,  L = 5.0, tol = 5e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e6, input_source = np.array([0.0]), quad_type='gauss')
    reaction_rate_bench = reaction_rate(cell_centersb, phib, sigmas[0], -0.5, 0.5)
    print(reaction_rate_bench, 'bench reaction')
    print(Jb[1], 'bench J+')
    for iang, ang in tqdm.tqdm(enumerate(N_ang_list)):
        psicc, phicc, cell_centerscc, muscc, tableaucc, Jcc, tableauJcc, sigmas = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = opacity, wynn_epsilon = True, laststep = True,  L = 5.0, tol = 5e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e6, input_source = np.array([0.0]))
        
        psig, phig, cell_centersg, musg, tableaug, Jg, tableauJg, sigmas = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = opacity, wynn_epsilon = False, laststep = False,  L = 5.0, tol = 5e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e6, input_source = np.array([0.0]), quad_type = 'gauss')
        
        psigl, phigl, cell_centersgl, musgl, tableaugl, Jgl, tableauJgl, sigmas = solve(N_cells = N_cells, N_ang = ang, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
            opacity_function = opacity, wynn_epsilon = False, laststep = False,  L = 5.0, tol = 5e-16, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e6, input_source = np.array([0.0]), quad_type = 'gauss_legendre')
        
        phi_cc_true[iang,:] = phicc
        reaction_rate_cc[iang] = reaction_rate(cell_centerscc, phicc, sigmas[0], -0.5, 0.5)
        reaction_rate_gauss[iang] = reaction_rate(cell_centersg, phig, sigmas[0], -0.5, 0.5)
        reaction_rate_gauss_l[iang] = reaction_rate(cell_centersgl, phigl, sigmas[0], -0.5, 0.5)
        print(reaction_rate_cc, 'cc reaction rate')
        J_list[iang] = Jcc[1]
        gauss_err[0,iang] = RMSE(phig, phib)
        gaussl_err[0,iang] = RMSE(phigl, phib)
        cc_err[0, iang] = RMSE(phicc, phib)
        gauss_err[1,iang] = RMSE(Jg[0], Jb[0])
        gauss_err[2,iang] = RMSE(Jg[1], Jb[1])
        gaussl_err[1,iang] = RMSE(Jgl[0], Jb[0])
        gaussl_err[2,iang] = RMSE(Jgl[1], Jb[1])
        cc_err[0, iang] = RMSE(phicc, phib)
        cc_err[1, iang] = RMSE(Jb[0], Jcc[0])
        cc_err[2, iang] = RMSE(Jb[1], Jcc[1])

        phitest = np.zeros(N_cells)
        for ix in range(N_cells):
            phitest[ix] = tableaucc[ix][1:,1][iang]
        reaction_rate_nested[iang] = reaction_rate(cell_centerscc, phitest , sigmas[0], -0.5, 0.5)
        reaction_rate_tableau = wynn_epsilon_algorithm(reaction_rate_nested[0:iang+1])

        if iang >= 2:
            target_estimate = np.zeros(N_cells)
            J_err_estimate_lr[iang-1] = convergence_estimator(N_ang_list[0:iang], tableauJcc[-1][1:, 1][0:iang], method = 'linear_regression', target=N_ang_list[iang-1])
            J_err_estimate_rich[iang-1] = convergence_estimator(N_ang_list[0:iang], tableauJcc[-1][1:, 1][0:iang], method = 'richardson', target=N_ang_list[iang-1])
            J_err_estimate_diff[iang-1] = convergence_estimator(N_ang_list[0:iang], tableauJcc[-1][1:, 1][0:iang], method = 'difference', target=N_ang_list[iang-1])
            reaction_err_estimate_diff[iang-1] = convergence_estimator(N_ang_list[0:iang], reaction_rate_nested[0:iang], method = 'linear_regression', target=N_ang_list[iang-1])
            reaction_rate_estimate_lr[iang-1] = convergence_estimator(N_ang_list[0:iang], reaction_rate_nested[0:iang], method = 'difference', target=N_ang_list[iang-1])
            reaction_rate_estimate_rich[iang-1] = convergence_estimator(N_ang_list[0:iang], reaction_rate_nested[0:iang], method = 'richardson', target=N_ang_list[iang-1])
            print(reaction_err_estimate_diff)
            if iang < N_ang_list.size:
                reaction_rate_estimate_wynn[iang] =  np.abs(reaction_rate_tableau[3:,3][iang-2] - reaction_rate_tableau[1:,1][iang] )
                J_err_estimate_wynn[iang] = np.abs(tableauJcc[-1][3:,3][iang-2] - tableauJcc[-1][1:,1][iang] )
    target_estimate = np.zeros(N_cells)
    J_err_estimate_lr[-1] = convergence_estimator(N_ang_list, tableauJcc[-1][1:, 1], method = 'linear_regression', target=N_ang_list[-1])
    J_err_estimate_rich[-1] = convergence_estimator(N_ang_list, tableauJcc[-1][1:, 1], method = 'richardson', target=N_ang_list[-1])
    J_err_estimate_diff[-1] = convergence_estimator(N_ang_list, tableauJcc[-1][1:, 1], method = 'difference', target=N_ang_list[-1])
    reaction_err_estimate_diff[-1] = convergence_estimator(N_ang_list, reaction_rate_nested, method = 'linear_regression', target=N_ang_list[-1])
    reaction_rate_estimate_lr[-1] = convergence_estimator(N_ang_list, reaction_rate_nested, method = 'difference', target=N_ang_list[-1])
    reaction_rate_estimate_rich[-1] = convergence_estimator(N_ang_list, reaction_rate_nested, method = 'richardson', target=N_ang_list[-1])


    


    
    
    phi_err_estimate = np.zeros((N_ang_list.size, N_cells))
   
    
    for ang in range(2,N_ang_list.size+1):
        
        # J_err_estimate[ang] = convergence_estimator(N_ang_list[0:ang], J_list[0:ang], method = method, target=N_ang_bench)
        for ix in range(N_cells):
            target_estimate[ix] = convergence_estimator(N_ang_list[0:ang], tableaucc[ix][1:, 1][0:ang], method = 'difference', target = N_ang_list[ang-1])
            # target_estimate[ix] = convergence_estimator(N_ang_list[0:ang], phi_cc_true[0:ang, ix], method = method)
            phi_err_estimate[ang-1, ix] = target_estimate[ix]
            # print(target_estimate[ix], phib[ix])
        err_estimate[ang-1] = RMSE(target_estimate, target_estimate*0)
    print(tableauJcc[-1][1:, 1][0:], 'J tab')
    print(cc_err[2], 'cc err J')
    plt.figure('Scalar flux')
    plt.loglog(N_ang_list, cc_err[0], 'b-^', mfc = 'none', label = 'Clenshaw-Curtis')
    plt.loglog(N_ang_list, gauss_err[0], 'r-o', mfc = 'none', label = 'Gauss-Lobatto')
    plt.loglog(N_ang_list, gaussl_err[0], 'g-s', mfc = 'none', label = 'Gauss-Legendre')
    # plt.loglog(N_ang_list[2:], err_estimate[2:], '-s', mfc = 'none')
    print(err_estimate)
    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.legend()
    plt.ylabel('RMSE', fontsize = 16)
    show_loglog(f'flux_converge', 1, N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)

    # plt.savefig(f'flux_converge_method={method}.pdf')
    plt.show()

    plt.figure('J')

    # plt.loglog(N_ang_list, cc_err[1], 'r--^', mfc = 'none',  label = 'left')
    # plt.loglog(N_ang_list, gauss_err[1], 'b--o', mfc = 'none',  label = 'left')
    plt.loglog(N_ang_list, cc_err[2], 'b-^', mfc = 'none',  label = 'Clenshaw-Curtis')
    plt.loglog(N_ang_list, gauss_err[2], 'r-o', mfc = 'none', label = 'Gauss-Lobatto')
    plt.loglog(N_ang_list, gaussl_err[2], 'g-s', mfc = 'none', label = 'Gauss-Legendre')
    plt.loglog(N_ang_list[1:], J_err_estimate_diff[1:], 'k--', label = 'difference')
    plt.loglog(N_ang_list[1:], J_err_estimate_lr[1:], 'k-.', label = 'power law')
    plt.loglog(N_ang_list[1:], J_err_estimate_rich[1:], 'k:', label = 'Richardson')
    plt.loglog(N_ang_list[2:],J_err_estimate_wynn[2:] , 'k-x', label = r'Wynn-$\epsilon$' )
    # plt.loglog(N_ang_list[4:], np.abs(tableauJcc[-1][5:,5] - Jb[1]) , '-s', label = r'Wynn-$\epsilon$' )
    plt.legend()
    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel(r'$|J_b-J_N|$', fontsize = 16)
    show_loglog(f'J_converge_method', 1, N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)
    # plt.savefig(f'J_converge_method={method}.pdf')
    plt.show()


    plt.figure('J error accuracy')
    # plt.loglog(N_ang_list, cc_err[1], 'r--^', mfc = 'none',  label = 'left')
    # plt.loglog(N_ang_list, gauss_err[1], 'b--o', mfc = 'none',  label = 'left')

    plt.loglog(N_ang_list[1:], np.abs(cc_err[2][1:]/J_err_estimate_diff[1:])**-1, 'k--', label = 'difference')
    plt.loglog(N_ang_list[1:], np.abs(cc_err[2][1:]/J_err_estimate_lr[1:])**-1, 'k-.', label = 'power law')
    plt.loglog(N_ang_list[1:], np.abs(cc_err[2][1:]/J_err_estimate_rich[1:])**-1, 'k:', label = 'Richardson')
    plt.loglog(N_ang_list[2:], np.abs(cc_err[2][2:]/np.abs(tableauJcc[-1][3:,3][-1] - tableauJcc[-1][1:,1][2:]))**-1 , 'k-x', label = r'Wynn-$\epsilon$' )
    plt.loglog(N_ang_list, np.ones(N_ang_list.size), 'k-')
    # plt.loglog(N_ang_list[4:], np.abs(tableauJcc[-1][5:,5] - Jb[1]) , '-s', label = r'Wynn-$\epsilon$' )
    plt.legend()
    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel(r'FOM', fontsize = 16)
    show_loglog(f'J_error_compare', 1, N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)
    # plt.savefig(f'J_converge_method={method}.pdf')
    plt.show()




    plt.figure('phi')
    plt.plot(cell_centersb, phib, 'k-')
    plt.xlabel(r'$x$ [cm]', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show('scalar_flux_3mat')
    # plt.plot(cell_centersb, np.abs(phicc - phib), '--')

    plt.show()

    plt.figure('reaction rate')

    plt.loglog(N_ang_list, np.abs(reaction_rate_bench-reaction_rate_cc), 'b-^', mfc = 'none', label = 'Clenshaw-Curtis')
    plt.loglog(N_ang_list, np.abs(reaction_rate_bench-reaction_rate_gauss), 'r-o', mfc = 'none', label = 'Gauss-Lobatto')
    plt.loglog(N_ang_list, np.abs(reaction_rate_bench-reaction_rate_gauss_l), 'g-s', mfc = 'none', label = 'Gauss-Legendre')
    plt.loglog(N_ang_list[1:], np.abs(reaction_err_estimate_diff[1:]), 'k--', label = 'difference' )
    plt.loglog(N_ang_list[1:], np.abs(reaction_rate_estimate_lr[1:]), 'k-.', label = 'power law')
    plt.loglog(N_ang_list[1:], np.abs(reaction_rate_estimate_rich[1:]), 'k:', label = 'Richardson')
    plt.loglog(N_ang_list[2:], reaction_rate_estimate_wynn[2:] , 'k-x', label = r'Wynn-$\epsilon$' )

    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel('reaction rate error', fontsize = 16)
    plt.legend()
    show_loglog('reaction_rate', 1,  N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)

    plt.figure('reaction rate error accuracy')

    plt.loglog(N_ang_list[1:], (np.abs(reaction_rate_bench-reaction_rate_cc)[1:]/np.abs(reaction_err_estimate_diff[1:]))**-1, 'k--', label = 'difference' )
    plt.loglog(N_ang_list[1:], (np.abs(reaction_rate_bench-reaction_rate_cc)[1:]/np.abs(reaction_rate_estimate_lr[1:]))**-1, 'k-.', label = 'power law')
    plt.loglog(N_ang_list[1:], (np.abs(reaction_rate_bench-reaction_rate_cc)[1:]/np.abs(reaction_rate_estimate_rich[1:]))**-1, 'k:', label = 'Richardson')
    plt.loglog(N_ang_list[2:], (np.abs(reaction_rate_bench-reaction_rate_cc)[2:]/np.abs(reaction_rate_tableau[3:,3][-1] - reaction_rate_tableau[1:,1][2:] ))**-1 , 'k-x', label = r'Wynn-$\epsilon$' )
    plt.loglog(N_ang_list, np.ones(N_ang_list.size), 'k-')

    plt.xlabel(r'$S_N$ order', fontsize = 16)
    plt.ylabel('FOM', fontsize = 16)
    plt.legend()
    show_loglog('reaction_rate_error_compare', 1,  N_ang_list[-1] * 1.1, choose_ticks=True, ticks = N_ang_list)


def estimate_error(ang_list, tableau):
    err_estimate = np.zeros(ang_list.size)
    for ia in range(2, ang_list.size):
        # print(tableau[1:, 1][0:ia])
        err_estimate[ia] = convergence_estimator(ang_list[0:ia], tableau[1:, 1][0:ia])
    return err_estimate


def nested_plot(mkrs=8, width = 5, height = 5):
    x1 = 2
    x2 = 6
    x3 = 16
    x4= 46
    g1 = quadrature(x1, 'gauss_lobatto')[0]
    g2 = quadrature(x2, 'gauss_lobatto')[0]
    g3 = quadrature(x3, 'gauss_lobatto')[0]
    g4 = quadrature(x4, 'gauss_lobatto')[0]

    cc1 = cc_quad(x1)[0]
    cc2 = cc_quad(x2)[0]
    cc3 = cc_quad(x3)[0]
    cc4 = cc_quad(x4)[0]

    y1 = 0
    y2 = 0.005
    y3 = 0.01
    y4 = 0.015
    # width = 5
    # height = 10
    # mkrs = 2
    plt.figure(1, figsize=(width, height)) 
    ax = plt.gca()
    
    plt.plot(g1, np.ones(g1.size)*y1, 'k.', markersize = mkrs)
    plt.plot(g2, np.ones(g2.size)*y2, 'k.', markersize = mkrs)
    plt.plot(g3, np.ones(g3.size)*y3, 'k.',markersize = mkrs)
    plt.plot(g4, np.ones(g4.size)*y4, 'k.',markersize = mkrs)
    positions = [y1, y2, y3, y4]
    labels = ['2', '6', '16', '46']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # ax.set_yticks([2,6,16,46])
    # ax.get_yaxis().set_visible(False)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('N', fontsize = 16)
    # ax.spines['left'].set_visible(False)
    plt.ylim(-0.01 + 0.008, 0.025-0.008)
    show('nested_quad_example_gs')
    plt.figure(2, figsize=(width, height)) 
    ax = plt.gca()
    plt.plot(cc1 , np.ones(cc1.size)*y1, 'k.', markersize = mkrs)
    plt.plot(cc2 , np.ones(cc2.size)*y2, 'k.',markersize = mkrs)
    plt.plot(cc3 , np.ones(cc3.size)*y3, 'k.',markersize = mkrs)
    plt.plot(cc4 , np.ones(cc4.size)*y4, 'k.',markersize = mkrs)
    # ax.set_yticks([2,6,16,46])
    # ax.get_yaxis().set_visible(False)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('N', fontsize = 16)
    positions = [y1, y2, y3, y4]
    labels = ['2', '6', '16', '46']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # ax.spines['left'].set_visible(False)
    plt.ylim(-0.01+0.008, 0.025-0.008)
    # plt.ylim(-0.02, 0.08)

    # plt.plot(cc2, np.ones(cc2.size) * 2, 'k-')
    show('nested_quad_example_cc')
    plt.show()

def check_quadratures():
     N_ang_list = np.array([2,6,16,46, 136, 406])
     for N in N_ang_list:
        # mus, ws = cc_quad(N)
        mus, ws = quadrature(N, 'gauss_legendre')
        ws = ws/2
        print(np.sum(ws), 'zeroth')
        print(np.sum(ws*mus), 'first')
        print(np.sum(ws*mus**2), 'second')



