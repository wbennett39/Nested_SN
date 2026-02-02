import numpy as np
import math
from sn_transport_functions import scalar_flux_class, mesh_class, source_class, IC_class, cc_quad, sigma_class, boundary_class, mu_sweep, calculate_psi_moments, legendre_difference, mu_sweep_sphere, Pn_scalar,calculate_psi_moments_DPN, legendre_difference_DPN, Pn_scalar_minus, Pn_scalar_plus, moment0_Legendre, moment0_Legendre_alphas, diverging_moments
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from functions import quadrature

def solve(N_cells = 500, N_ang = 136, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
          opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0,
        sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type = 'cc', geometry = 'slab', N_psi_moments = 2, ang_diff_type = 'diamond'):
    # initialize mesh
    mesh_ob = mesh_class(N_cells, L, opacity_function, geometry)
    mesh_ob.make_mesh()
    mesh = mesh_ob.mesh
    # print(mesh, 'mesh_edges')
    # print(N_ang, 'angles')
    # Initialize source
    source_ob = source_class(source, mesh,  input_source, source_strength)
    # initialize cross sections
    c = sigma_s/sigma_t
    sigma_ob = sigma_class(opacity_function, mesh, sigma_a, sigma_s, sigma_t)
    sigma_ob.make_sigma_a()
    sigma_ob.make_sigma_s()
    sigma_a = sigma_ob.sigma_a
    sigma_s = sigma_ob.sigma_s
    sigma_t = sigma_ob.sigma_t
    alphas = np.zeros(N_ang)

    if quad_type == 'cc':
        mus, ws = cc_quad(N_ang)
    elif quad_type == 'gauss':
        mus, ws = quadrature(N_ang, 'gauss_lobatto' )
    elif quad_type == 'gauss_legendre':
        mus, ws = quadrature(N_ang, 'gauss_legendre' )
    if geometry == 'sphere' and (ang_diff_type == 'diamond' or ang_diff_type =='SH' or ang_diff_type=='SHDPN') :
        mu_halfs = np.zeros(N_ang+1)
        mu_halfs[0] = -1
        mu_halfs[-1] = 1
        mu_halfs[1:-1] = 0.5 *(mus[1:] + mus[:-1])
        refl_index = N_ang -1
        if ang_diff_type == 'diamond':
            N_ang +=1
            refl_index = N_ang 
            new_angles = np.sort(np.append(mus.copy(), -1))
            new_weights = np.zeros(N_ang)
            new_weights[1:] = ws
            mus = new_angles
            ws = new_weights
            # print(mus, 'new angles')
            # print(ws, 'new weights')
        alphas = np.zeros(N_ang)
        for ia in range(1, alphas.size):
                alphas[ia] = alphas[ia-1] - mus[ia] * ws[ia] 
        print(np.sum(ws), 'should be 2')
    # Initialize angular flux
    IC_ob = IC_class(N_ang, N_cells, IC, mesh, mus)
    IC_ob.make_IC()
    angular_flux_IC = IC_ob.angular_flux
    # Initialize scalar flux class
    phi_ob = scalar_flux_class(N_ang, N_cells, mesh, False, quad_type= quad_type)
    # Initialize angles
    # Initialize boundaries
    boundary_ob = boundary_class(left_edge, right_edge, strength)
    # begin sweep
    psi = angular_flux_IC
    tolerance_achieved = False
    err = 1.0
    iteration = 0
    phi_ob.make_phi(psi, ws)
    phi = phi_ob.phi
    phi_old = np.copy(phi)
    count = 0
    cell_centers = np.zeros(N_cells)
    psiminus_mu = np.zeros(N_cells)
    for ix in range(N_cells):
        cell_centers[ix] = (mesh[ix+1] + mesh[ix])/2
    psi_moments = np.zeros((N_psi_moments, N_cells))
    psi_moments_half = np.zeros((N_psi_moments, N_cells))
    psi_momentsL = np.zeros((N_psi_moments, N_cells))
    psi_momentsR = np.zeros((N_psi_moments, N_cells))
    psiplus_origin = np.zeros(N_ang)
    psi_deriv_np = np.zeros(N_cells)
    psi_at_halfs = np.zeros((N_ang, N_cells))
    outer_tolerance_achieved = False
    outer_its = 0
    psi_moments_old = np.copy(psi_moments) + 10
    psi_new = psi.copy()
    psi_minus_half_mu = np.zeros((N_ang, N_cells))
    phi_old = np.copy(phi) * 0
    while outer_its < 1:
        iteration =0
        tolerance_achieved = False
        
        max_err =10
        err = 1
        for k in range(N_cells):
                        psi_moments[:, k] = calculate_psi_moments(N_psi_moments, psi[:,k], ws, N_ang, mus)
                        if ang_diff_type == 'SHDPN':
                            psi_momentsL[:, k], psi_momentsR[:, k] = calculate_psi_moments_DPN(N_psi_moments, psi[:,k], ws, N_ang, mus)
                            psi_moments_half[:, k] = calculate_psi_moments(N_psi_moments, psi_at_halfs[:,k], ws, N_ang, mus)
                        # psi_origin_moms = calculate_psi_moments(N_psi_moments, psiplus_origin, ws, N_ang, mus)
        outer_its += 1
        while tolerance_achieved == False:
        
            iteration += 1
            if iteration == maxits -1:
                print('max iterations')
            source_ob.make_source()

            s = source_ob.s
            if geometry =='sphere':
                ang_diff_term = np.zeros(N_cells)
                ang_diff_term2 = np.zeros((N_ang, N_cells))
                if ang_diff_type =='SH' or ang_diff_type == 'SHDPN' or ang_diff_type == 'diamond':

                    psi_for_moms = psi.copy()
                    # psi_for_moms[int(N_ang/2):,0] = psi_for_moms[:int(N_ang/2),0]
                    for k in range(N_cells):
                        psi_moments[:, k] = calculate_psi_moments(N_psi_moments, psi[:,k], ws, N_ang, mus)
                        psi_moments_half[:, k] = calculate_psi_moments(N_psi_moments, psi_at_halfs[:,k], ws, N_ang, mus)
                        psi_momentsL[:, k], psi_momentsR[:, k] = calculate_psi_moments_DPN(N_psi_moments, psi[:,k], ws, N_ang, mus)
                    # plt.figure('phi recon')
                    # psi_recon = np.zeros((N_ang, N_cells))
                    # for ik in range(N_cells):
                    #     for imom in range(N_psi_moments):
                    #         for l in range(N_ang):
                    #             psi_recon[l, ik] += 0.5 *(2 * imom + 1) * psi_moments[imom, ik] * Pn_scalar(imom, mus[l])
                    # for ikplot in range(N_cells):
                    #     if np.max(psi[:,ikplot])>1e-8:
                    #         plt.figure(f'{ikplot}')
                    #         plt.plot(mus, psi_recon[:, ikplot], '-o', mfc = 'none')
                    #         plt.plot(mus, psi[:, ikplot], 'k-')
                    #         plt.show()
                    # for ni in range(N_psi_moments):
                    #     if ni %2 != 0:
                    #         psi_origin_moms[ni] *= 0
                    # diff_term = 0

                    # for im, mmu in enumerate(mus):
                    #     for n in range(1, N_psi_moments):
                    #         diff_term += 0.5 * (2 * n+1) * psi_origin_moms[n] * Pn_scalar(n, mmu) * ws[im]
                    # psi_origin_moms[0] = old_flux - diff_term
    
                    # psiplus_origin *= 0
                    # for im, mmu in enumerate(mus):
                    #     for n in range(N_psi_moments):
                    #         psiplus_origin[im] += 0.5 * (2 * n+1) * psi_origin_moms[n] * Pn_scalar(n, mmu)

                    # print(current_center, 'current')
                    # np.testing.assert_allclose(abs(current_center), 1e-10, 1e-6, 1e-6)
                    # print(psiplus_origin, 'psi at r=0')

                    psi_grad_at_0 = np.gradient(psiplus_origin, mus)
            for iang, mu in enumerate(mus):
                if source == 'input':
                    snew = s[iang, :]
                else:
                    snew = s
                psiplusright = 0
                psiminusleft = 0 
                if mu >0:
                    if geometry == 'sphere' and ang_diff_type == 'diamond':
                        psiminusleft = psi[refl_index - iang, 0]
                        assert abs(mus[refl_index-iang]) == abs(mus[iang])
                    elif geometry == 'sphere' and (ang_diff_type == 'SH' or ang_diff_type =='SHDPN'):
                            refl_index = N_ang -1
                            # print(mus[refl_index])
                            # print(mus[iang])
                            # print(mus[refl_index+1])
                            # assert 0
                            assert abs(mus[refl_index-iang]) == abs(mus[iang])
                            psiminusleft = psi[refl_index-iang, 0] 
                    else:
                            psiminusleft = boundary_ob('left', mu)
                elif mu <0:
                    psiplusright = boundary_ob('right', mu)
                    if right_edge == 'reflecting':
                        psiplusright = psi[N_ang - iang-1, -1]
                if geometry == 'sphere' and (ang_diff_type =='SH' or ang_diff_type == 'SHDPN'):
                    # print(phi -0.5*psi_moments[0,:], 'should be zero')
                    psi_deriv_np = np.zeros(N_cells)
                   
                    for ik in range(N_cells):
                            psi_deriv_np[ik] = np.gradient((1-mus**2)*psi[:, ik], mus)[iang]
                    for k in range(N_cells):
                        if ang_diff_type == 'SH':
                            # psi_moments = diverging_moments(psi_moments)
                            psi_moments[1:, 0] *=0
                            ang_diff_term[k] = legendre_difference(N_psi_moments, psi_moments[:,k], mu) 
                            ang_diff_term2[iang, k] = ang_diff_term[k].copy()
                        elif ang_diff_type == 'SHDPN':
                            psi_momentsL = diverging_moments(psi_momentsL)
                            psi_momentsR = diverging_moments(psi_momentsR)
                            # psi_momentsR[1:, 0] *=0
                            psi_momentsL[1:,0] = psi_momentsR[1:,0]
                            ang_diff_term[k] = legendre_difference_DPN(N_psi_moments, psi_momentsL[:,k], psi_momentsR[:,k], mu) 

                        zero_mom = 0.0
                        first_mom = 0.0
                        for n in range(N_psi_moments):
                            for im, mmu in enumerate(mus):
                                zero_mom += ws[im] * psi_moments[n,k]  * (2 * n+1) * 0.5 * (mmu * (n-1) * Pn_scalar(n, mmu, -1,1) - (n+1) * Pn_scalar(n+1, mmu, -1,1))
        
                                first_mom += ws[im] * mmu * psi_moments[n,k] * (2 * n+1) * 0.5  * (mmu * (n-1) * Pn_scalar(n, mmu, -1,1) - (n+1) * Pn_scalar(n+1, mmu, -1,1)) 
                        # if ang_diff_type == 'SH':
                            # np.testing.assert_allclose(abs(zero_mom), 0,rtol =1e-12, atol =1e-6)
                            # if N_psi_moments >=3:
                            #     # print(first_mom / (-4/3 * psi_moments[0, k]/2 + 4/ 15 * psi_moments[2,k] * 5/2 + 1e-12), 'first mom')
                            #     # print(k, 'k')
                            #     # print(first_mom)
                            #     # print(-4/3 * psi_moments[0, k]/2 + 4/ 15 * psi_moments[2,k] * 5/2)
                            #     np.testing.assert_allclose(first_mom, (-4/3 * psi_moments[0, k]/2 + 4/ 15 * psi_moments[2,k] * (5/2)),rtol =  1e-6, atol = 1e-6)

                        plot_ang_sol = False
                        if plot_ang_sol == True:
                            plt.ioff()

                            psi_deriv = np.gradient((1-mus**2)*psi[:, k], mus)
                            # psi_deriv = mus
                            psi_moments_grad = calculate_psi_moments(N_psi_moments, psi_deriv, ws, N_ang, mus)
                            psi_moments_gradL, psi_moments_gradR = calculate_psi_moments_DPN(N_psi_moments, psi_deriv, ws, N_ang, mus)
                            psi_deriv_grad = psi_deriv * 0
                            Pnvec = mus*0
                            PnvecL = mus* 0
                            PnvecR = mus*0
                            if ang_diff_type == 'SH':
                                for n in range(N_psi_moments):
                                    for im in range(mus.size):
                                            Pnvec[im] = Pn_scalar(n, mus[im])

                                    psi_deriv_grad += (2 * n +1) / 2. * psi_moments_grad[n] * Pnvec
                            
                            elif ang_diff_type == 'SHDPN':
                                for n in range(N_psi_moments):
                                    for im in range(mus.size):
                                            if mus[im] < 0:
                                                PnvecL[im] = Pn_scalar_minus(n, mus[im])
                                            else:
                                                PnvecR[im] = Pn_scalar_plus(n, mus[im])


                                    psi_deriv_grad += (2 * n +1) * psi_moments_gradL[n] * PnvecL +  (2 * n +1) * psi_moments_gradR[n] * PnvecR


                            mu_analytic = np.zeros(mus.size)
                            mu_analytic2 = np.zeros(mus.size)
                            for im, mmu in enumerate(mus):
                                # if ang_diff_type == 'SH':
                                    mu_analytic2[im] = legendre_difference(N_psi_moments, psi_moments[:,k], mmu)
                                # elif ang_diff_type == 'SHDPN':
                                    mu_analytic[im] = legendre_difference_DPN(N_psi_moments, psi_momentsL[:,k], psi_momentsR[:, k], mmu)

                            if np.max(np.abs(mu_analytic) > 0.001):
                
                                plt.figure('angular flux')
                                plt.plot(mus, mu_analytic, label = 'analytic gradient DPN')
                                plt.plot(mus, mu_analytic2, label = 'analytic gradient PN')

                                plt.plot(mus, psi_deriv_grad, 'o', mfc = 'none', label = f'{ang_diff_type} expansion of gradient term')
        
                                plt.plot(mus, psi_deriv, 'k--', label = 'gradient term')
                                plt.xlabel(r'$\mu$', fontsize = 16)
                                plt.ylabel(r'$\partial_\mu(1-\mu^2) \: \psi$')
                                plt.legend()
                                if k < mesh.size:
                                    plt.title(f'x = {0.5*(mesh[k] + mesh[k+1])}')
                                plt.show()



         
                if geometry == 'sphere':
                    if iang >0:
                        alphaplus = alphas[iang]
                        alphaminus = alphas[iang-1]

                    else:
                        alphaplus = alphas[iang]
                        alphaminus = 0
        
                    # print(mu_halfs[iang], 'mu half')

    

                    psi[iang], psiminus_mu_s, psiplus_origin[iang], psi_at_halfs[iang] = mu_sweep_sphere(N_cells, psi[iang], mu,  ws[iang], psiminus_mu, alphaplus, alphaminus, sigma_t, sigma_s, mesh, snew, phi, psiplusright, ang_diff_term, psi_at_halfs[iang], ang_diff_type,  psiplus_origin[refl_index-iang-1])

                    # psiplus_origin *=0
                    if iang == 0:
                        psiminus_mu = psiminus_mu_s
                        psi_origin = psiminus_mu.copy()[0]
                        
                    else:
                        psiminus_mu[0:] = 2 * psi[iang][:] - psiminus_mu.copy()[:]
                    psiplus_origin[:] = psi_origin
                    psi_minus_half_mu[iang, :] = psiminus_mu.copy()
                    # psi[iang] = psi_new[iang]
                        # psiminus_mu[0] = psiminus_mu_s[0]

                    # if ang_diff_type =='diamond':
                    # if iang == 0:
                    #     psiminus_mu = psi[iang]
                    # else:    
                    #     psiminus_mu = 2 * psi[iang] - psiminus_mu.copy()
                else:
                    psi[iang] = mu_sweep(N_cells, psi[iang], mu, sigma_t, sigma_s, mesh, snew, phi, psiminusleft, psiplusright, geometry)
            # for k in range(N_cells):
            #     print(np.sum(ang_diff_term2[1:, k] * (mu_halfs[1:] - mu_halfs[:-1])), 'ang diff sum 0')
        
            # print(ang_diff_term2)
            # assert 0
                
            phi_ob.make_phi(psi, ws)
            phi = phi_ob.phi
            if ang_diff_type == 'SHDPN' or ang_diff_type =='SH':
                if plot_ang_sol == True:
                    plt.ioff() 
                    plt.figure('scalar flux')

                    plt.plot(cell_centers, phi)
                    plt.show()
                    plt.figure('angular flux derivative')
                    psi_grad_at_0 = np.gradient(psiplus_origin, mus)
                    plt.plot(mus, psi_grad_at_0, '-')
                    plt.show()
                    print(psi_grad_at_0, 'gradient')
            if np.isnan(phi).any():
                raise ValueError('nan phi')

            if ang_diff_type == 'diamond':
                err = np.abs(phi_old - phi)
            else:
                 err = np.abs(psi_moments_old - psi_moments)

            max_err = np.max(err)
            max_err_loc = np.argmin(np.abs(max_err - err ))

            phi_old = np.copy(phi)
            psi_moments_old = np.copy(psi_moments)
            count += 1
            # print(iteration, ' iteration', max_err, ' maximum error')
            if count  == 1000:

                print(iteration, ' iteration', max_err, ' maximum error', cell_centers[max_err_loc], 'max err x location' )
                print(phi, 'phi')
                count = 0
            if max_err <= tol or iteration >= maxits:
                print(iteration, 'iterations required to converge')
                tolerance_achieved = True
                if wynn_epsilon == True and laststep == True:
                    phi_ob_we =  scalar_flux_class(N_ang, N_cells, mesh, True, quad_type)
                    phi_ob_we.make_phi(psi, ws)
                    phi = phi_ob_we.phi
                    tableau = phi_ob_we.tableau 
                    tableau_J = phi_ob_we.tableauJp
                    # print(phi-phi_old)
                else:
                    tableau = phi_ob.tableau
                    tableau_J = phi_ob.tableauJp

    if geometry =='sphere':


        if ang_diff_type == 'SH' or ang_diff_type == 'SHDPN' or ang_diff_type=='diamond':
            for k in range(N_cells):
                if ang_diff_type == 'SH' or ang_diff_type =='diamond':
                    # psi[int(N_ang/2):,0] = psi[:int(N_ang/2),0]
                    psi_moments[:, k] = calculate_psi_moments(N_psi_moments, psi[:,k], ws, N_ang, mus)
                
                elif ang_diff_type == 'SHDPN':
                    psi_momentsL[:, k], psi_momentsR[:, k] = calculate_psi_moments_DPN(N_psi_moments, psi[:,k], ws, N_ang, mus)
        for imn in range(N_psi_moments):
             if imn %2 != 0:
                  psi_moments[imn, 0] *=0
        print(psi_moments[:,0])


    psir0 = psi[0,:]
    for iang, mu in enumerate(mus):
            if source == 'input':
                snew = s[iang, :]
                # print(snew[99].size)
                # assert 0
            else:
                    snew = s
            psiplusright = 0
            psiminusleft = 0 

            if mu >0:
                psiminusleft = boundary_ob('left', mu)
                if geometry == 'sphere' and ang_diff_type == 'diamond':
                    psiminusleft = psi[N_ang - iang, 0]
                    # psiminus_mu[int(N_ang/2):] = psiminus_mu[:int(N_ang/2)]

      
                elif geometry == 'sphere' and (ang_diff_type == 'SH' or ang_diff_type == 'SHDPN'):
                    #  psiminusleft = psi[N_ang - iang-1, 0]
                    # refl_index = N_ang - iang - 1
         
                    assert abs(mus[refl_index-iang]) == abs(mus[iang])

            elif mu <0:
                psiplusright = boundary_ob('right', mu)
            if geometry == 'sphere':
                
                if iang >0:
                    alphaplus = alphas[iang]
                    alphaminus = alphas[iang-1]

                else:
                    alphaplus = 0
                    alphaminus = 0

                if ang_diff_type == 'SH' or ang_diff_type == 'SHDPN' or ang_diff_type == 'diamond':
                    # psir0 = 
                    for k in range(N_cells):
                        if ang_diff_type == 'SH':
                            psi_moments = diverging_moments(psi_moments)
                            psi_moments[1:, 0] *=0
                            
                            ang_diff_term[k] = legendre_difference(N_psi_moments, psi_moments[:,k], mu) 
                            # ang_diff_term[k] = moment0_Legendre(iang, mus, N_psi_moments, psi_moments[:, k], N_ang, psir0[k])
                            # ang_diff_term[k] = 1/ws[iang] * (alphaplus * psiplus - alpham * psiminus)
                            # ang_diff_term[k] = 2/ws[iang] * psi[iang, k] * alphaplus -1/ws[iang] * (alphaplus + alphaminus) * psi_minus_half_mu[iang, k]
                  

                            # ang_diff_term[k] = moment0_Legendre_alphas(iang, mus, N_psi_moments, psi_moments[:,k], N_ang, psir0[k], ws[iang], alphas[iang], alpham)

                            alpham = 0
                            if iang>0:
                                alpham = alphas[iang-1]
                            # ang_diff_term[k] = moment0_Legendre_alphas(iang, mus, N_psi_moments, psi_moments[:,k], N_ang, psir0[k], ws[iang], alphas[iang], alpham)
                            # ang_diff_term2[:, k] = moment0_Legendre(mu_halfs, N_psi_moments, psi_moments[:, k], N_ang)
                        elif ang_diff_type == 'SHDPN':
                            psi_momentsL = diverging_moments(psi_momentsL)
                            psi_momentsR = diverging_moments(psi_momentsR)
                            # psi_momentsR[1:, 0] *=0
                            psi_momentsL[1:,0] = psi_momentsR[1:,0]
                            ang_diff_term[k] = legendre_difference_DPN(N_psi_moments, psi_momentsL[:,k], psi_momentsR[:,k], mu)

                    # zero_mom = 0.0
                    # first_mom = 0.0
                    # for n in range(N_psi_moments):
                    #     for im, mmu in enumerate(mus):
                    #         zero_mom += ws[im] * psi_moments[n,k] * (2 * n+1) * 0.5  * (mmu * (n-1) * Pn_scalar(n, mu, -1,1) - (n+1) * Pn_scalar(n+1, mu, -1,1))
                    #         fist_mom += ws[im] * mmu * psi_moments[n,k] * (2 * n+1) * 0.5  * (mmu * (n-1) * Pn_scalar(n, mu, -1,1) - (n+1) * Pn_scalar(n+1, mu, -1,1)) 
                    # print(zero_mom, 'should be zero')
                    # print(first_mom - (-4/3 * psi_moments[0, k]) + 4/ 15 * psi_moments[2,k], 'should be zero')

                

            
                psi[iang], psiminus_mu_s, psiplus_origin[iang], psi_at_halfs[iang] = mu_sweep_sphere(N_cells, psi[iang], mu,  ws[iang], psiminus_mu, alphaplus, alphaminus, sigma_t, sigma_s, mesh, snew, phi, psiplusright, ang_diff_term, psi_at_halfs[iang],ang_diff_type,  psiplus_origin[refl_index-iang-1])
                if iang == 0:
                        psiminus_mu = psiminus_mu_s
                        psi_origin = psiminus_mu.copy()[0]
                        
                else:
                        psiminus_mu[0:] = 2 * psi[iang][:] - psiminus_mu.copy()[:]
                psiplus_origin[:] = psi_origin
                psi_minus_half_mu[iang, :] = psiminus_mu.copy()
                #   psiminus_mu = psi[iang]
                # else:
                #     psiminus_mu = 2 * psi[iang] - psiminus_mu.copy()
            else:
                psi[iang] = mu_sweep(N_cells, psi[iang], mu, sigma_t, sigma_s, mesh, snew, phi, psiminusleft, psiplusright, geometry)
    phi_ob.make_phi(psi, ws)
    phi = phi_ob.phi

    J = np.zeros(2)
    J[0] = phi_ob.J(psi[:,0])
    J[1] = phi_ob.J(psi[:,-1])

    return psi, phi, cell_centers, mus, tableau, J, tableau_J, np.array([sigma_a, sigma_s])

# Olson-Henderson shell source problem
def run_sphere(N_ang = 8, N_mom = 4):
    plt.ion()
    plt.figure(1)
    psi, phi, centers, mus, tableau, J, tableau_J, sigmas = solve(N_ang = N_ang, geometry='sphere', source = 'shell_OH', L = 10, sigma_s = 0.9, sigma_t= 1.0, N_psi_moments=N_mom)
    plt.plot(centers, phi, '-')
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    # plt.ylim(0, 1.5)
    plt.show()
   

def run_slab(N_ang = 136):
    plt.ion()
    plt.figure(1)
    psi, phi, centers, mus, tableau, J, tableau_J, sigmas = solve(N_ang = N_ang)
    plt.plot(centers, phi, '-')
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    # plt.ylim(0, 1.5)
    psibench = lambda x, mu: np.exp(-(2.5 + x)/mu)
    err = 0
    for im in range(int(mus.size/2)+1, mus.size):
        # print(mus[im])
        # print(psibench(centers, mus[im]-psi[im, :]))
        err += math.sqrt(np.mean((psibench(centers, mus[im])-psi[im, :]))**2) / (N_ang/2)
    print(err, 'err')

    # mu_test = mus[34]
    # print(mu_test)
    # plt.figure(2)
    # plt.plot(centers, psibench(centers, mu_test), 'k-')
    # plt.plot(centers, psi[50, :], 'x')
    # plt.show()

    
    # plt.show()


def run_siewert(N_cells = 400, N_ang = 136, wynn_epsilon= True):
    psi, phi, centers, mus = solve(N_cells, N_ang, L = 5, c = 1.0, wynn_epsilon=wynn_epsilon)
    
    # phi_we = 
    interp_psi = interp1d(mus, psi[:,0])
    interp_psi_right = interp1d(mus, psi[:,-1])

    siewert_sinf = np.array([[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                             [0.89780, 0.88784, 0.86958, 0.85230, 0.83550, 0.81900, 0.80278, 
                                    0.78649, 0.77043, 0.75450, 0.73872] ])
    siewert_sinf_right = np.array([0.10220, 0.11216, 0.13042, 0.14770, 0.16450, 0.18100, 0.19732, 
                                    0.21351, 0.22957, 0.24550, 0.26128])

    psi_interp = interp_psi(-siewert_sinf[0])
    psi_interp_right = interp_psi_right(siewert_sinf[0])

    err = math.sqrt(np.mean((psi_interp-siewert_sinf[1])**2))
    err2 = math.sqrt(np.mean((psi_interp_right-siewert_sinf_right)**2))
    
    

    # plt.ion()
    # # plt.plot(-siewert_sinf[0], siewert_sinf[1], 'x')
    # # plt.plot(mus, psi[:,0], label = f'{N_ang}')
    # # plt.plot(-siewert_sinf[0], psi_interp-siewert_sinf[1], label = f'{N_ang}')
    # plt.legend()
    # plt.xlim(-1,0)
    # plt.show()

    plt.figure(1)
    plt.ion()
    plt.plot(centers, phi)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    plt.show()
    
    return err, err2

def siewert_converge():
    angles_list = np.array([2,4,6,8])
    RMS =[] 
    RMS2 = []

    for i, ang in enumerate(angles_list):
        res1, res2 = run_siewert(N_ang = ang)
        RMS.append(res1)
        RMS2.append(res2)

    print(RMS)
    print(angles_list)
    plt.figure(2)
    plt.ion()
    plt.loglog(angles_list, np.array(RMS), '-o', label = 'left side')
    plt.loglog(angles_list, np.array(RMS2), '-o', label = 'right side')
    plt.legend()
    plt.show()


# siewert_converge()
# run_slab()
# run_siewert()
def siewert_we_test():
    cc = 0.7
    psi_b, phi_b, centers_b, mus_b,tb1 = solve(100, 64, L = 5, wynn_epsilon=False)

    psi, phi, centers, mu,tbs = solve(100,46, L = 5, wynn_epsilon=False)
    psiw, phiw, centersw, musw,tb = solve(100, 46, L = 5, wynn_epsilon=True)
    phi_we = np.zeros(100)
    for k in range(100):
        phi_we[k] = tb[k,3:,3][-1]

    err1 = math.sqrt(np.mean((phi_b-phi_we)**2))
    err2 = math.sqrt(np.mean((phi_b-phi)**2))
    print(err1, 'err with we')
    print(err2, 'err without we')
    plt.figure(1)
    plt.plot(centersw, phi_b-phi, 'k-', label = 'without wynn epsilon')
    # plt.plot(centersw, phi, 'k-', label = 'without wynn-epsilon')
    plt.plot(centers_b, phi_b - phi_we, 'rx', label = 'with wynn epsilon')
    plt.legend()
    plt.show()

    plt.figure(2)
    for k in range(1, 2):

        plt.loglog(np.array([16,46]), np.abs(phi_b[k] - tb[k,3:,3]), '-o', mfc = 'none')
        plt.loglog(np.array([2,6,16,46]), np.abs(phi_b[k] - tb[k,1:,1]), '-^', mfc = 'none')
        plt.loglog(np.array([46]), np.abs(phi_b[k] - phi[k]), '-s', mfc = 'none')

    plt.show()



# siewert_we_test()



    
    
    