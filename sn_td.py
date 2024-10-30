import sys 
from transport_sn import solve
import numpy as np
from sn_transport_functions import IC_class, mesh_class, scalar_flux_class,cc_quad
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from benchmarks import integrate_greens as intg
from scipy.interpolate import interp1d
import math



def timedep_solve(tf = 1.0, dt = 500,  N_cells = 25, N_ang = 136, left_edge = 'vacuum', right_edge = 'vacuum', IC = 'pl', source = 'off',
          opacity_function = 'constant', wynn_epsilon = True, L = 2.0, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0, strength = [1.0,0.0]):
    
    delta_t = tf/dt
    mesh_ob = mesh_class(N_cells, L, opacity_function)
    mesh_ob.make_mesh()
    mesh = mesh_ob.mesh
    mus, ws = cc_quad(N_ang)
    IC_ob = IC_class(N_ang, N_cells, IC, mesh)
    IC_ob.make_IC()

 

    phi_ob = scalar_flux_class(N_ang, N_cells, mesh, False)

    phi_ob.make_phi(IC_ob.angular_flux, ws)
    phi_list = np.zeros((dt, N_cells))
    tableau_list = np.zeros((dt, N_cells, 6,6))
    phi_list[0] = phi_ob.phi
    psi_old = IC_ob.angular_flux
    lstep = False
    for it in tqdm(range(1,dt)):
        if it == dt-1:
            lstep = True
        # print(it)
        # print(psi_old.size)
        psi, phi, cell_centers, mus, tableau = solve(N_cells = N_cells, N_ang = N_ang, left_edge = left_edge, right_edge = right_edge, IC = 'cold', source = 'input',
            opacity_function = opacity_function, wynn_epsilon = wynn_epsilon, laststep = lstep, L = L, tol = 1e-12, source_strength = 1.0, sigma_a = sigma_a, sigma_s = sigma_s, sigma_t = sigma_t + 1/delta_t,  strength = strength, maxits = 1e10, input_source = psi_old/delta_t )
        phi_list[it] = phi
        tableau_list[it] = tableau
        psi_old = np.copy(psi)
        # print(tableau.shape)
        # assert 0
    plt.ion()
    plt.plot(cell_centers, phi_list[-1], label = f'phi + {N_ang}')
    # plt.plot(cell_centers, tableau[:, 1, 1], label = 's2')
    if wynn_epsilon == True:
        plt.plot(cell_centers, tableau[ :, 1, 1],  label = 's2')
        plt.plot(cell_centers, tableau[:, 2, 1],  label = 's6')
        plt.plot(cell_centers, tableau[:, 3, 1],  label = 's16')
        plt.plot(cell_centers, tableau[:, 4, 1],  label = 's46')
        plt.plot(cell_centers, tableau[:, 5, 1], label = 's136')
    # plt.plot(cell_centers, tableau[:, 6, 1])
    plt.legend()
    plt.show()

    plt.figure(2)
    if wynn_epsilon == True:
        itt = -1

        plt.plot(cell_centers, phi_list[itt], label = f'phi + {N_ang}')
        plt.plot(cell_centers, tableau_list[itt, :, 1, 1],  label = 's2')
        plt.plot(cell_centers, tableau_list[itt, :, 2, 1],  label = 's6')
        
        plt.plot(cell_centers, tableau_list[itt, :, 3, 1],  label = 's16')
        plt.plot(cell_centers, tableau_list[itt, :, 4, 1],  label = 's46')
        plt.plot(cell_centers, tableau_list[itt, :, 5, 1], label = 's136')
        
        bench = intg.plane_IC(tf, 100)
        plt.plot(bench[0], bench[1] + bench[2], 'k-', label = 'benchmark')
        plt.legend()
        plt.show()

        plt.figure(3)
        ik = int(N_cells/2)
        plt.loglog(np.array([2,6,16,46,136]),tableau_list[itt, ik, 1:, 1],  label = 'Data')
        plt.loglog(np.array([136]), np.array([phi_list[itt,ik]]),  'x')
        plt.show()


        # plt.figure(4)
        bench_interpolated = interp1d(bench[0], bench[1]+bench[2])
        RMSE136 = math.sqrt(np.mean((bench_interpolated(np.abs(cell_centers)) - tableau_list[itt, :, 5, 1])**2))
        RMSEepsilon = math.sqrt(np.mean((bench_interpolated(np.abs(cell_centers)) - phi_list[itt])**2))
        print(RMSE136, 'phi 136 RMSE')
        print(RMSEepsilon, 'phi accelerated RMSE')


        



    


