import numpy as np
import math
from sn_transport_functions import scalar_flux_class, mesh_class, source_class, IC_class, cc_quad, sigma_class, boundary_class, mu_sweep
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def solve(N_cells = 500, N_ang = 136, left_edge = 'source1', right_edge = 'source1', IC = 'cold', source = 'off',
          opacity_function = 'constant', wynn_epsilon = False, laststep = False,  L = 5.0, tol = 1e-13, source_strength = 1.0, sigma_a = 0.0, sigma_s = 1.0, sigma_t = 1.0,  strength = [1.0,0.0], maxits = 1e10, input_source = np.array([0.0]), quad_type = 'cc'):
    # initialize mesh
    mesh_ob = mesh_class(N_cells, L, opacity_function)
    mesh_ob.make_mesh()
    mesh = mesh_ob.mesh
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
    # Initialize angular flux
    IC_ob = IC_class(N_ang, N_cells, IC, mesh)
    IC_ob.make_IC()
    angular_flux_IC = IC_ob.angular_flux
    # Initialize scalar flux class
    phi_ob = scalar_flux_class(N_ang, N_cells, mesh, False, quad_type= quad_type)
    # Initialize angles
    mus, ws = cc_quad(N_ang)
    # print(mus, 'mus')
    # print(mus, 'mus')

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
    
    while tolerance_achieved == False and iteration <= maxits:
        
        iteration += 1
        
        source_ob.make_source()

        s = source_ob.s
        

        for iang, mu in enumerate(mus):
            if source == 'input':
                snew = s[iang, :]
                # print(snew[99].size)
                # assert 0
            else:
                snew = s

                 
            psi[iang] = mu_sweep(N_cells, psi[iang], mu, sigma_t, sigma_s, mesh, snew, phi, boundary_ob)
        
        phi_ob.make_phi(psi, ws)
        phi = phi_ob.phi


        err = np.abs(phi_old - phi)
        max_err = np.max(err)
        phi_old = np.copy(phi)
        # print(iteration, ' iteration', max_err, ' maximum error')
        if max_err <= tol:
            tolerance_achieved = True
            if wynn_epsilon == True and laststep == True:
                phi_ob_we =  scalar_flux_class(N_ang, N_cells, mesh, True)
                phi_ob_we.make_phi(psi, ws)
                phi = phi_ob_we.phi
                tableau = phi_ob_we.tableau
                # print(phi-phi_old)
            else:
                tableau = phi_ob.tableau

    for iang, mu in enumerate(mus):
            if source == 'input':
                snew = s[iang, :]
                # print(snew[99].size)
                # assert 0
            else:
                snew = s
            
            psi[iang] = mu_sweep(N_cells, psi[iang], mu, sigma_t, sigma_s, mesh, snew, phi, boundary_ob)
    cell_centers = np.zeros(N_cells)
    for ix in range(N_cells):
        cell_centers[ix] = (mesh[ix+1] + mesh[ix])/2
    J = np.zeros(2)
    J[0] = phi_ob.J(psi[:,0])
    J[1] = phi_ob.J(psi[:,-1])
    return psi, phi, cell_centers, mus, tableau, J





def run_slab(N_ang = 136):
    plt.ion()
    plt.figure(1)
    psi, phi, centers, mus = solve(N_ang = N_ang)
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



    
    
    