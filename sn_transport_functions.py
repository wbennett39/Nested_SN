from typing import Any
import numpy as np
from chaospy.quadrature import clenshaw_curtis
import math
from functions import quadrature
from scipy.interpolate import interp1d
from numba.extending import get_cython_function_address
import ctypes 
import scipy.integrate as integrate
from numba import njit
def cc_quad(N):
    x, w= clenshaw_curtis(N-1,(-1,1))
    return x[0], w
    # res = quadpy.c1.gauss_lobatto(N)
    # x = res.points
    # w = res.weights
    # return x, w

    


class scalar_flux_class:
    def __init__(self, N_ang, N_cells, mesh, wynn_epsilon, quad_type = 'cc'):
        self.N_ang = N_ang
        self.N_cells = int(N_cells)
        self.mesh = mesh
        self.wynn_epsilon = wynn_epsilon
        ns_list = np.array([2,6,16,46,136, 406, 1216, 3646])
        narg = np.argmin(np.abs(ns_list-N_ang))
        self.ns_list = ns_list[0:narg+1] 
        if wynn_epsilon == True:
            if self.ns_list[-1] != self.N_ang:
                assert 0 
        self.w_mat = np.zeros((self.ns_list.size, self.ns_list[-1] ))

        self.xs_mat = np.zeros((self.ns_list.size, self.ns_list[-1] ))
        self.index_mat = np.zeros((self.ns_list.size, self.N_ang ))
        self.nested_phi = np.zeros((self.ns_list.size, self.N_cells))
        self.nested_J = np.zeros((self.ns_list.size, self.N_cells))
        self.tableau = np.zeros((self.N_cells, self.ns_list.size+1,self.ns_list.size+1))
        self.tableauJp = np.zeros((self.N_cells, self.ns_list.size+1,self.ns_list.size+1))
        # store all Clenshaw-Curtis weights if WE acceleration is activated
       
        
        self.quad_type = quad_type
        self.weights_matrix()
        self.phi = np.zeros(self.mesh.size-1)
        if quad_type == 'cc':
            print('Clenshaw-Curtis quadrature')
            self.mus, self.ws = cc_quad(self.N_ang)
            print(self.mus)
            
        elif quad_type == 'gauss':
            print('Gaussian quadrature')
            self.mus, self.ws = quadrature(self.N_ang, 'gauss_lobatto')
        # print(self.mus, self.ws, 'mus, ws')
        elif quad_type == 'gauss_legendre':
            print('Gauss-Legendre quadrature')
            self.mus, self.ws = quadrature(self.N_ang, 'gauss_legendre')
        
        
        # print(self.ws, 'weights')
    
    def weights_matrix(self):
        
        for i in range(self.ns_list.size):
            if self.quad_type == 'cc':
                self.w_mat[i, 0:self.ns_list[i]] = cc_quad(self.ns_list[i])[1]
                self.xs_mat[i, 0:self.ns_list[i]] = cc_quad(self.ns_list[i])[0]
            elif self.quad_type == 'gauss':
                self.xs_mat[i, 0:self.ns_list[i]], self.w_mat[i, 0:self.ns_list[i]] = quadrature(self.ns_list[i], 'gauss_lobatto')
            igrab = False
            ig = 0
        # print(self.w_mat[0])
    
    def make_lower_order_fluxes(self, psi):
        psi_new = []
        xs_test = []
        count = 1
        psi_new.append(psi[0])
        xs_test.append(self.mus[0])
        if psi.size == 6:
            psi_new = np.array([psi[0], psi[-1]])
        else:
            for ix in range(1,psi.size):
                if count%3 == 0:
                    psi_new.append(psi[count])
                    xs_test.append(self.mus[count])
                    # count = 0
                count += 1
        # xs_test.append(self.mus[-1])
        xs_test = np.array(xs_test)
        # print(xs_test.size, 'xs size')
        # psi_new.append(psi[-1])
        # print(xs_test, 'xs')
        return np.array(psi_new)
    
        
    
    def make_phi(self, psi, ws):
        
            for k in range(self.N_cells):
                # print(psi[:,k])
                # print(self.ws)
                # self.phi[k] = self.quadrature(psi[:,k], self.ws)
                self.phi[k] = np.sum(psi[:,k]*ws)*0.5
                if self.wynn_epsilon == True:
                    # for n in range(self.ns_list.size):
                        # print(n, 'n')
                    self.nested_phi[:, k], self.nested_J[:,k] = self.make_nested_phi(psi[:,k])
                    tableau = self.wynn_epsilon_algorithm(self.nested_phi[:,k])
                    tableau_J = self.wynn_epsilon_algorithm(self.nested_J[:,k])
                    
                    # self.phi[k] = tableau[3:,3]
                    # print(tableau[3:,3][0], 'we phi')
                    # print(self.phi[k], 'phi')
                    
                    self.tableau[k, :] = tableau
                    self.tableauJp[k,:] = tableau_J
                    # print(self.tableau[k,1:,1][-1]-self.phi[k])
   
                    # self.phi[k] = tableau[3:,3][-1] 

        # assert(np.abs(np.sum(self.ws)-2)<=1e-10)
        # print(np.abs(np.sum(self.ws)-2))
        # print(self.ws, 'ws')


    def make_nested_phi(self, psi):
        phi_list = np.zeros(self.ns_list.size)
        Jp_list = np.zeros(self.ns_list.size)
        # self.make_phi(psi, self.w_mat[-1])
        # phi = np.sum(psi[:]*self.w_mat[-1])*0.5
        phi = np.sum(psi[:]*self.ws)*0.5
        J = np.sum(psi[:]*self.ws*self.mus)*0.5


        phi_list[-1] = phi
        Jp_list[-1] = J
        psi_old = psi 
        for ix in range(2, self.ns_list.size+1):
            # print(ix)
            psi_lower = self.make_lower_order_fluxes(psi_old)
            
            # print(psi_lower.size, 'psi l')
            # print(self.w_mat[-ix-1, 0:self.ns_list[-ix-1]])
            # phi_list[-ix] = self.make_phi(psi_lower, self.w_mat[-ix, 0:self.ns_list[ix]])
            phi_list[-ix] = np.sum(psi_lower[:]*self.w_mat[-ix, 0:self.ns_list[-ix]])*0.5
            Jp_list[-ix] = np.sum(psi_lower[:]*self.w_mat[-ix, 0:self.ns_list[-ix]] * self.xs_mat[-ix, 0:self.ns_list[-ix]])*0.5

            # print(self.ns_list[-ix-1])
            # print(self.w_mat[-ix-1, 0:self.ns_list[-ix-1]], 'ws')
        
            

            psi_old = psi_lower

        # print(phi_list[-1]-phi)
        # print(len(phi_list))
        return np.array(phi_list), np.array(Jp_list)

    def  wynn_epsilon_algorithm(self, S):
        n = S.size
        width = n-1
        # print(width)
        tableau = np.zeros((n + 1, width + 2))
        tableau[:,0] = 0
        tableau[1:,1] = S.copy() 
        for w in range(2,width + 2):
            for r in range(w,n+1):
                #print(r,w)
                # if abs(tableau[r,w-1] - tableau[r-1,w-1]) <= 1e-15:
                #     print('potential working precision issue')
                tableau[r,w] = tableau[r-1,w-2] + 1/(tableau[r,w-1] - tableau[r-1,w-1])
        return tableau
    
    def J(self, psi):
        return np.sum(psi * self.mus* self.ws) * 0.5


    def quadrature(psik, w):
        return np.sum(psik*w)
    


class sigma_class:
    def __init__(self, opacity_function, mesh, sigma_a, sigma_s, sigma_t):
        self.opacity_function = opacity_function
        self.mesh = mesh
        # self.c = c
        self.sigma_a_bar = sigma_a
        self.sigma_t_bar = sigma_t
        self.sigma_s_bar = sigma_s
    def make_sigma_a(self):
        if self.opacity_function == 'constant':
            self.sigma_a = np.ones(self.mesh.size-1) * self.sigma_a_bar
        elif self.opacity_function == '3_material':
            self.sigma_a = np.zeros(self.mesh.size-1)
            for it in range(self.mesh.size-1):
                if self.mesh[it] < -0.5:
                    self.sigma_a[it] = 0.1
                elif -0.5 <= self.mesh[it] <=0.5:
                    self.sigma_a[it] = 0.4
                elif self.mesh[it] > 0.5:
                    self.sigma_a[it] = 0.1
        elif self.opacity_function == 'larsen':
            self.sigma_a = np.zeros(self.mesh.size-1)
            for it in range(self.mesh.size-1):
                
                    for it in range(self.mesh.size-1):
                        if -5.5 <= self.mesh[it] < -4.5:
                            self.sigma_a[it] = 2.0
                        else:
                            self.sigma_a[it] = 0.0
                            
        else:
            assert 0 

    def make_sigma_s(self):
        if self.opacity_function == 'constant':
            self.sigma_s = np.ones(self.mesh.size-1) * self.sigma_s_bar
            self.sigma_t = self.sigma_t_bar*  np.ones(self.mesh.size-1) 
        elif self.opacity_function == '3_material':
            self.sigma_t = self.sigma_t_bar*  np.ones(self.mesh.size-1) 
            self.sigma_s = np.zeros(self.mesh.size-1) 
            for it in range(self.mesh.size-1):
                if self.mesh[it] < -0.5:
                    self.sigma_s[it] = 0.9
                elif -0.5 <= self.mesh[it] <=0.5:
                    self.sigma_s[it] = 0.6
                elif self.mesh[it] > 0.5:
                    self.sigma_s[it] = 0.9
        elif self.opacity_function == 'larsen':
            self.sigma_t = np.zeros(self.mesh.size-1)  
            self.sigma_s = np.zeros(self.mesh.size-1) 
            for it in range(self.mesh.size-1):
                if -5.5 <= self.mesh[it] < -4.5:
                    self.sigma_t[it] = 2.0
                else:
                    self.sigma_s[it] = 100.0
                    self.sigma_t[it] = 100.0

        else: 
            assert 0 



class mesh_class:
    def __init__(self, N_cells, L, opacity_function, geometry = 'slab'):
        self.N_cells = N_cells
        self.opacity_function = opacity_function
        self.L = L
        self.geometry = geometry
    def make_mesh(self):
        if self.opacity_function == 'constant':
            if self.geometry == 'sphere':
                self.mesh = np.linspace(0, self.L, self.N_cells+1)
            else:
                self.mesh = np.linspace(-self.L/2, self.L/2, self.N_cells+1)

        elif self.opacity_function == '3_material':
            third = int(2*int(self.N_cells + 1)/5)
            rest = int(self.N_cells+1-2*third)
            N = rest-1
            x1 = (N/(2*(1-N)))

            # print(np.linspace(-0.5-dx, 0.5+dx, rest)[1:]-np.linspace(-0.5-dx, 0.5+dx, rest)[:-1]) 
            dx = 0.125
            # self.mesh = np.concatenate((np.linspace(-self.L/2, x1, third + 1)[:-1], np.linspace(x1, -x1, rest), np.linspace(x1, self.L/2, third+1)[1:]))
            # print(self.mesh)
            self.mesh = np.concatenate((np.linspace(-self.L/2, -0.5, third + 1)[:-1], np.linspace(-0.5, 0.5, rest), np.linspace(0.5, self.L/2, third+1)[1:]))
            assert self.mesh.size == self.N_cells +1
            # assert (self.mesh == -0.5).any()
            # assert (self.mesh == 0.5).any()
            cell_centers = np.copy(self.mesh*0)
            # print(cell_centers)
            for ix in range(cell_centers.size-1):
                cell_centers[ix] = (self.mesh[ix+1] + self.mesh[ix])/2

            # assert (cell_centers == -0.5).any()
            # assert (cell_centers == 0.5).any()
        elif self.opacity_function == 'larsen':
            third = int(int(self.N_cells + 1)/2)
            rest = int(self.N_cells+1-third)
            N = rest-1
            self.mesh = np.concatenate((np.linspace(-self.L/2, -4.5, third )[:], np.linspace(-4.5, self.L/2, rest+1)[1:]))
            
            # assert (self.mesh == -0.5).any()
            # assert (self.mesh == 0.5).any()

            # print(cell_centers)

            # absorbing = 10
            # rest = self.N_cells + 1 - 10
            # self.mesh = np.concatenate((np.linspace(-self.L/2, -4.5, absorbing), np.linspace(-4.5, self.L/2, rest+1)[1:]))

            assert self.mesh.size == self.N_cells +1
            cell_centers = np.copy(self.mesh*0)
            for ix in range(cell_centers.size-1):
                cell_centers[ix] = (self.mesh[ix+1] + self.mesh[ix])/2
            # print(self.mesh, 'mesh')



        else:
            assert 0 


class source_class:

    def __init__(self, source_type, mesh, input_source, source_strength = 1.0):
        self.source_type = source_type
        self.mesh = mesh
        self.source_strength = source_strength
        self.input_source = input_source

    def make_source(self):
        if self.source_type == 'off':
            self.s = np.ones(self.mesh.size-1) * 0
        elif self.source_type == 'input':
            self.s = self.input_source
        elif self.source_type == 'shell_OH':
            self.s = np.ones(self.mesh.size-1) * 0
            for ix, xx in enumerate(self.s):
                if self.mesh[ix] >= 8 and self.mesh[ix + 1] <= 10:
                    self.s[ix] = 1.0
        elif self.source_type == 'volume':
            self.s = np.ones(self.mesh.size-1) * self.source_strength

        
class IC_class:
    def __init__(self, N_ang, N_cells, IC, mesh, angles):
        self.IC = IC
        self.mesh = mesh
        self.N_ang = N_ang
        self.N_cells = int(N_cells)
        self.angles = angles
    
    def make_IC(self):
        if self.IC == 'cold':
            self.angular_flux = np.zeros((int(self.N_ang), int(self.N_cells)))
        if self.IC == 'pl':
            self.angular_flux = np.zeros((self.N_ang, self.N_cells))
            middle = int(self.N_cells/2)
            dx = self.mesh[middle+1]-self.mesh[middle-1]
            x = np.linspace(-self.N_cells * dx + dx/2, self.N_cells * dx - dx/2, self.N_cells)
            sigma =  self.N_cells * dx /40
            # self.angular_flux[:, middle-1:middle+1] = 1/dx
            for ang in range(self.N_ang):
                self.angular_flux[ang,: ] = np.exp(-x**2/2/sigma**2)/sigma/math.sqrt(2*math.pi) * 4
        elif self.IC == 'larsen':
            self.angular_flux = np.zeros((self.N_ang, self.N_cells))
            mat_bound = np.argmin(np.abs(self.mesh+4.5))
            x =  0.5*(self.mesh[1:] + self.mesh[0:-1])
            for ang in range(self.N_ang):
                mu = self.angles[ang]
                if mu > 0:
                    self.angular_flux[ang,:mat_bound ] = np.exp(-2/mu)
                for ix, xx in enumerate(x):
                    if ix > mat_bound:
                        self.angular_flux[ang, ix] = 0.5 * (0.15 * (1 - 1/11 * xx ))
                


 
class boundary_class:
    def __init__(self, left_edge, right_edge, strength = [1,0]):
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.strength = strength
   #need to write a function for reflecting
    
    def __call__(self, side, mu):
        if side == 'left':
            if self.left_edge == 'vacuum':
                return 0.0
            elif self.left_edge == 'source1':
                if mu > 0.0:
                    return self.strength[0]
                else:
                    return 0.0
        elif side == 'right':
            if self.right_edge == 'vacuum':
                return 0.0
            elif self.right_edge == 'source1':
                if mu < 0.0:
                    return self.strength[-1]
                else:
                    return 0.0
            else:
                return 0.0



@njit
def mu_sweep(N_cells, psis, mun, sigma_t, sigma_s, mesh, s, phi, psiminusleft, psiplusright, geometry = 'slab'):
    psin = psis * 0
    # sigma_t = sigma_a + sigma_s
    phi = phi *sigma_s

    # print(sigma_s, 'scattering')
    # print(sigma_t, 'total')
    # print(mesh)
    if mun >0.0:
        for k in range(0, N_cells):
            q = s[k] + phi[k]


            delta = mesh[k+1]-mesh[k]
            if k == 0:
                # psiminus = boundary_class('left', mun)
                psiminus = psiminusleft
            
            if k == 0 and geometry == 'sphere': # only for sphere
                psin[k] = psiminusleft
            else:
                psin[k] = (1 + 0.5 * sigma_t[k] * delta/abs(mun))**-1 * (psiminus + 0.5 * delta * q/abs(mun))
            psiminus_new = 2 * psin[k] - psiminus
            psiminus = psiminus_new
        # error = 0

    elif mun <0.0:
        for kk in range(0, N_cells):
            k = N_cells - kk -1
            q = s[k] + phi[k]
            # print(s[99], 's99')
            # print(s.size)
            # print(k)
            delta = mesh[k+1]-mesh[k]
            if k == N_cells-1:
                #  psiplus = boundary_class('right', mun)
                psiplus = psiplusright
            psin[k] = (1 + 0.5 * sigma_t[k] * delta/abs(mun))**-1 * (psiplus + 0.5 * delta * q/abs(mun))
            psiplus_new = 2 * psin[k] - psiplus
            psiplus = psiplus_new
        # error = 0
    
    return psin


@njit
def mu_sweep_sphere(N_cells, psis, mun, wn, psiminus_mu, alphaplus, alphaminus, sigma_t, sigma_s, mesh, s, phi, psiminusleft, psiplusright, ang_diff_term, diff_type = 'diamond', psiplus_origin = 0.0):
    psin = psis * 0
    # sigma_t = sigma_a + sigma_s
    phi = phi *sigma_s

    # print(sigma_s, 'scattering')
    # print(sigma_t, 'total')
    # print(mesh)

    if mun >0.0:
        for k in range(0, N_cells):
            # if k ==0:
            #     psin[0] = psi_refl
            #     psiminus = psin[0]
            #     # psiminus_new = 2 * psin[k] - psiminus
            #     psiminus_mu[k] = 2 * psin[k] - psiminus_mu[k]
            # else:

                

                if k < N_cells-1:

                    rplus = 0.5 * (mesh[k+1] + mesh[k])
                else:
                    rplus = mesh[k] + (mesh[k-1] - mesh[k-2])/2
                if k>0:
                    rminus = 0.5 * (mesh[k] + mesh[k-1])
                else:
                    rminus = 0.0
                Aplus = 4 * math.pi * rplus**2
                Aminus = 4 * math.pi * rminus**2
                Vi = 4 * math.pi/3 * (rplus**3 - rminus**3)
            
                q = s[k] + phi[k]
                qnew = 1/ Vi * 4 * math.pi * q * (rplus**3 - rminus ** 3)/3
                q = qnew
                if k == 0:
                    # psiminus = psiminus_mu[0]
                    psiminus = psiplus_origin
                #     # psiminus = 10
                #     # psiminus = boundary_class('left', mun)
                    # psiminus = psiminusleft
                #     # psin[k] = psiminus
                    # print(psiminus, 'psi minus', mun)
                # else:
                if diff_type == 'diamond':
                    
                    psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aplus + 2/wn *(Aplus - Aminus) * alphaplus)**(-1) * (abs(mun) * (Aplus + Aminus) * psiminus + 1/wn * (Aplus-Aminus) * (alphaplus + alphaminus) * psiminus_mu[k] + Vi * q)
                elif diff_type == 'SH':
                    psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aplus)**-1 * (abs(mun) * (Aplus + Aminus) * psiminus - (Aplus - Aminus) * ang_diff_term[k]/2 + Vi * q)
                    # psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aplus + 4/wn *(Aplus - Aminus) * alphaplus)**-1 * (abs(mun) * (Aplus + Aminus) * psiminus + 2/wn * (Aplus-Aminus) * (alphaplus + alphaminus) * psiminus_mu[k] + Vi * q)
            
                psiminus_new = 2 * psin[k] - psiminus
                psiminus = psiminus_new
                # if k>0:
                # psiminus_mu[k] = 2 * psin[k] - psiminus_mu[k].copy()
    
        # error = 0

    elif mun <0.0:
        for kk in range(0, N_cells):
            k = N_cells - kk -1
            q = s[k] + phi[k]
            
            if k < N_cells-1:

                rplus = 0.5 * (mesh[k+1] + mesh[k])
            else:
                rplus = mesh[k] + (mesh[k-1] - mesh[k-2])/2
            if k>0:
                rminus = 0.5 * (mesh[k] + mesh[k-1])
            else:
                rminus = 0.0
            if rplus <= rminus:
                assert 0
            Aplus = 4 * math.pi * rplus**2
            Aminus = 4 * math.pi * rminus**2
            Vi = 4 * math.pi/3 * (rplus**3 - rminus**3)
            qnew = 1/ Vi * 4 * math.pi * q * (rplus**3 - rminus ** 3)/3 # assuming constant flux + source in each cell
            q = qnew
            # print(s[99], 's99')
            # print(s.size)
            # print(k)
            if k == N_cells-1:

                #  psiplus = boundary_class('right', mun)
                psiplus = psiplusright
            if abs(mun - -1) <=1e-13:
                psiminus_mu[k]  = (2 * psiplus + (rplus - rminus) * q) / (2 + sigma_t[k] *(rplus - rminus))
                psin[k] = psiminus_mu[k]
                psiplus_new =  2 * psin[k] - psiplus
                psiplus = psiplus_new
            else:
                if diff_type == 'diamond':
                    psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aminus + 2/wn *(Aplus - Aminus) * alphaplus)**(-1) * (abs(mun) * (Aplus + Aminus) * psiplus + 1/wn * (Aplus-Aminus) * (alphaplus + alphaminus) * psiminus_mu[k] + Vi * q)
                elif diff_type =='SH':
                    psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aminus)**-1 * (abs(mun) * (Aplus + Aminus) * psiplus - (Aplus - Aminus) * ang_diff_term[k]/2 + Vi * q)
                    # psin[k] = (sigma_t[k] * Vi + 2 * abs(mun) * Aminus + 4/wn *(Aplus - Aminus) * alphaplus)**-1 * (abs(mun) * (Aplus + Aminus) * psiplus + 2/wn * (Aplus-Aminus) * (alphaplus + alphaminus) * psiminus_mu[k] + Vi * q)
                psiplus_new = 2 * psin[k] - psiplus
                psiplus = psiplus_new
                # if k > 0:
                # psiminus_mu[k] = 2 * psin[k] - psiminus_mu[k].copy() 
                if k == 0:
                    psiplus_origin = 2*psin[k] - psiplus
                    # psiplus_origin = psin[k]
                    # print(psiplus_origin, 'psi+ 0', mun)

        # error = 0
    
    return psin, psiminus_mu, psiplus_origin


                


def convergence_estimator(xdata, ydata, target = 256, method = 'linear_regression'):
    if method == 'linear_regression':
        # lastpoint = ydata[-1]
        # ynew = np.log(np.abs(ydata[1:]-ydata[:-1]))
        # xnew = np.log(np.abs(xdata[1:]-xdata[:-1]))
        # a, b = np.polyfit(xnew, ynew,1)
        # err_estimate = (np.exp(b) * np.abs(target-xdata[:-1])**a)[-1]
        # print(err_estimate, 'err estimate')
        ynew = np.log(np.abs(ydata[-1]-ydata[:-1]))
        xnew = np.log(xdata[:-1])
        a, b = np.polyfit(xnew, ynew,1)
        c1 = np.exp(b)
        err_estimate = c1 * target ** a

        
    elif method == 'difference':
        # err_estimate = np.abs(ydata[-1] - ydata[-2]) /(xdata[-1]-xdata[-2]) 
        
        # alpha = np.abs(ydata[-1] - ydata[-2]) *xdata[-2]
        # err_estimate = alpha/target

        # err_estimate = np.abs(ydata[-1] - ydata[-2]) / np.abs(xdata[-2]-xdata[-1])/target*xdata[-2]*xdata[-1]
        err_estimate = np.abs(ydata[-1]-ydata[-2])
    
    
    elif method == 'richardson':
        ynew = np.log(np.abs(ydata[-1]-ydata[:-1]))
        xnew = np.log(xdata[:-1])
        a, b = np.polyfit(xnew, ynew,1)
        c1 = np.exp(b)
        k0 = -a
        h = 1/ xdata[-2]
        t = h * xdata[-1]
        A1 = (t**k0 * ydata[-1] -ydata[-2]) / (t**k0 - 1)
        err_estimate = np.abs(ydata[-1] - A1)

    return err_estimate    # return a
# @njit
def trapezoid_integrator(x_array, y_array, const_dx = True):
    if const_dx == True:
        dx = x_array[1]-x_array[0]

        res = 0 
        res +=  y_array[0]
        res +=  y_array[-1]
        for it in range(1, y_array.size -1):
            res += 2  *y_array[it]

        return 0.5 * dx * res
    
def reaction_rate(xs, phi, sigma, x1, x2):

    index1 = np.argmin(np.abs(xs-x1))
    index2 = np.argmin(np.abs(xs-x2))
    if xs[index1 + 1] < x2:
        index1 +=1 
    if x1 < xs[0]:
        x1 = xs[0]
    
    # print(xs[index1:index2], 'xs in integral')
    interp_phi = interp1d(xs, phi * sigma)
    result = integrate.quad(interp_phi, x1, x2)[0]
    # result = trapezoid_integrator(xs[index1:index2], phi[index1:index2] * sigma[index1:index2])
    return result
# @njit
def  wynn_epsilon_algorithm(S):
        n = S.size
        width = n-1
        # print(width)
        tableau = np.zeros((n + 1, width + 2))
        tableau[:,0] = 0
        tableau[1:,1] = S.copy() 
        for w in range(2,width + 2):
            for r in range(w,n+1):
                #print(r,w)
                # if abs(tableau[r,w-1] - tableau[r-1,w-1]) <= 1e-15:
                #     print('potential working precision issue')
                tableau[r,w] = tableau[r-1,w-2] + 1/(tableau[r,w-1] - tableau[r-1,w-1])
        return tableau


@njit
def calculate_psi_moments(N_mom, V, ws, N_ang, mus):
    moments = np.zeros(N_mom)
    for n in range(N_mom):
            for l in range(N_ang):
                moments[n] +=   ws[l] * V[l] * Pn_scalar(n, mus[l], -1, 1) 
    # print(moments, 'moments')
    # print(V,'solution vector in moments')
    # go backwards and delete moments that are basically zero
    # N_mom_needed = N_mom
    # tol = 1e-10
    # for n in range(N_mom-1):
    #     if np.max(np.abs(moments[n])) <= tol:
    #        if np.max(np.abs(moments[n+1])) <= tol:
    #            N_mom_needed = n
    #            break
    # N_mom_needed = N_mom
    return moments
@njit
def calculate_psi_moments_DPN(N_mom, V, ws, N_ang, mus):
    momentsL = np.zeros(N_mom)
    momentsR = np.zeros(N_mom)
    for n in range(N_mom):
            for l in range(N_ang):
                momentsL[n] +=   (0.5) * ws[l] * V[l] * Pn_scalar_minus(n, mus[l] * .5 - .5, -1, 0) 
                momentsR[n] +=   ws[l] * V[l] * Pn_scalar_plus(n, mus[l] * .5 + .5, 0, 1) 
    # print(moments, 'moments')
    # print(V,'solution vector in moments')
    # go backwards and delete moments that are basically zero
    # N_mom_needed = N_mom
    # tol = 1e-10
    # for n in range(N_mom-1):
    #     if np.max(np.abs(moments[n])) <= tol:
    #        if np.max(np.abs(moments[n+1])) <= tol:
    #            N_mom_needed = n
    #            break
    # N_mom_needed = N_mom
    return momentsL, momentsR
@njit
def Pn_scalar(n,x,a=-1.0,b=1.0):
    tmp = 0.0
    z = x

    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, z)
    return tmp

@njit
def Pn_scalar_plus(n,x,a=-1.0,b=1.0):
    tmp = 0.0
    z = x

    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, 2*z-1)
    return tmp
@njit
def Pn_scalar_minus(n,x,a=-1.0,b=1.0):
    tmp = 0.0
    z = x

    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, 2*z+1)
    return tmp
@njit
def numba_eval_legendre_float64(n, x):
      return eval_legendre_float64_fn(n, x)
  
_dble = ctypes.c_double
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0_1eval_legendre")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)
eval_legendre_float64_fn = functype(addr)

@njit 
def legendre_difference(N_mom, psi_moments, mu):
    res = 0.0
    for n in range(N_mom):
        if n > 0 and n < N_mom-1:
            res += psi_moments[n] * 0.5 * (n * (n-1) * Pn_scalar(n-1, mu, -1,1) - (n+2)*(n+1)* Pn_scalar(n+1, mu, -1,1) )
        elif n == 0:
            res += psi_moments[n] * 0.5 * ( - (n+2)*(n+1)* Pn_scalar(n+1, mu, -1,1) )
        elif n == N_mom -1:
            res += psi_moments[n] * 0.5 * (n * (n-1) * Pn_scalar(n-1, mu, -1,1))
        # res += psi_moments[n] * (2 * n+1) * 0.5  * (mu * (n-1) * Pn_scalar(n, mu, -1,1) - (n+1) * Pn_scalar(n+1, mu, -1,1))  
    return res


@njit 
def legendre_difference_DPN(N_mom, mom_L, mom_R, mu):
    res = 0.0
    for n in range(N_mom):
        res += mom_R[n] * ((1 + 2*n)*((-1 + mu + 2*mu**2*(-1 + n) - n + mu*n)*Pn_scalar_plus(n,mu) - (1 + mu)*(1 + n)*Pn_scalar_plus(1 + n,mu)))/(2.*mu)
        res +=  mom_L[n] * ((1 + 2*n)*((-1 - mu*(1 + 2*mu) + (-1 + mu)*(1 + 2*mu)*n)*Pn_scalar_minus(n,mu) - (-1 + mu)*(1 + n)*Pn_scalar_minus(1 + n,mu)))/(2.*mu)
    return res