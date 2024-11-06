from typing import Any
import numpy as np
from chaospy.quadrature import clenshaw_curtis
import math
from functions import quadrature

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
        self.N_cells = N_cells
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
        elif quad_type == 'gauss':
            print('Gaussian quadrature')
            self.mus, self.ws = quadrature(self.N_ang, 'gauss_lobatto')

        
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
            for it in range(self.mesh.size):
                if self.mesh[it] < -0.5:
                    self.sigma_a[it] = 0.1
                elif -0.5 <= self.mesh[it] <=0.5:
                    self.sigma_a[it] = 0.8
                elif self.mesh[it] > 0.5:
                    self.sigma_a[it] = 0.2
        else:
            assert 0 

    def make_sigma_s(self):
        if self.opacity_function == 'constant':
            self.sigma_s = np.ones(self.mesh.size-1) * self.sigma_s_bar
            self.sigma_t = self.sigma_t_bar*  np.ones(self.mesh.size-1) 
        elif self.opacity_function == '3_material':
            self.sigma_t = self.sigma_t_bar*  np.ones(self.mesh.size-1) 
            for it in range(self.mesh.size):
                if self.mesh[it] < -0.5:
                    self.sigma_s[it] = 0.8
                elif -0.5 <= self.mesh[it] <=0.5:
                    self.sigma_s[it] = 0.2
                elif self.mesh[it] > 0.5:
                    self.sigma_s[it] = 0.8


        else: 
            assert 0 



class mesh_class:
    def __init__(self, N_cells, L, opacity_function):
        self.N_cells = N_cells
        self.opacity_function = opacity_function
        self.L = L

    def make_mesh(self):
        if self.opacity_function == 'constant':
            self.mesh = np.linspace(-self.L/2, self.L/2, self.N_cells+1)
        elif self.opacity_function == '3_material':
            third = int(int(self.N_cells + 1)/3)
            rest = int(self.N_cells+1-2*third)
            self.mesh = np.concatenate((np.linspace(-self.L/2, -0.5, third), np.linspace(-0.5, 0.5, rest+2)[1:-1]), (np.linspace(0.5, self.L/2, third)))
            assert self.mesh.size == self.N_cells +1
            assert (self.mesh == -0.5).any()
            assert (self.mesh == 0.5).any()

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
        
class IC_class:
    def __init__(self, N_ang, N_cells, IC, mesh):
        self.IC = IC
        self.mesh = mesh
        self.N_ang = N_ang
        self.N_cells = N_cells
    
    def make_IC(self):
        if self.IC == 'cold':
            self.angular_flux = np.zeros((self.N_ang, self.N_cells))
        if self.IC == 'pl':
            self.angular_flux = np.zeros((self.N_ang, self.N_cells))
            middle = int(self.N_cells/2)
            dx = self.mesh[middle+1]-self.mesh[middle-1]
            x = np.linspace(-self.N_cells * dx + dx/2, self.N_cells * dx - dx/2, self.N_cells)
            sigma =  self.N_cells * dx /40
            # self.angular_flux[:, middle-1:middle+1] = 1/dx
            for ang in range(self.N_ang):
                self.angular_flux[ang,: ] = np.exp(-x**2/2/sigma**2)/sigma/math.sqrt(2*math.pi) * 4
 
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




def mu_sweep(N_cells, psis, mun, sigma_t, sigma_s, mesh, s, phi, boundary_class):
    psin = psis * 0
    # sigma_t = sigma_a + sigma_s
    phi = phi *sigma_s
    if mun >0.0:
        for k in range(0, N_cells):
            q = s[k] + phi[k]


            delta = mesh[k+1]-mesh[k]
            if k == 0:
                psiminus = boundary_class('left', mun)

            psin[k] = (1 + 0.5 * sigma_t[k] * delta/abs(mun))**-1 * (psiminus + 0.5 * delta * q/abs(mun))
            psiminus_new = 2 * psin[k] - psiminus
            psiminus = psiminus_new
        error = 0

    elif mun <0.0:
        for kk in range(0, N_cells):
            k = N_cells - kk -1
            q = s[k] + phi[k]
            # print(s[99], 's99')
            # print(s.size)
            # print(k)
            delta = mesh[k+1]-mesh[k]
            if k == N_cells-1:
                 psiplus = boundary_class('right', mun)
            psin[k] = (1 + 0.5 * sigma_t[k] * delta/abs(mun))**-1 * (psiplus + 0.5 * delta * q/abs(mun))
            psiplus_new = 2 * psin[k] - psiplus
            psiplus = psiplus_new
        error = 0
    
    return psin

                


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
    return err_estimate
    
        # return a

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
    result = trapezoid_integrator(xs[index1:index2], phi[index1:index2] * sigma[index1:index2])
    return result

