using PyCall
using JLD, HDF5
using NLsolve
const a = 0.01372
const c = 299.98
using LinearAlgebra
@pyimport product_quad as pq

function linspace(a,b,N)
    return range(a,stop=b,length=N)
end
    
function sweep2D_scb(Nx,Ny,hx,hy,etax,etay,sigmat,Q,boundaryx, boundaryy)
    #q will have dimensions Nx,Ny,4
    #sigmat will have the same
    #in cell unknown layou
    # ----------
    # | 3    2 |
    # | 0    1 |
    #-----------
    Lmat = zeros((4,4))
    tmpLmat = Lmat
    psi = zeros((Nx,Ny,4))
    Ax = [ 1 1 0 0; -1 -1 0 0; 0 0 -1 -1; 0 0 1 1.0];
    Ay =  [1.0 0 0 1; 0 1 1 0; 0 -1 -1 0; -1 0 0 -1];
    ihx = 1/hx
    ihy = 1/hy
    Lmat = etax*ihx*Ax + etay.*ihy.*Ay
    b = zeros(4)
    if (etax > 0) & (etay > 0)
        Lmat[2,2] += 2*etax*ihx
        Lmat[3,3] += 2*etax*ihx + 2*etay*ihy
        Lmat[4,4] += 2*etay*ihy

        for i in 1:Nx
            for j in 1:Ny
                if i==1
                    psileft1 = boundaryx
                    psileft2 = boundaryx
                else
                    psileft1 = psi[i-1,j,2]
                    psileft2 = psi[i-1,j,3]
                end
                if j==1
                    psibottom3 = boundaryy
                    psibottom2 = boundaryy
                else
                    psibottom3 = psi[i,j-1,4]
                    psibottom2 = psi[i,j-1,3]
                end


                tmpLmat = Lmat + diagm(sigmat[i,j,:])
                #print(tmpLmat,Lmat,np.diag(sigmat[i,j,:]))
                b[1] = Q[i,j,1] + psibottom3*2*etay*ihy + psileft1*2*etax*ihx
                b[2] = Q[i,j,2] + psibottom2*2*etay*ihy
                b[3] = Q[i,j,3]
                b[4] = Q[i,j,4] + psileft2*2*etax*ihx
                psi[i,j,:] = tmpLmat\b
            end
        end

    elseif (etax < 0) & (etay < 0)
        Lmat[2,2] += -2*etay*ihy
        Lmat[1,1] += -2*etax*ihx - 2*etay*ihy
        Lmat[4,4] += -2*etax*ihx

        for i in Nx:-1:1
            for j in Ny:-1:1
                if i==Nx
                    psiright3 = boundaryx
                    psiright0 = boundaryx
                else
                    psiright3 = psi[i+1,j,4]
                    psiright0 = psi[i+1,j,1]
                end
                if j==Ny
                    psitop0 = boundaryy
                    psitop1 = boundaryy
                else
                    psitop0 = psi[i,j+1,1]
                    psitop1 = psi[i,j+1,2]
                end

                tmpLmat = Lmat + diagm(sigmat[i,j,:])
                b[1] = Q[i,j,1]
                b[2] = Q[i,j,2] - psiright0*2*etax*ihx
                b[3] = Q[i,j,3] - psitop1*2*etay*ihy - psiright3*2*etax*ihx
                b[4] = Q[i,j,4] - psitop0*2*etay*ihy
                psi[i,j,:] = tmpLmat\b
            end
        end
    elseif (etax > 0) & (etay < 0)
        Lmat[2,2] += 2*etax*ihx- 2*etay*ihy
        Lmat[1,1] += - 2*etay*ihy
        Lmat[3,3] += 2*etax*ihx

        for i in 1:Nx
            for j in Ny:-1:1
                if i==1
                    psileft1 = boundaryx
                    psileft2 = boundaryx
                else
                    psileft1 = psi[i-1,j,2]
                    psileft2 = psi[i-1,j,3]
                end
                if j==Ny
                    psitop0 = boundaryy
                    psitop1 = boundaryy
                else
                    psitop0 = psi[i,j+1,1]
                    psitop1 = psi[i,j+1,2]
                end

                tmpLmat = Lmat + diagm(sigmat[i,j,:])
                b[1] = Q[i,j,1] +  psileft1*2*etax*ihx
                b[2] = Q[i,j,2]
                b[3] = Q[i,j,3] - psitop1*2*etay*ihy
                b[4] = Q[i,j,4] - psitop0*2*etay*ihy + psileft2*2*etax*ihx
                psi[i,j,:] = tmpLmat\b
            end
        end
    elseif (etax < 0) & (etay > 0)

        Lmat[1,1] += -2*etax*ihx
        Lmat[3,3] +=  2*etay*ihy
        Lmat[4,4] += 2*etay*ihy -2*etax*ihx

        for i in Nx:-1:1
            for j in 1:Ny
                if i==Nx
                    psiright3 = boundaryx
                    psiright0 = boundaryx
                else
                    psiright3 = psi[i+1,j,4]
                    psiright0 = psi[i+1,j,1]
            end
                if j==1
                    psibottom3 = boundaryy
                    psibottom2 = boundaryy
                else
                    psibottom3 = psi[i,j-1,4]
                    psibottom2 = psi[i,j-1,3]
            end


                tmpLmat = Lmat + diagm(sigmat[i,j,:])
                b[1] = Q[i,j,1] + psibottom3*2*etay*ihy
                b[2] = Q[i,j,2] + psibottom2*2*etay*ihy - psiright0*2*etax*ihx
                b[3] = Q[i,j,3] - psiright3*2*etax*ihx
                b[4] = Q[i,j,4]
                psi[i,j,:] = tmpLmat\b
        end
    end
end
psi
end

function flatten_phi(phi,Nx,Ny,hx,hy)
    x = zeros((2*Nx,2*Ny))
    X = linspace(hx/2,(Nx*hx)-hx/2,Nx)
    y = zeros((2*Nx,2*Ny))
    Y = linspace(hy/2,(Ny*hy)-hy/2,Ny)
    phi_out = zeros((2*Nx,2*Ny))
    for i in 1:Nx
        for j in 1:Ny
            phi_out[2*i-1,2*j-1] = phi[i,j,1]
            phi_out[2*i,2*j-1] = phi[i,j,2]
            phi_out[2*i,2*j] = phi[i,j,3]
            phi_out[2*i-1,2*j] = phi[i,j,4]

            x[2*i-1,2*j-1] = X[i] - hx/4
            x[2*i,2*j-1] = X[i] + hx/4
            x[2*i,2*j] = X[i] + hx/4
            x[2*i-1,2*j] = X[i] - hx/4

            y[2*i-1,2*j-1] = Y[j] - hy/4
            y[2*i,2*j-1] = Y[j] - hy/4
            y[2*i,2*j] = Y[j] + hy/4
            y[2*i-1,2*j] = Y[j] + hy/4
        end
    end
    (x,y,phi_out)
end

function average_phi(phi,Nx,Ny,hx,hy)
    X = linspace(hx/2,(Nx*hx)-hx/2,Nx)
    Y = linspace(hy/2,(Ny*hy)-hy/2,Ny)
    phi_out = zeros((Nx,Ny))
    for i in 1:Nx
        for j in 1:Ny
            phi_out[i,j] = 0.25*(phi[i,j,1] + phi[i,j,2] + phi[i,j,3] + phi[i,j,4])
        end
    end
    (X,Y,phi_out)
end




function SI(Nx,Ny,hx,hy,phi_old,Nord,sigmat,q,boundaryx, boundaryy)
    w,etax,etay = pq.prod_quad(Nord)
    #Qin = (q + sigmas.*phi_old)
    #print(w,np.sum(w),np.sum(Qin))
    angles = length(w)
    phi = zeros((Nx,Ny,4))
    for n in 1:angles
        phi += w[n]*sweep2D_scb(Nx,Ny,hx,hy,etax[n],etay[n],sigmat,q[:,:,:,n],boundaryx[n], boundaryy[n])
    end
    phi
end


function SI(Nx,Ny,hx,hy,phi_old,Nord,sigmat,sigmas,q,boundaryx, boundaryy)
    w,etax,etay = pq.prod_quad(Nord)
    #print(w,np.sum(w),np.sum(Qin))
    angles = length(w)
    phi = zeros((Nx,Ny,4))
    #println(size(q)," ", size(phi_old), " ", size(sigmas))
    for n in 1:angles
        Qin = (q[:,:,:,n] + sigmas.*phi_old)
        phi += w[n]*sweep2D_scb(Nx,Ny,hx,hy,etax[n],etay[n],sigmat,Qin,boundaryx[n], boundaryy[n])
    end
    phi
end

function SI_psi(Nx,Ny,hx,hy,phi_old,Nord,sigmat,q,boundaryx, boundaryy)
    w,etax,etay = pq.prod_quad(Nord)
    #print(w,np.sum(w),np.sum(Qin))
    angles = length(w)
    psi = zeros((Nx,Ny,4,angles))
    for n in 1:angles
        psi[:,:,:,n] += sweep2D_scb(Nx,Ny,hx,hy,etax[n],etay[n],sigmat,q[:,:,:,n],boundaryx[n], boundaryy[n])
    end
    psi
end

function SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat,sigmas,q,
             boundaryx, boundaryy, L2tol=1e-8, Linftol = 1e-4, maxits = 20; LOUD = 0)
    phi = copy(phi_old)
    converged = false
    iteration = 0
    w,etax,etay = pq.prod_quad(Nord)
    angles = length(w)
    norm_const = 1/sum(w)
    #println(norm_const)
    L2diff = 0
    Linfdiff = 0
    while !(converged)
        qtot = copy(q)
        for angle in 1:angles
            qtot[:,:,:,angle] .+= norm_const*phi.*sigmas
        end
        phi = SI(Nx,Ny,hx,hy,phi_old,Nord,sigmat,qtot,boundaryx, boundaryy)
        L2diff = sqrt(sum(((phi-phi_old)./(abs.(phi) .+1e-14)).^2/(Nx*Ny*4)))
        Linfdiff = reduce(max,abs.(phi-phi_old)./(abs.(phi) .+1e-14))
        if LOUD > 0
            println("Iteration: ",iteration+1," L2 Diff: ",L2diff," Linf Diff: ",Linfdiff)
        end
        if (L2diff < L2tol) & (Linfdiff < Linftol)
            converged = true
        elseif (iteration >= maxits)
            converged = true
        end
        iteration += 1
        phi_old = phi
    end
    #compute psi
    qtot = copy(q)
    for angle in 1:angles
        qtot[:,:,:,angle] .+= phi_old.*sigmas*norm_const
    end
        if LOUD == -1
            println("Iteration: ",iteration+1," L2 Diff: ",L2diff," Linf Diff: ",Linfdiff)
        end
    psi = SI_psi(Nx,Ny,hx,hy,phi_old,Nord,sigmat,qtot,boundaryx, boundaryy)
    (psi,copy(phi),iteration)
end

function TD_Solve(Nx,Ny,hx,hy,psi_init,Nord,sigmat,sigmas,q,v,dt,Tfinal,
             boundaryx, boundaryy, L2tol=1e-8, Linftol = 1e-4, maxits = 200; LOUD = 0)
    time = 0
    times = []
    its = []
    done = false
    phi_old = copy(psi_init[:,:,:,1])
    psi = copy(psi_init)
    phi = copy(phi_old)
    while !(done)
        delta_t = min(Tfinal-time,dt)
        sigmat_star = sigmat + 1/(v*delta_t)
        Qstar = q + psi/(v*delta_t)
        psi,phi,iteration = SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat_star,sigmas,Qstar,
                 boundaryx, boundaryy, L2tol, Linftol, maxits, LOUD=LOUD)
        phi_old = deepcopy(phi)
        time = time + delta_t
        if (LOUD > 0) | (LOUD == -1)
            println("time = ", time)
        end
        times = push!(times,time)
        its = push!(its,iteration)
        done = time >= Tfinal

        if (LOUD == 0.25)
            figure()
            x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            pcolor(phi_flat)
            colorbar()
            show()
        end
    end
    return psi, phi, times, its
end

function RT_Solve(Nx,Ny,hx,hy,psi_init,T_init,Nord,density,sigma_func,q_func,Cv,inveos,eos,v,dt,Tfinal,
             boundaryx, boundaryy; L2tol=1e-8, Linftol = 1e-4, maxits = 20000, LOUD = 0)
    time = 0.
    times = []
    its = []
    done = false
    phi_old = copy(psi_init[:,:,:,1])
    psi = copy(psi_init)
    nsteps = Int64(round(Tfinal/dt))
    phi_t = zeros((Nx,Ny,4,nsteps+1))
    T_t = zeros((Nx,Ny,4,nsteps+1))
    phi = 0*phi_old
    #print(size(T_init))
    T = zeros((Nx,Ny,4)) + T_init
    w,etax,etay = pq.prod_quad(Nord)
    angles = length(w)
    norm_const = 1.0/sum(w)
    Qstar = 0 * q_func(0)
    step = 1
    boundary_energy_x = 0
    boundary_energy_y = 0
    for angle=1:angles
        phi_old += w[angle] * psi[:,:,:,angle]
        boundary_energy_x += Nx*w[angle]*boundaryx[angle]/c
        boundary_energy_y += Ny*w[angle]*boundaryx[angle]/c
    end
    phi = copy(phi_old)
    source_energy = 0
    while !(done)
        delta_t = min(Tfinal-time,dt)
        sigma = sigma_func(T,density)
        sigmat_star = sigma .+ 1/(v*delta_t)
        q = q_func(time + dt)

        source_energy = 0

        for angle = 1:angles
            source_energy += sum(Q[:,:,:,angle]*w[angle])/c
        end
        #update C_v
        curr_Cv = Cv(T,density)

        beta = 4*a*c*T.^3 ./curr_Cv
        f = 1 ./ (1 .+ beta.*sigma*delta_t)

        #update Sigma
        sigmas = (1 .-f).*sigma

        psi_n = copy(psi)
        phi_n = copy(phi)
        T_n = copy(T)

        #update source
        qsource = a*c*T.^4
        for angle in 1:angles
            Qstar[:,:,:,angle] = q[:,:,:,angle] + psi[:,:,:,angle]/(v*delta_t) + f.*sigma.*qsource*norm_const
        end


        #Update Psi
        psi,phi,iteration = SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat_star,sigmas,Qstar,
                 boundaryx, boundaryy, L2tol, Linftol, maxits, LOUD=LOUD)
        phi_old = deepcopy(phi)

        #update T
        #println(reduce(max,curr_Cv), " ", reduce(min,curr_Cv))
        T = inveos(eos(T,density) + delta_t .* f .* sigma .* (phi - qsource/norm_const),density)



        time = time + delta_t

        energy_leaving = 0
        for angle = 1:angles
            if (etax[angle] > 0)
                energy_leaving += sum(w[angle] * psi[Nx,:,2,angle])/c + sum(w[angle] * psi[Nx,:,3,angle])/c
            else
                energy_leaving += sum(w[angle] * psi[1,:,1,angle])/c + sum(w[angle] * psi[1,:,4,angle])/c
            end
            if (etay[angle] > 0)
                energy_leaving += sum(w[angle] * psi[:,Ny,3,angle])/c + sum(w[angle] * psi[:,Ny,4,angle])/c
            else
                energy_leaving += sum(w[angle] * psi[:,1,1,angle])/c + sum(w[angle] * psi[:,1,2,angle])/c
            end
        end

        println("Step $(step): Initial Energy = $(sum(eos(T_n,density)) + sum(phi_n/c)),
        Energy Added = $(boundary_energy_x + boundary_energy_y + source_energy), Energy Leaving = $(energy_leaving)")

        if (LOUD > 0) | (LOUD == -1)
            println("time = ", time)
        end
        times = push!(times,time)
        its = push!(its,iteration)
        done = time >= Tfinal

        if (LOUD == 0.25)
            figure()
            x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            pcolor(phi_flat)
            colorbar()
            show()
        end
        T_t[:,:,:,step] = copy(T)
        phi_t[:,:,:,step] = copy(phi)
        step += 1
    end
    return psi, phi_t, T_t, time, times, its
end



function RT_Solve_Iterate(Nx,Ny,hx,hy,psi_init,T_init,Nord,density,sigma_func,q_func,Cv,inveos,eos,v,dt,Tfinal,
             boundaryx, boundaryy, L2tol=1e-8, Linftol = 1e-4; maxits = 200, LOUD = 0)
    time = 0.
    times = []
    its = []
    done = false
    phi_old = copy(psi_init[:,:,:,1])*0

    psi = copy(psi_init)
    nsteps = Int64(round(Tfinal/dt))
    phi_t = zeros((Nx,Ny,4,nsteps+1))
    T_t = zeros((Nx,Ny,4,nsteps+1))
    #print(size(T_init))
    T = zeros((Nx,Ny,4)) + T_init
    w,etax,etay = pq.prod_quad(Nord)
    angles = length(w)
    for angle=1:angles
        phi_old += w[angle] * psi[:,:,:,angle]
    end

    phi = phi_old*1
    norm_const = 1.0/sum(w)
    Qstar = 0 * q_func(0)
    step = 1

    boundary_energy_x = 0
    boundary_energy_y = 0
    for angle=1:angles
        phi_old += w[angle] * psi[:,:,:,angle]
        boundary_energy_x += Nx*w[angle]*boundaryx[angle]/c
        boundary_energy_y += Ny*w[angle]*boundaryx[angle]/c
    end
    phi = copy(phi_old)
    source_energy = 0
    while !(done)
        delta_t = min(Tfinal-time,dt)
        sigma = sigma_func(T,density)
        sigmat_star = sigma + 1/(v*delta_t)
        q = q_func(time + dt)



        source_energy = 0

        for angle = 1:angles
            source_energy += sum(Q[:,:,:,angle]*w[angle])/c
        end
        #update C_v
        curr_Cv = Cv(T,density)

        beta = 4*a*c*T.^3 ./curr_Cv
        f = 1 ./ (1 + beta.*sigma*delta_t)

        #update Sigma
        sigmas = (1-f).*sigma

        psi_n = copy(psi)
        phi_n = copy(phi)
        T_n = copy(T)

        #update source
        qsource = a*c*T.^4
        for angle in 1:angles
            Qstar[:,:,:,angle] = q[:,:,:,angle] + psi[:,:,:,angle]/(v*delta_t) + f.*sigma.*qsource*norm_const
        end


        #Update Psi
        psi,phi,iteration = SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat_star,sigmas,Qstar,
                 boundaryx, boundaryy, L2tol, Linftol, maxits, LOUD=LOUD)
        phi_old = deepcopy(phi)

        #update T
        #println(reduce(max,curr_Cv), " ", reduce(min,curr_Cv))
        T = inveos(eos(T,density) + delta_t .* f .* sigma .* (phi - qsource/norm_const),density)



        time = time + delta_t

        energy_leaving = 0
        for angle = 1:angles
            if (etax[angle] > 0)
                energy_leaving += sum(w[angle] * psi[Nx,:,2,angle])/c + sum(w[angle] * psi[Nx,:,3,angle])/c
            else
                energy_leaving += sum(w[angle] * psi[1,:,1,angle])/c + sum(w[angle] * psi[1,:,4,angle])/c
            end
            if (etay[angle] > 0)
                energy_leaving += sum(w[angle] * psi[:,Ny,3,angle])/c + sum(w[angle] * psi[:,Ny,4,angle])/c
            else
                energy_leaving += sum(w[angle] * psi[:,1,1,angle])/c + sum(w[angle] * psi[:,1,2,angle])/c
            end
        end

        println("Step $(step): Initial Energy = $(sum(eos(T_n,density)) + sum(phi_n/c)),
        Energy Added = $(boundary_energy_x + boundary_energy_y + source_energy), Energy Leaving = $(energy_leaving)")

        time = time + delta_t
        if (LOUD > 0) | (LOUD == -1)
            println("time = ", time)
        end
        times = push!(times,time)
        its = push!(its,iteration)
        done = time >= Tfinal

        if (LOUD == 0.25)
            figure()
            x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            pcolor(phi_flat)
            colorbar()
            show()
        end
        T_t[:,:,:,step] = copy(T)
        phi_t[:,:,:,step] = copy(phi)
        step += 1
    end
    return psi, phi_t, T_t, time, times, its
end



function RT_Solve_Iterate_Gentile(Nx,Ny,hx,hy,psi_init,T_init,Nord,density,sigma_func,q_func,Cv,inveos,eos,v,dt,Tfinal,
             boundaryx, boundaryy; L2tol=1e-8, Linftol = 1e-4, maxits = 200, LOUD = 0)
    time = 0.
    times = []
    its = []
    done = false
    phi_old = copy(psi_init[:,:,:,1])*0

    psi = copy(psi_init)
    nsteps = Int64(round(Tfinal/dt))
    phi_t = zeros((Nx,Ny,4,nsteps+1))
    T_t = zeros((Nx,Ny,4,nsteps+1))
    #print(size(T_init))
    T = zeros((Nx,Ny,4)) + T_init
    w,etax,etay = pq.prod_quad(Nord)
    angles = length(w)
    for angle=1:angles
        phi_old += w[angle] * psi[:,:,:,angle]
    end

    phi = phi_old*1
    norm_const = 1.0/sum(w)
    Qstar = 0 * q_func(0)
    step = 1
    psi_r = 0

    while !(done)
        delta_t = min(Tfinal-time,dt)

        #T iteration
        converged = false
        T_n = deepcopy(T)
        psi_n = deepcopy(psi)
        phi_n = deepcopy(phi)
        Temperature_its = 1
        iteration = 0
        T_prev = deepcopy(T)
        phi_prev = deepcopy(phi)
        T_two_prev = deepcopy(T)
        while ~(converged)
            sigma = sigma_func(T_prev,density)
            sigmat_star = sigma + 1/(v*delta_t)
            q = q_func(time + dt)



            #transport boundary and sources


            #update source
            for angle in 1:angles
                Qstar[:,:,:,angle] = q[:,:,:,angle] + psi_n[:,:,:,angle]/(v*delta_t) #+ sigma.*qsource*norm_const
            end
            if (Temperature_its == 1)
                psi_r,phi_r,iteration = SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat_star,0*sigma,Qstar,
                                                 boundaryx, boundaryy, L2tol, Linftol, maxits, LOUD=LOUD)
            end



            #now add in source and solve again


            #update source to only have emission
            qsource = a*c*(T.^4 - T_two_prev.^4) #beta*c.*(curr_Cv.*T_n - 0.75*curr_Cv.*T)./(1+beta.*c.*sigma*delta_t)
            if (Temperature_its == 1)
                qsource =  a*c*(T.^4)
            end
            for angle in 1:angles
                Qstar[:,:,:,angle] = sigma.*qsource*norm_const
            end

            #Update Psi to get Psi_i
            psi_i,phi_i,iteration = SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat_star,0*sigma,Qstar,
                                             boundaryx*0, boundaryy*0, L2tol, Linftol, maxits, LOUD=LOUD)

            #update temperature
            if (Temperature_its == 1)
                psi = psi_r + psi_i
                phi = phi_r + phi_i
            else
                psi += psi_i
                phi += phi_i
            end

            for x_cell = 1:Nx
                for y_cell = 1:Ny
                    for node = 1:4
                        function f!(F,T)
                            F[1] = eos(T[1],density[x_cell,y_cell,node])- eos(T_n[x_cell,y_cell,node],density[x_cell,y_cell,node]) -
                            delta_t*sigma[x_cell,y_cell,node]*(phi[x_cell,y_cell,node] - a*c*T[1]^4)
                        end
                        T[x_cell,y_cell,node] = nlsolve(f!,[T_prev[x_cell,y_cell,node]],ftol=1e-12).zero[1]
                    end
                end
            end

            #T = inveos(eos(T_n,density) + delta_t  .* sigma.* (phi  - a*c*T.^4),density)
            #figure()
            #x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            #plot(x-Lx/2,phi_flat[:,Int64(round(Ny/2))])
            #title("phi  $(Temperature_its)")
            #figure()
            #x,y,phi_flat = flatten_phi(T,Nx,Ny,hx,hy)
            #plot(x-Lx/2,phi_flat[:,Int64(round(Ny/2))])
            #title("T $(Temperature_its)")


            Linfdiff =  reduce(max,abs.(T-T_prev)./(abs.(T)+1e-14))
            if (Linfdiff < 10*Linftol) | (Temperature_its > maxits)
                converged = true
            else
                converged = false
            end
            if (LOUD == 1)
                println("Temperature Iteration $Temperature_its : $Linfdiff")
            end
            Temperature_its += 1
            T_two_prev = deepcopy(T_prev)
            T_prev = deepcopy(T)
            phi_prev = deepcopy(phi)

        end


        time = time + delta_t
        if (LOUD > 0) | (LOUD == -1)
            println("time = $(time), Temperature iterations = $(Temperature_its-1)")
        end
        times = push!(times,time)
        its = push!(its,iteration)
        done = time >= Tfinal

        if (LOUD == 0.25)
            figure()
            x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            pcolor(phi_flat)
            colorbar()
            show()
        end
        T_t[:,:,:,step] = copy(T)
        phi_t[:,:,:,step] = copy(phi)
        step += 1
    end
    return psi, phi_t, T_t, time, times, its
end




function solver_with_dmd(matvec, b,
        K = 10, Rits = 2, steady = 1, x = zeros(1),
        step_size = 10, L2_tol = 1e-8, Linf_tol = 1e-5; max_its = 10, LOUD=0)

    N = length(b)
    if length(x) != N
        x = b
        println("Length x NE b")
    end
    iteration = 0
    converged = false
    total_its = 0
    while ((!(converged)) & (iteration < max_its))
        for r in 1:(Rits)
            x_new = matvec(x) + b
            total_its += 1
            #check convergence
            L2err = sum((x./(x_new)  .- 1).^2 ./sqrt((N)))
            Linferr = reduce(max,abs.(x./(x_new) .- 1))
            if (L2err < L2_tol) & (Linferr < Linf_tol)
                println("Iteration: ", iteration+1, " Rich: ", r, " Resid= ", L2err, " ", Linferr, " ", converged)
                converged = true
                break
            end
            x = x_new
            if LOUD>0
                println("Iteration: ", iteration+1, " Rich: ", r, " Resid= ", L2err, " ", Linferr, " ", converged)
                if (LOUD>1)
                    print("x =",x)
                end
            end

        end
        if !(converged)
            x[1:N] = DMD_prec(matvec, b, K, steady, x, step_size)
            if LOUD>0
                println("Iteration: ", iteration+1, " DMD completed." )
                if (LOUD>1)
                    print("Post DMD x =",x)
                end
            end
            total_its += K
        iteration += 1
        end
    end
    if LOUD > 0
        println("Total iterations is ", total_its)
    end
    (x, total_its)
end

function DMD_prec(matvec,b, K = 10, steady = 1, x = zeros(1), step_size = 10, GM = 0)

    N = length(b)
    if length(x) != N
        x = b
        println("Length x NE b")
    end
    x_new = x*0
    x_orig = x
    x_0 = x
    #perform K iterations of matvec
    Yplus = zeros((N,K-1))
    Yminus = zeros((N,K-1))

    for k in 1:(K)
        x_new = matvec(x) + b
        if (k < K)
            Yminus[:,k] = x_new - x
            x_0 = x_new
        end
        if (k>1)
            Yplus[:,k-1] = x_new-x
        end
        x = x_new
    end
    #now perform update

    #compute svd
    #println(Yminus)
    u,s,v = svd(Yminus,thin=true)
    #println(u)
    #println(s)
    #println(v)
    #find the non-zero singular values
    if (length(x) > 1) & (length(s[find(x-> x> 1.0e-12, (1-cumsum(s)/sum(s)))]) >= 1)
        spos = s[find(x-> x> 1.0e-12, (1-cumsum(s)/sum(s)))]
    else
        spos =  max.(s[1:1,1:1],1e-12) #s
    end
    #create diagonal matrix
    mat_size = reduce(min,[K,length(spos)])
    S = zeros((mat_size,mat_size))

    #select the u and v that correspond with the nonzero singular values
    unew = 1.0*u[:,1:mat_size]
    vnew = 1.0*v'[1:mat_size,:]
    #S will be the inverse of the singular value diagonal matrix
    for ii = 1:mat_size
        S[ii,ii] = 1/spos[ii]
    end
    #not sure we need this
    En = u

    #the approximate A operator is Ut A U = Ut Y+ V S
    part1 =  unew' * Yplus
    part2 =  part1 * vnew'
    Atilde = part2 * S'

    eigsN,vsN = eig(Atilde)
    if (reduce(max,abs.(eigsN))>1)
        #an eigenvalue is too big
        print("*****Warning*****  The number of steps may be too small")
        eigsN[find(x->x>1,abs.(eigsN))] = 0
    end
    eigsN = real(eigsN)
    #change Atilde to only have the right eigenvalues
    Atilde = real(((vsN * diagm(eigsN)) * inv(vsN)))

    if steady==1
        Z = unew*vsN
        Zdagger = (Z'*Z) \ Z' #np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
        rhs = unew'*Yplus[:,end] #np.dot(np.matrix(unew).getH(),Yplus[:,-1])
        delta_y = (eye(size(Atilde)[1]) -  Atilde) \ (rhs)
            #np.linalg.solve(np.identity(Atilde.shape[0]) - Atilde,np.transpose(rhs))
        x_old = - (Yplus[:,K-1] - x)

        #println(vnew')
        #println(size((unew*delta_y)'))
        steady_update = x_old + (unew*delta_y)
    end

    steady_update
end

function DMD_prec_nonlin(xs; steady = 1, x = zeros(1), step_size = 10, GM = 0)

    N = size(xs)[1]
    K = size(xs)[2] - 1
    println("N = $(N), K = $(K)")
    if length(x) != N
        x = xs[:,1]
        println("Length x NE b, $(N)")
    end
    x_new = x*0
    x_orig = x
    x_0 = x
    #perform K iterations of matvec
    Yplus = zeros((N,K-1))
    Yminus = zeros((N,K-1))

    for k in 1:(K-1)
        Yminus[:,k] = xs[:,k+1]-xs[:,k]
        Yplus[:,k] = xs[:,k+2]-xs[:,k+1]
    end
    #println(Yminus)
    #println(Yplus)
    #now perform update

    #compute svd
    #println(Yminus)
    u,s,v = svd(Yminus,thin=true)
    #println(u)
    #println(s)
    #println(v)
    #find the non-zero singular values
    if (N > 1) & (length(s[find(x-> x> 1.0e-12, (1-cumsum(s)/sum(s)))]) >= 1)
        spos = s[find(x-> x> 1.0e-12, (1-cumsum(s)/sum(s)))]
    else
        spos = max.(s[1:1,1:1],1e-12)
    end
    #create diagonal matrix
    mat_size = reduce(min,[K,length(spos)])
    S = zeros((mat_size,mat_size))

    #select the u and v that correspond with the nonzero singular values
    unew = 1.0*u[:,1:mat_size]
    vnew = 1.0*v'[1:mat_size,:]
    #S will be the inverse of the singular value diagonal matrix
    for ii = 1:mat_size
        S[ii,ii] = 1/spos[ii]
    end
    #println(S)
    #not sure we need this
    En = u

    #the approximate A operator is Ut A U = Ut Y+ V S
    part1 =  unew' * Yplus
    part2 =  part1 * vnew'
    Atilde = part2 * S'

    println("Atilde = $(Atilde)")
    eigsN,vsN = eig(Atilde)
    if (reduce(max,abs.(eigsN))>=1-1e-3)
        #an eigenvalue is too big
        print("*****Warning*****  The number of steps may be too small")
        eigsN[find(x->x>=1-1e-3,abs.(eigsN))] = 0
    end
    eigsN = real(eigsN)
    #change Atilde to only have the right eigenvalues
    Atilde = real(((vsN * diagm(eigsN)) * inv(vsN)))
    if steady == 1
    Z = unew*vsN
    Zdagger = (Z'*Z) \ Z' #np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
    rhs = unew'*Yplus[:,end] #np.dot(np.matrix(unew).getH(),Yplus[:,-1])
    delta_y = (eye(size(Atilde)[1]) -  Atilde) \ (rhs)
            #np.linalg.solve(np.identity(Atilde.shape[0]) - Atilde,np.transpose(rhs))
    x_old = - (Yplus[:,K-1] - xs[:,end-1])
    #println(x_old)
    steady_update = x_old + (unew*delta_y)
    else
                Z = unew*vsN
                Zdagger = (Z'*Z)\Z' #np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
                rhs = xs[:,end-1]
        step_1 = (Zdagger * Yplus[:,end])' #np.dot(Zdagger,Yplus[:,-1]).getH()
        if (length(Atilde) > 1)
        println("$(size(step_1)) $(size( (eye(size(Atilde)[1]) - diagm(eigsN))))")
            step_2 = (eye(size(Atilde)[1]) - diagm(eigsN)) \ step_1
        else
            step_2 = step_1/1-eigsN[1,1]
           end
            #np.linalg.solve(np.identity(Atilde.shape[0])- np.diag(eigsN), step_1)
                step_3 = (eye(size(Atilde)[1]) - diagm(eigsN.^step_size)) * step_2 #np.dot(np.identity(Atilde.shape[0])- np.diag(eigsN.^step_size), step_2)
                step_4 = Z*step_3 #np.dot(Z,step_3)

                x_old = - (Yplus[:,K-1-1] - xs[:,end-1])
        nonsteady = zeros(N)
        println("$(size(step_4)) $(size(x_old))")
        nonsteady[1:N] = x_old + reshape(step_4',N)
        return nonsteady
end
    steady_update
end



function RT_Solve_DMD(Nx,Ny,hx,hy,psi_init,T_init,Nord,density,sigma_func,q_func,Cv,inveos,eos,v,dt,Tfinal,
             boundaryx, boundaryy; L2tol=1e-8, Linftol = 1e-4, maxits = 200, LOUD = 0, K = 10, Rits = 3, fname = "tmp.jld")
    time = 0.
    steady = 1
    step_size = 10 #not used
    times = []
    its = []
    done = false
    phi_old = copy(psi_init[:,:,:,1])
    psi = copy(psi_init)
    nsteps = Int64(round(Tfinal/dt))+1
    phi_t = zeros((Nx,Ny,4,nsteps))
    T_t = zeros((Nx,Ny,4,nsteps))
    phi = 0*phi_old
    #print(size(T_init))
    T = zeros((Nx,Ny,4)) + T_init
    w,etax,etay = pq.prod_quad(Nord)
    angles = length(w)
    norm_const = 1.0/sum(w)
    Qstar = 0 * q_func(0)
    step = 1
    while !(done)
        delta_t = min(Tfinal-time,dt)
        sigma = sigma_func(T,density)
        sigmat_star = sigma .+ 1/(v*delta_t)
        q = q_func(time + dt)

        #update C_v
        curr_Cv = Cv(T,density)

        beta = 4*a*c*T.^3 ./curr_Cv
        f = 1 ./ (1 .+ beta .* sigma .* delta_t)

        #update Sigma
        sigmas = (1 .- f).*sigma

        #update source
        qsource = a*c*T.^4
        for angle in 1:angles
            Qstar[:,:,:,angle] = q[:,:,:,angle] + psi[:,:,:,angle]/(v*delta_t) + f.*sigma.*qsource*norm_const
        end


        #Update Psi
        mv(phi) = reshape(SI(Nx,Ny,hx,hy,reshape(phi,(Nx,Ny,4)),Nord,sigmat_star,sigmas,q*0,boundaryx*0, boundaryy*0),
                            (Nx*Ny*4))
        #             SI(Nx,Ny,hx,hy,phi_old,        Nord,sigmat,            q,boundaryx, boundaryy)
        b = reshape(SI(Nx,Ny,hx,hy,phi_old*0,Nord,sigmat_star,sigmas,Qstar,boundaryx, boundaryy),(Nx*Ny*4))

        x = reshape(phi_old,Nx*Ny*4)


        phi,iteration = solver_with_dmd(mv, b, K, Rits, steady, x, step_size, LOUD=LOUD);


        phi = reshape(phi,(Nx,Ny,4))

        for angle in 1:angles
            Qstar[:,:,:,angle] = q[:,:,:,angle] + psi[:,:,:,angle]/(v*delta_t) + f.*sigma.*qsource*norm_const + sigmas.*norm_const.*phi
        end
        psi = SI_psi(Nx,Ny,hx,hy,phi, Nord, sigmat_star, Qstar, boundaryx, boundaryy)
        phi_old = deepcopy(phi)

        #update T
        #println(reduce(max,curr_Cv), " ", reduce(min,curr_Cv))
        T = inveos(eos(T,density) + delta_t .* f .* sigma .* (phi - qsource/norm_const),density)


        time = time + delta_t
        if (LOUD > 0) | (LOUD == -1)
            println("time = ", time)
        end
        times = push!(times,time)
        its = push!(its,iteration)
        done = time >= Tfinal

        if (LOUD == 0.25)
            figure()
            x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)
            pcolor(phi_flat)
            colorbar()
            show()
        end
        T_t[:,:,:,step] = copy(T)
        phi_t[:,:,:,step] = copy(phi)

        save(fname, "Nx", Nx, "Ny", Ny, "hx", hx, "hy", hy, "times", times, "iterations", its, "T", T_t, "phi", phi_t)
        step += 1
    end
    return psi, phi_t, T_t, time, times, its
end

