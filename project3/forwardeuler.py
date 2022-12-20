# -*- coding: utf-8 -*-
"""
FYS-STK4155

Project 3

Solving diffusion eq. using explicit forward Euler
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

dx_list = [1/10, 1/100] # spatial step lengths
T = 1. # total integration time

for dx in dx_list:
    dt = 0.5*dx**2 # stability criterion dt/dx^2 <= 1/2
    
    Nx = int(1./dx) + 1 # no. of spatial points
    x = np.linspace(0.,1.,Nx)
    
    Nt = int(T/dt) + 1 # no. of time points
    t = np.linspace(0.,T,Nt)

    # initialise matrix to contain solution. BCs = 0 for all t.
    U = np.zeros(shape=(Nt, Nx))
    U[0] = np.sin(np.pi*x) # initial condition
    
    # initialise matrix describing system of difference equations
    A = -2*np.eye(Nx-2) 
    for k in range(Nx-2-1):
        A[k,k+1] = 1 # superdiagonal = 1
        A[k+1,k] = 1 # subdiagonal = 1

    for j in range(Nt-1):
        # evolve system
        U[j+1,1:-1] = U[j,1:-1] + 0.5*A@U[j,1:-1].T
    
    # choose two timepoints to inspect
    idx_1 = Nt//10
    idx_2 = Nt//2
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    for idx in [idx_1,idx_2]:
        ax.plot(x, U[idx], label=f"t = {np.round(t[idx],2)}")
        anal_sol = np.exp(-np.pi**2 * t[idx])*np.sin(np.pi*x)
        ax.plot(x, anal_sol, linestyle="--", label="analytic")
        
        # calculate error
        mean_rel_err = np.mean(np.abs( (U[idx,1:-1] - anal_sol[1:-1])/anal_sol[1:-1] ))
        print(f"Mean relative error at t={np.round(t[idx],2)}:", mean_rel_err)
    
    ax.set_title(f"$\Delta x =$ 1/{int(1/dx)}")
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    #fig.savefig(f"forwardeuler_{dx}.pdf")
    
    x_grid, t_grid = np.meshgrid(x,t)
    fig1 = plt.figure(figsize=(6,6))
    ax1 = fig1.gca(projection='3d')
    ax1.plot_surface(x_grid,t_grid,U,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax1.set_xlabel('Position $x$')
    ax1.set_ylabel('Time $t$')
    ax1.view_init(30,60)
    ax1.set_title(f"$\Delta x =$ 1/{int(1/dx)}")
    #fig1.savefig(f"FE_full_sol_{dx}.pdf")
    plt.show()