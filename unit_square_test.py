#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:20:53 2021

@author: usera
"""

import dolfin
import numpy as np
import matplotlib.pylab as plt

N=20
N_tot=(N+1)*(N+1)
#==================Real solutions===========================

lambds_real=np.unique(np.sort(np.array([i**2+j**2 for i in range(N_tot)\
                              for j in range(N_tot)])))



#=================Finite Element solutions==================




mesh=dolfin.RectangleMesh(dolfin.Point(0,0), dolfin.Point(np.pi,np.pi),N,N)

V = dolfin.FunctionSpace(mesh, "Lagrange", 1)
v = dolfin.TestFunction(V)
u = dolfin.TrialFunction(V)

s = dolfin.inner(dolfin.curl(v), dolfin.curl(u))*dolfin.dx
t = dolfin.inner(v, u)*dolfin.dx

S = dolfin.PETScMatrix()
T = dolfin.PETScMatrix()

dolfin.assemble(s, tensor=S)
dolfin.assemble(t, tensor=T)
print(S.size(1))

markers=dolfin.MeshFunction('size_t',mesh,1)
markers.set_all(0)
dolfin.DomainBoundary().mark(markers,1)

electric_wall = dolfin.DirichletBC(V, 
                                   dolfin.Constant(0.0),
                                   markers,
                                   1)
electric_wall.apply(S)
electric_wall.apply(T)

#from scipy.linalg import eig
#lambds, vectors=eig(S.array(),T.array())
# Solve the eigensystem
esolver = dolfin.SLEPcEigenSolver(S,T)
esolver.solve(S.size(1))

res=[esolver.get_eigenpair(i) for i in range(esolver.get_number_converged())]
res.sort(key=lambda tup: tup[0])
lambds=np.array([r[0] for r in res])
vect=[r[2] for r in res]
#lambds=np.sort(np.array(lambds))
plt.close('all')
plt.figure()
plt.plot(lambds, '.', color='g')
for lambd in lambds_real[:20]:
    plt.plot([0,len(lambds)],[lambd, lambd], '--', color='r')
plt.ylim([-1,lambds_real[20]+1])