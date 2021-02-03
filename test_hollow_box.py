#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:09:08 2021

@author: usera
"""

import dolfin
import numpy as np


width=1.
height=1.
thick=1.
mesh=dolfin.BoxMesh(dolfin.Point(-width/2.,-height/2.,-thick/2.),
                    dolfin.Point(width/2.,height/2.,thick/2.),
                    8,8,8)
V = dolfin.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 2)
v = dolfin.TestFunction(V)
u = dolfin.TrialFunction(V)
s = dolfin.inner(dolfin.curl(v), dolfin.curl(u))*dolfin.dx
t = dolfin.inner(v, u)*dolfin.dx

# Assemble the stiffness matrix (S) and mass matrix (T)
S = dolfin.PETScMatrix()
T = dolfin.PETScMatrix()

dolfin.assemble(s, tensor=S)
dolfin.assemble(t, tensor=T)
print(S.size(1))
# Solve the eigensystem
esolver = dolfin.SLEPcEigenSolver(S,T)
#esolver.parameters["spectrum"]="smallest real"
#esolver.set("eigenvalue spectrum", "smallest real")
esolver.solve(S.size(1))

cutoff = None
for i in range(S. size(1)):
    (lr, lc) = esolver.get_eigenvalue(i)
#    print N.complex(lr, lc)
    print(np.sqrt(lr))
    if lr > 1 and lc == 0:
        cutoff = np.sqrt(lr)
        break
    
if cutoff is None:
    print("Unable to find dominant mode")
else:
    print("Cutoff frequency:" +str(cutoff))