#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:33:18 2021

@author: usera
"""

import gmsh, os, sys
import dolfin, meshio
import numpy as np
import time
t_ini=time.time()
gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)


model = gmsh.model

width=1
height=1

points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=0.01),
        model.occ.addPoint(-width/2.,height/2.,0, meshSize=0.01),
        model.occ.addPoint(width/2.,height/2.,0, meshSize=0.01),
        model.occ.addPoint(width/2.,-height/2.,0, meshSize=0.01)]
lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loop=model.occ.addCurveLoop(lines)
rect1=model.occ.addPlaneSurface([loop])
points=[model.occ.addPoint(-width/4.,-height/4.,0, meshSize=0.02),
        model.occ.addPoint(-width/4.,height/4.,0, meshSize=0.02),
        model.occ.addPoint(width/4.,height/4.,0, meshSize=0.02),
        model.occ.addPoint(width/4.,-height/4.,0, meshSize=0.02)]
lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loop=model.occ.addCurveLoop(lines)
rect2=model.occ.addPlaneSurface([loop])

rest=model.occ.cut([[2,rect1]], [[2,rect2]], removeTool=False)


model.occ.synchronize()
model.addPhysicalGroup(2,[rect2],tag=150)
model.addPhysicalGroup(2,[rest[0][0][1]],tag=10)
groups=model.getPhysicalGroups()
entities_1=model.getEntitiesForPhysicalGroup(2,10)
entities_2=model.getEntitiesForPhysicalGroup(2,150)

# Mesh (2D)
boundaries_1=[j for i,j in model.getBoundary([[2,k] for k in entities_1])]
boundaries_2=[j for i,j in model.getBoundary([[2,k] for k in entities_2])]
boundaries_1=[k for k in boundaries_1 if k not in boundaries_2]
metal=model.addPhysicalGroup(1,boundaries_1,tag=196)
other=model.addPhysicalGroup(1,boundaries_2,tag=296)
groups=model.getPhysicalGroups()
entities_1=model.getEntitiesForPhysicalGroup(1,metal)
entities_2=model.getEntitiesForPhysicalGroup(1,other)
model.mesh.generate(2)
# Write on disk
gmsh.write("MyDisk.vtk")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("MyDisk.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write("MyDiskbis.msh")
nodes_1=model.mesh.getNodesForPhysicalGroup(1,metal)
nodes_2=model.mesh.getNodesForPhysicalGroup(1,other)
elements=model.mesh.getElements()
nodes=model.mesh.getNodes()
gmsh.finalize()

print('gmsh took {:.2f} s'.format(time.time()-t_ini))
t_ini=time.time()
#os.system('dolfin-convert MyDisk.msh testbis.xml')
meshio.read("MyDiskbis.msh").write("test.xml")
mesh=dolfin.Mesh('test.xml')
print('write read meshes took {:.2f} s'.format(time.time()-t_ini))
markers=dolfin.MeshFunction('size_t',mesh,1)
values=[0 for i in range(markers.array().shape[0])]
for line in dolfin.facets(mesh):
    k,v=line.entities(0)
    if k in nodes_1[0] and v in nodes_1[0]:
        values[line.index()]=metal
    elif k in nodes_2[0] and v in nodes_2[0]:
        values[line.index()]=other
markers.set_values(values)       
def top(x):
    return np.abs(x[0]-width/2) < dolfin.DOLFIN_EPS 

def bottom(x):
    return np.abs(x[0]+width/2) < dolfin.DOLFIN_EPS 
print('ugly stuff took {:.2f} s'.format(time.time()-t_ini))
nodal_space=dolfin.FunctionSpace(mesh,'Lagrange',1)
(L_i,)=dolfin.TestFunctions(nodal_space)
(L_j,)=dolfin.TrialFunctions(nodal_space)

bc_ground=dolfin.DirichletBC(nodal_space,dolfin.Constant(0.0), markers, other)
bc_source=dolfin.DirichletBC(nodal_space,dolfin.Constant(1.0), markers, metal)
rho=dolfin.Constant(0.0)
A_ij=dolfin.inner(dolfin.grad(L_i), dolfin.grad(L_j))*dolfin.dx
b_ij=rho*L_j*dolfin.dx

A=dolfin.assemble(A_ij)
b=dolfin.assemble(b_ij)
bc_ground.apply(A,b)
bc_source.apply(A,b)
phi=dolfin.Function(nodal_space)
c=phi.vector()
print('before solve took {:.2f} s'.format(time.time()-t_ini))
t_ini=time.time()
dolfin.solve(A,c,b)
file=dolfin.File('test.pvd')
file<<phi
print('solve and write took {:.2f} s'.format(time.time()-t_ini))