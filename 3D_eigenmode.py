#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:35:53 2021

@author: usera
"""

import gmsh, sys, time
import dolfin, os
import numpy as np
from utils import write_mesh_vtk
print('start')
t_ini=time.time()
gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
gmsh.option.setNumber("Mesh.SaveAll", 0)

SI=140
AIR=150
BBOX=160
er_dict=dict({'Air':1.0,
         'Si':11.7})
model = gmsh.model

width=0.7
height=0.7
thick_air=0.5
thick_Si=0.2
meshsize=0.05
points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=meshsize),
        model.occ.addPoint(-width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,-height/2.,0, meshSize=meshsize)]

lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loop=model.occ.addCurveLoop(lines)
rect1=model.occ.addPlaneSurface([loop])
box1=model.occ.extrude([[2,rect1]],0,0,thick_air)
for b in box1:
    if b[0]==3:
        box1=b[1]
        break
model.occ.synchronize()
model.addPhysicalGroup(3,[box1],tag=AIR)

print('before generating mesh : {:} s'.format(time.time()-t_ini))
model.mesh.generate(3)
print('after generating mesh : {:} s'.format(time.time()-t_ini))

elementTypes, elementTags, elementnodeTags = model.mesh.getElements()
nodetags, nodecoords, _ = model.mesh.getNodes()
gmsh.write("eigen3d.vtk")

gmsh.finalize()
print('after gmsh : {:} s'.format(time.time()-t_ini))

mesh=dolfin.Mesh()
editor=dolfin.MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices_global(len(nodetags), len(nodetags))
for i, tag in enumerate(nodetags):
    editor.add_vertex(i, [nodecoords[3*i],
                            nodecoords[3*i+1],
                            nodecoords[3*i+2]])
print('after mesh vertices import: {:} s'.format(time.time()-t_ini))
index_tetra=np.where(elementTypes==4)[0][0]
editor.init_cells_global(len(elementTags[index_tetra]),
                         len(elementTags[index_tetra]))
for i, tag in enumerate(elementTags[index_tetra]):

    editor.add_cell(i,
                    [int(elementnodeTags[index_tetra][4*i]-1),
                    int(elementnodeTags[index_tetra][4*i+1]-1),
                    int(elementnodeTags[index_tetra][4*i+2]-1),
                    int(elementnodeTags[index_tetra][4*i+3]-1)])
editor.close()
print('after mesh import: {:} s'.format(time.time()-t_ini))


print("starting the state space")
vector_order, nodal_order = 1,1
mesh.init()
vector_space=dolfin.FunctionSpace(mesh,'Nedelec 1st kind H(curl)', vector_order)
#nodal_space=dolfin.FunctionSpace(mesh,'Lagrange',nodal_order)
#combined_space=dolfin.FunctionSpace(mesh, vector_space.ufl_element() * nodal_space.ufl_element())
(N_i,)=dolfin.TestFunctions(vector_space)
(N_j,)=dolfin.TrialFunctions(vector_space)


print('made functions')
er=1.
ur=1.

s_ij=1./ur*dolfin.inner(dolfin.curl(N_i), dolfin.curl(N_j))*dolfin.dx
t_ij=er*dolfin.inner(N_i, N_j)*dolfin.dx

S=dolfin.PETScMatrix()
T=dolfin.PETScMatrix()
print('before assemble: {:}s'.format(time.time()-t_ini))
dolfin.assemble(s_ij, tensor=S)
print('first assembly')
dolfin.assemble(t_ij, tensor=T)


print("starting the boundary conditions")
t_ini=time.time()
markers=dolfin.MeshFunction('size_t',mesh,2 )
markers.set_all(0)
dolfin.DomainBoundary().mark(markers,1)

electric_wall = dolfin.DirichletBC(vector_space, 
                                   dolfin.Constant((0.0,0.0,0.0)),
                                   markers,
                                   1)
electric_wall.apply(S)
electric_wall.apply(T)
print('boundary conditions took {:} s'.format(time.time()-t_ini))
#%%
print("starting the solving")
solver=dolfin.SLEPcEigenSolver(S,T)
solver.parameters["spectral_transform"]="shift-and-invert"
solver.parameters['spectral_shift']=50.
solver.parameters['spectrum']='smallest real'

n_to_solve = 0
lambdas=[]
n_converged=0
while (n_converged==0 or len(lambdas)==0) and n_to_solve<100:
    n_to_solve+=1
    print('n_to_solve:{:}'.format(n_to_solve))
    t_ini=time.time()
    solver.solve(n_to_solve)
    print("solving is done, converged {:}, time: {:} s".format(solver.get_number_converged(),
                                                                  time.time()-t_ini))
    t_ini=time.time()
    n_converged=solver.get_number_converged()
    lambdas=[solver.get_eigenvalue(i) for i in range(n_converged)]
    print('first part of stupid stuff took {:} s'.format(time.time()-t_ini))
    lambdas=[lambdas[i] for i in np.where(np.abs(lambdas)>1.5)[0]]
    print('stupid stuff took {:} s'.format(time.time()-t_ini))
#%%
#print("solving is done, converged {:}".format(solver.get_number_converged()))
t_ini=time.time()
lambdas, vectors=np.empty(solver.get_number_converged()), []
for i in range(solver.get_number_converged()):
    r,c,RV,IV=solver.get_eigenpair(i)
    lambdas[i]=r
    vectors.append(RV)
print('first part of ugly stuff took {:} s'.format(time.time()-t_ini))
vectors=[vectors[i] for i in np.where(np.abs(lambdas)>1.5)[0]]
lambdas=lambdas[np.where(np.abs(lambdas)>1.5)[0]]
index=np.argsort(lambdas)
vectors=[vectors[i] for i in index]
lambdas=lambdas[index]
print('second part of ugly stuff took {:} s'.format(time.time()-t_ini))
print('ugly stuff took {:} s'.format(time.time()-t_ini))
#%%

write_mesh_vtk('eigen3D.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)

with open('eigen3D.vtk','a') as f:
    f.write('POINT_DATA {:}\n'.format(mesh.num_vertices()))
    for j, vec in enumerate(vectors):
        f.write('VECTORS Te{:} double\n'.format(j))
        mode=dolfin.Function(vector_space)
        mode.vector().set_local(vec)
        #(Te,Tm)=mode.split()
        values=mode.compute_vertex_values()
       
        for i in range(mesh.num_vertices()):
            f.write('{:} {:} {:}\n'.format(values[i],
                                           values[mesh.num_vertices()+i],
                                           values[2*mesh.num_vertices()+i]))
        f.write('\n')