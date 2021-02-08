#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:35:53 2021

@author: usera
"""

import gmsh, sys, time
import dolfin
import numpy as np
from test_export import write_mesh_vtk
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
box1=model.occ.extrude([[2,rect1]],0,0,thick_air+thick_Si)
for b in box1:
    if b[0]==3:
        box1=b[1]
        break

points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=meshsize),
        model.occ.addPoint(-width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,-height/2.,0, meshSize=meshsize)]

lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loop=model.occ.addCurveLoop(lines)
rect2=model.occ.addPlaneSurface([loop])
box2=model.occ.extrude([[2,rect2]],0,0,thick_Si)
for b in box2:
    if b[0]==3:
        box2=b[1]
        break
box1=model.occ.cut([[3,box1]], [[3,box2]], removeTool=False)
for b in box1[0]:
    if b[0]==3:
        box1=b[1]
        break
model.occ.synchronize()    
model.addPhysicalGroup(3,[box2],tag=SI)
model.addPhysicalGroup(3,[box1],tag=AIR)

entities_in_BBox=model.getEntitiesInBoundingBox(-width, -height,thick_Si*9/10,
                                                width, height, thick_Si+thick_air/10, dim=2)
entities=model.getEntities(dim=2)
boundaries_entities=[k[1] for k in entities if k not in entities_in_BBox]
model.addPhysicalGroup(2,boundaries_entities,tag=BBOX)
print('before generating mesh : {:} s'.format(time.time()-t_ini))
model.mesh.generate(3)
print('after generating mesh : {:} s'.format(time.time()-t_ini))
(boundaries_elements, elements_2D)=[], []
entities=model.getEntitiesForPhysicalGroup(2,BBOX)
entities_2D=model.getEntities(dim=2)
for e in entities_2D:
    elements_2D+=list(model.mesh.getElements(dim=2, tag=e[1])[1][0])
for e in entities:
    boundaries_elements+=list(model.mesh.getElements(dim=2, tag=e)[1][0])
boundaries_nodes=dict()
for entity in boundaries_entities:
    res=gmsh.model.mesh.getElements(tag=entity,dim=2)    
    for i, tag in enumerate(res[1][0]):
        boundaries_nodes[tag]=(int(res[2][0][3*i]-1),
                               int(res[2][0][3*i+1]-1),
                               int(res[2][0][3*i+2]-1))
    
t_ini_bis=time.time()
cell_physical=dict({'Si':dict(),
                         'Air':dict()})
entities_physical=dict({'Air':model.getEntitiesForPhysicalGroup(3,AIR),
                        'Si':model.getEntitiesForPhysicalGroup(3,SI)})
for e in entities_physical['Si']:
    elementTypes, elementTags, elementnodeTags = model.mesh.getElements(dim=3, tag=e)
    for i, tag in enumerate(elementTags[0]):
        cell_physical['Si'][tag]=[elementnodeTags[0][4*i],
                                             elementnodeTags[0][4*i+1],
                                             elementnodeTags[0][4*i+2],
                                             elementnodeTags[0][4*i+3]]
for e in entities_physical['Air']:
    elementTypes, elementTags, elementnodeTags = model.mesh.getElements(dim=3, tag=e)
    for i, tag in enumerate(elementTags[0]):
        cell_physical['Air'][tag]=[elementnodeTags[0][4*i],
                                             elementnodeTags[0][4*i+1],
                                             elementnodeTags[0][4*i+2],
                                             elementnodeTags[0][4*i+3]]
print('filling dictionaries took {:} s'.format(time.time()-t_ini_bis))
elementTypes, elementTags, elementnodeTags = model.mesh.getElements()
nodetags, nodecoords, _ = model.mesh.getNodes()
gmsh.write("bilayer.vtk")
gmsh.finalize()
print('after gmsh : {:} s'.format(time.time()-t_ini))



t_ini_bis=time.time()
mesh=dolfin.Mesh()
editor=dolfin.MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices_global(len(nodetags), len(nodetags))
nodemap, nodemap_inv, cellmap, cellmap_inv=dict(), dict(), dict(), dict()
for i, tag in enumerate(nodetags):
    nodemap[i]=tag
    nodemap_inv[tag]=i
    editor.add_vertex(i, [nodecoords[3*i],
                            nodecoords[3*i+1],
                            nodecoords[3*i+2]])
print('after mesh vertices import: {:} s'.format(time.time()-t_ini_bis))
t_ini_bis=time.time()
index_tetra=np.where(elementTypes==4)[0][0]
editor.init_cells_global(len(elementTags[index_tetra]),
                         len(elementTags[index_tetra]))
cell_physical_dolfin=dict({'Si':[],
                           'Air':[]})
i=0
for material in ['Air', 'Si']:
    for tag, nodes in cell_physical[material].items():
        cellmap[i]=tag
        cellmap_inv[tag]=i
        editor.add_cell(i, [nodemap_inv[node] for node in nodes])
        i+=1

editor.close()
print('after mesh import: {:} s'.format(time.time()-t_ini_bis))
boundary_facets=[]

class MyDict(dict):
    def get(self, key):
        return dict.get(self, sorted(key))
mesh.init(2,0)
v_2_f = MyDict((tuple(facet.entities(0)), facet.index())
             for facet in dolfin.facets(mesh))
for f in boundaries_nodes.values():
    boundary_facets.append(v_2_f[tuple(sorted(f))])

t_ini=time.time()
print("starting the state space")
vector_order, nodal_order = 1,1
mesh.init()
vector_space=dolfin.FunctionSpace(mesh,'Nedelec 1st kind H(curl)', vector_order)
(N_i,)=dolfin.TestFunctions(vector_space)
(N_j,)=dolfin.TrialFunctions(vector_space)


print('made functions')
er=1.
ur=1.

ermarkers=dolfin.MeshFunction('double', mesh, 3)
vals=np.zeros(mesh.num_cells())
for material in cell_physical.keys():
    for cell_tag in cell_physical[material].keys():
        vals[cellmap_inv[cell_tag]]=er_dict[material]
ermarkers.set_values(vals)
class Er(dolfin.UserExpression):
    
    def __init__(self, marker, **kwargs):
        self.marker=marker
        super().__init__(**kwargs)
        pass
    
    def eval_cell(self, values, x, cell):
        values[0]=self.marker[cell.index]
er=Er(ermarkers)

s_ij=1./ur*dolfin.inner(dolfin.curl(N_i), dolfin.curl(N_j))*dolfin.dx
t_ij=er*dolfin.inner(N_i, N_j)*dolfin.dx

S=dolfin.PETScMatrix()
T=dolfin.PETScMatrix()
print('before assemble: {:}s'.format(time.time()-t_ini))
dolfin.assemble(s_ij, tensor=S)
print('first assembly')
dolfin.assemble(t_ij, tensor=T)

#%%


print("starting the boundary conditions")
t_ini=time.time()
markers=dolfin.MeshFunction('size_t',mesh,2 )
vals=np.zeros(markers.array().shape, dtype=int)
vals[boundary_facets]=int(BBOX)
markers.set_values(vals)
electric_wall = dolfin.DirichletBC(vector_space, 
                                   dolfin.Constant((0.0,0.0,0.0)),
                                   markers,
                                   BBOX)
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

write_mesh_vtk('bilayer.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)

with open('bilayer.vtk','a') as f:
    f.write('POINT_DATA {:}\n'.format(mesh.num_vertices()))
    for j, vec in enumerate(vectors):
        f.write('VECTORS Te{:} double\n'.format(j))
        mode=dolfin.Function(vector_space)
        mode.vector().set_local(vec)
        values=mode.compute_vertex_values()
       
        for i in range(mesh.num_vertices()):
            f.write('{:} {:} {:}\n'.format(values[i],
                                           values[mesh.num_vertices()+i],
                                           values[2*mesh.num_vertices()+i]))
        f.write('\n')