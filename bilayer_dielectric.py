#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:35:53 2021

@author: usera
"""

import gmsh, sys, time
import dolfin, os
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
#model.occ.synchronize()
#model.addPhysicalGroup(3,[box1],tag=AIR)

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
#gmsh.write("bilayer.msh")
gmsh.finalize()
print('after gmsh : {:} s'.format(time.time()-t_ini))


#meshio.read("eigen3d.msh").write("eigen3d.xml")
#os.system('dolfin-convert eigen3d.msh eigen3d.xml')
#print('after dolfin_convert : {:} s'.format(time.time()-t_ini))
#mesh=dolfin.Mesh('eigen3d.xml')
t_ini_bis=time.time()
mesh=dolfin.Mesh()
editor=dolfin.MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices_global(len(nodetags), len(nodetags))
nodemap, nodemap_inv, cellmap=dict(), dict(), dict()
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
for f in boundaries_nodes.values():parameters
    boundary_facets.append(v_2_f[tuple(sorted(f))])

t_ini=time.time()
print("starting the state space")
vector_order, nodal_order = 1,1
mesh.init()
vector_space=dolfin.FunctionSpace(mesh,'Nedelec 1st kind H(curl)', vector_order)
#vector_element = dolfin.VectorElement("Nedelec 1st kind H(curl)", mesh.ufl_cell(), vector_order)
nodal_space=dolfin.FunctionSpace(mesh,'Lagrange',nodal_order)
#nodal_element = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), nodal_order)
#combined_space = vector_space * nodal_space
combined_space=dolfin.FunctionSpace(mesh, vector_space.ufl_element() * nodal_space.ufl_element())
#combined_element=dolfin.MixedElement([vector_element, nodal_element])
#combined_space=dolfin.FunctionSpace(mesh, combined_element)
#combined_space=dolfin.FunctionSpace(mesh, dolfin.MixedElement([vector_element, nodal_element]))
(N_i, L_i)=dolfin.TestFunctions(combined_space)
(N_j, L_j)=dolfin.TrialFunctions(combined_space)
print('made functions')
er=1.
ur=1.

'''ermarkers=dolfin.MeshFunction('size_t', mesh, 3)
vals=np.zeros(mesh.num_cells())
offset=np.min([np.min(v) for v in cell_physical.values()])
for k,v in cell_physical.items():
    for jj in v:
        vals[int(jj-1-offset)]=er_dict[k]parameters
ermarkers.set_values(vals)
class Er(dolfin.UserExpression):
    
    def __init__(self, marker, **kwargs):
        self.marker=marker
        super().__init__(**kwargs)
        pass
    
    def eval_cell(self, values, x, cell):
        return self.marker[cell.index]
er=Er(ermarkers)'''  

er=1.    
s_tt_ij=1./ur*dolfin.inner(dolfin.curl(N_i), dolfin.curl(N_j))
t_tt_ij=er*dolfin.inner(N_i, N_j)
s_zz_ij=1./ur*dolfin.inner(dolfin.grad(L_i), dolfin.grad(L_j))
t_zz_ij=er*L_i*L_j
s_ij=(s_tt_ij+s_zz_ij)*dolfin.dx
t_ij=(t_tt_ij+t_zz_ij)*dolfin.dx
#S=dolfin.PETScMatrix()
#T=dolfin.PETScMatrix()
print('before assemble: {:}s'.format(time.time()-t_ini))
S=dolfin.assemble(s_ij)#, tensor=S)
T=dolfin.assemble(t_ij)#, tensor=T)


print("starting the boundary conditions")
t_ini=time.time()
markers=dolfin.MeshFunction('size_t',mesh,2 )
#markers.set_all(0)
#dolfin.DomainBoundary().mark(markers,1)'''
vals=np.zeros(markers.array().shape, dtype=int)
vals[boundary_facets]=int(BBOX)
markers.set_values(vals)
electric_wall = dolfin.DirichletBC(combined_space, 
                                   dolfin.Constant((0.0,0.0,0.0,0.0)),
                                   markers,
                                   BBOX)
electric_wall.apply(S)
electric_wall.apply(T)
print('boundary conditions took {:} s'.format(time.time()-t_ini))


indicators=np.ones(S.size(0))
keys=[k for k in electric_wall.get_boundary_values().keys()]
indicators[keys]=0
free_dofs=np.where(indicators==1)[0]
S_np=S.array()[free_dofs,:][:,free_dofs]
T_np=T.array()[free_dofs,:][:,free_dofs]
#S.set(S_np)
#T.set(T_np)

print("starting the solving")
from scipy.linalg import eig
kc_squared,ev=eig(S_np, T_np)
sort_index=np.argsort(kc_squared)
first_mode_idx=np.where(kc_squared[sort_index]>1e-8)[0][0]


'''solver=dolfin.SLEPcEigenSolver(S,T)
solver.solve()
r, c, RV, IV = solver.get_eigenpair(solver.get_number_converged()-1)
mode=dolfin.Function(combined_space)
mode.vector().set_local(RV)
(Te,Tm)=mode.split()'''

print("The cutoff frequencies of the 4 dominant modes are :")
print(kc_squared[sort_index][first_mode_idx:first_mode_idx+4])
mode_idx=1
coefficients_global=np.zeros(S.size(0))
coefficients_global[free_dofs]=ev[:,sort_index[first_mode_idx+mode_idx]]

mode=dolfin.Function(combined_space)
mode.vector().set_local(coefficients_global)

(Te,Tm)=mode.split()
#dolfin.plot(Te)



write_mesh_vtk('eigen3D.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)

with open('eigen3D.vtk','a') as f:
    f.write('POINT_DATA {:}\n'.format(mesh.num_vertices()))
    f.write('VECTORS Te double\n')
    #f.write('SCALARS norm double 1\n')
    #f.write('LOOKUP_TABLE default\n')
    for i in range(mesh.num_vertices()):
        f.write('{:} {:} {:}\n'.format(Te.compute_vertex_values()[i],
                                       Te.compute_vertex_values()[mesh.num_vertices()+i],
                                       Te.compute_vertex_values()[2*mesh.num_vertices()+i]))
    f.write('\n')