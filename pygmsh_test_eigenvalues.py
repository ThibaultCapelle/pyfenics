import gmsh, sys
import dolfin, meshio
import numpy as np
from test_export import write_mesh_vtk

gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 1)


model = gmsh.model

width=1
height=1

points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=0.1),
        model.occ.addPoint(-width/2.,height/2.,0, meshSize=0.1),
        model.occ.addPoint(width/2.,height/2.,0, meshSize=0.1),
        model.occ.addPoint(width/2.,-height/2.,0, meshSize=0.1)]
lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loop=model.occ.addCurveLoop(lines)
rect1=model.occ.addPlaneSurface([loop])
model.occ.synchronize()
model.mesh.generate(2)
elementTypes, elementTags, elementnodeTags = model.mesh.getElements()
nodetags, nodecoords, _ = model.mesh.getNodes()
gmsh.write("eigen.vtk")
gmsh.write("eigen.msh")
gmsh.finalize()
meshio.read("eigen.msh").write("eigen.xml")

mesh=dolfin.Mesh('eigen.xml')

#width=1
#height=1
#mesh=dolfin.RectangleMesh(dolfin.Point(0,0,0),dolfin.Point(width,height,0),8,4)

print("starting the state space")
vector_order, nodal_order = 2,2
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
er=1.
ur=1.

s_tt_ij=1./ur*dolfin.inner(dolfin.curl(N_i), dolfin.curl(N_j))
t_tt_ij=er*dolfin.inner(N_i, N_j) 
s_zz_ij=1./ur*dolfin.inner(dolfin.grad(L_i), dolfin.grad(L_j))
t_zz_ij=er*L_i*L_j
s_ij=(s_tt_ij+s_zz_ij)*dolfin.dx
t_ij=(t_tt_ij+t_zz_ij)*dolfin.dx
S=dolfin.assemble(s_ij)
T=dolfin.assemble(t_ij)
print("starting the boundary conditions")
markers=dolfin.MeshFunction('size_t',mesh,1)

markers.set_all(0)
dolfin.DomainBoundary().mark(markers, 1)
electric_wall = dolfin.DirichletBC(combined_space, 
                                   dolfin.Constant((0.0,0.0,0.0,0.0)),
                                   markers,
                                   1)
electric_wall.apply(S)
electric_wall.apply(T)

indicators=np.ones(S.size(0))
keys=[k for k in electric_wall.get_boundary_values().keys()]
indicators[keys]=0
free_dofs=np.where(indicators==1)[0]
S_np=S.array()[free_dofs,:][:,free_dofs]
T_np=T.array()[free_dofs,:][:,free_dofs]

print("starting the solving")
from scipy.linalg import eig
kc_squared,ev=eig(S_np, T_np)
sort_index=np.argsort(kc_squared)
first_mode_idx=np.where(kc_squared[sort_index]>1e-8)[0][0]

print("The cutoff frequencies of the 4 dominant modes are :")
print(kc_squared[sort_index][first_mode_idx:first_mode_idx+4])
mode_idx=0
coefficients_global=np.zeros(S.size(0))
coefficients_global[free_dofs]=ev[:,sort_index[first_mode_idx+mode_idx]]
mode=dolfin.Function(combined_space)
mode.vector().set_local(coefficients_global)

(Te,Tm)=mode.split()
#dolfin.plot(Te)


x,y,z=[[v.x(i) for v in dolfin.vertices(mesh)] for i in range(3)]
'''Tevaltemp=np.reshape(Te.compute_vertex_values(),(mesh.num_vertices(),2))
Teval=np.zeros((mesh.num_vertices(),3))
Teval[:,:2]=Tevaltemp'''
Teval=np.reshape(Te.compute_vertex_values(),(mesh.num_vertices(),3))
norm=np.sum(np.abs(Teval)**2, axis=1)

write_mesh_vtk('eigen.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)

with open('eigen.vtk','a') as f:
    f.write('POINT_DATA {:}\n'.format(mesh.num_vertices()))
    f.write('VECTORS Te double\n')
    #f.write('SCALARS norm double 1\n')
    #f.write('LOOKUP_TABLE default\n')
    for i in range(mesh.num_vertices()):
        f.write('{:} {:} {:}\n'.format(Te.compute_vertex_values()[i],
                                       Te.compute_vertex_values()[mesh.num_vertices()+i],
                                       Te.compute_vertex_values()[2*mesh.num_vertices()+i]))
    f.write('\n')
