import gmsh, sys
import dolfin, time
import numpy as np
from utils import write_mesh_vtk

gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 1)


model = gmsh.model

width=1
height=2
meshsize=0.02
points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=meshsize),
        model.occ.addPoint(-width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,height/2.,0, meshSize=meshsize),
        model.occ.addPoint(width/2.,-height/2.,0, meshSize=meshsize)]
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


mesh=dolfin.Mesh()
editor=dolfin.MeshEditor()
editor.open(mesh, "triangle", 2, 3)
editor.init_vertices_global(len(nodetags), len(nodetags))
for i, tag in enumerate(nodetags):
    editor.add_vertex(i, [nodecoords[3*i],
                            nodecoords[3*i+1],
                            nodecoords[3*i+2]])
index_triangle=np.where(elementTypes==2)[0][0]
editor.init_cells_global(len(elementTags[index_triangle]),
                         len(elementTags[index_triangle]))
for i, tag in enumerate(elementTags[index_triangle]):
    editor.add_cell(i,
                    [int(elementnodeTags[index_triangle][3*i]-1),
                    int(elementnodeTags[index_triangle][3*i+1]-1),
                    int(elementnodeTags[index_triangle][3*i+2]-1)])
editor.close()

print("starting the state space")
vector_order, nodal_order = 2,2
vector_space=dolfin.FunctionSpace(mesh,'Nedelec 1st kind H(curl)', vector_order)
nodal_space=dolfin.FunctionSpace(mesh,'Lagrange',nodal_order)

combined_space=dolfin.FunctionSpace(mesh, vector_space.ufl_element() * nodal_space.ufl_element())

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
S=dolfin.PETScMatrix()
T=dolfin.PETScMatrix()
dolfin.assemble(s_ij, tensor=S)
dolfin.assemble(t_ij, tensor=T)
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
#%%
print("starting the solving")
solver=dolfin.SLEPcEigenSolver(S,T)
solver.parameters["spectral_transform"]="shift-and-invert"
solver.parameters['spectral_shift']=2.
solver.parameters['spectrum']='smallest real'

solver.solve(100)
print("solving is done")
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
mode=dolfin.Function(combined_space)
mode.vector().set_local(vectors[0])
(Te,Tm)=mode.split()

write_mesh_vtk('eigen2D.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)

with open('eigen2D.vtk','a') as f:
    f.write('POINT_DATA {:}\n'.format(mesh.num_vertices()))
    for j, vec in enumerate(vectors):
        f.write('VECTORS Te{:} double\n'.format(j))
        mode=dolfin.Function(combined_space)
        mode.vector().set_local(vec)
        (Te,Tm)=mode.split()
        values=Te.compute_vertex_values()
       
        for i in range(mesh.num_vertices()):
            f.write('{:} {:} {:}\n'.format(values[i],
                                           values[mesh.num_vertices()+i],
                                           values[2*mesh.num_vertices()+i]))
        f.write('\n')
