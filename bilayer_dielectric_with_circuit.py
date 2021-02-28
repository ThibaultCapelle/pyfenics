#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:35:53 2021

@author: usera
"""

import gmsh, sys
import numpy as np
from utils import Expression, DOLFIN_Mesh, addBox

gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
gmsh.option.setNumber("Mesh.SaveAll", 0)

model = gmsh.model
x,y,z=0,0,0
dx,dy,dz=1.,1.,1.
meshsize=0.2
A,B,C,D,E,F,G,H=[model.occ.addPoint(x-dx/2.,y-dy/2.,z, meshSize=meshsize),
                 model.occ.addPoint(x-dx/2.,y+dy/2.,z, meshSize=meshsize),
                 model.occ.addPoint(x+dx/2.,y+dy/2.,z, meshSize=meshsize),
                 model.occ.addPoint(x+dx/2.,y-dy/2.,z, meshSize=meshsize),
                 model.occ.addPoint(x-dx/2.,y-dy/2.,z+dz, meshSize=meshsize),
                 model.occ.addPoint(x-dx/2.,y+dy/2.,z+dz, meshSize=meshsize),
                 model.occ.addPoint(x+dx/2.,y+dy/2.,z+dz, meshSize=meshsize),
                 model.occ.addPoint(x+dx/2.,y-dy/2.,z+dz, meshSize=meshsize)]
AB=model.occ.addLine(A,B)
BC=model.occ.addLine(B,C)
CD=model.occ.addLine(C,D)
DA=model.occ.addLine(D,A)
AE=model.occ.addLine(A,E)
EF=model.occ.addLine(E,F)
FG=model.occ.addLine(F,G)
GH=model.occ.addLine(G,H)
HE=model.occ.addLine(H,E)
BF=model.occ.addLine(B,F)
CG=model.occ.addLine(C,G)
DH=model.occ.addLine(D,H)
loop1=model.occ.addCurveLoop([AB,BC,CD,DA])
loop2=model.occ.addCurveLoop([AB,BF,-EF,-AE])
loop3=model.occ.addCurveLoop([EF,FG,GH,HE])
loop4=model.occ.addCurveLoop([BF,FG,-CG,-BC])
loop5=model.occ.addCurveLoop([CG,GH,-DH,-CD])
loop6=model.occ.addCurveLoop([DH,HE,-AE,-DA])

dx,dy=0.1,0.8
meshsize=0.01
points=[model.occ.addPoint(x-dx/2.,y-dy/2.,z, meshSize=meshsize),
        model.occ.addPoint(x-dx/2.,y+dy/2.,z, meshSize=meshsize),
        model.occ.addPoint(x+dx/2.,y+dy/2.,z, meshSize=meshsize),
        model.occ.addPoint(x+dx/2.,y-dy/2.,z, meshSize=meshsize)]
lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
loophole=model.occ.addCurveLoop(lines)
recthole=model.occ.addPlaneSurface([loophole])
rect1=model.occ.addPlaneSurface([loop1,loophole])
rect2=model.occ.addPlaneSurface([loop2])
rect3=model.occ.addPlaneSurface([loop3])
rect4=model.occ.addPlaneSurface([loop4])
rect5=model.occ.addPlaneSurface([loop5])
rect6=model.occ.addPlaneSurface([loop6])
model.occ.addVolume([model.occ.addSurfaceLoop([rect1,recthole,rect2,rect3,rect4,rect5,rect6])])

model.occ.synchronize() 
model.mesh.generate(3)
entities=model.getEntities()
gmsh.write('circuit_bis.vtk')
gmsh.finalize()
