#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:31:18 2021

@author: usera
"""

import gmsh, sys, time
import dolfin
import numpy as np
from utils import Expression, DOLFIN_Mesh, addBox, addRectangle, integrate
model=gmsh.model
gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
gmsh.option.setNumber("Mesh.SaveAll", 0)
box=addBox(0,0,0,1,1,1, meshsize=0.1)
box2=addBox(0,0,-0.5,1,1,0.5, meshsize=0.1)
rect=addRectangle(0,0,0,0.1,0.8, meshsize=0.01)

integrate([[3,box]], [[2,rect]])
model.occ.removeAllDuplicates()
integrate([[3,box2]], [[2,rect]])
model.mesh.generate(3)
gmsh.write('bilayer_with_circuit_mesh.vtk')
gmsh.finalize()