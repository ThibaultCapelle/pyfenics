#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:46:25 2021

@author: usera
"""

import gmsh
import sys
import numpy as np

def write_mesh_vtk(filename, elementTypes, elementTags, elementnodeTags,
                   nodetags, nodecoords):
    ncells, size_cells=0, 0
    for i,Type in enumerate(elementTypes):
        ncells+=len(elementTags[i])
        if Type==1:
            size_cells+=3*len(elementTags[i])
        elif Type==2:
            size_cells+=4*len(elementTags[i])
        elif Type==4:
            size_cells+=5*len(elementTags[i])
        elif Type==15:
            size_cells+=2*len(elementTags[i])
    with open(filename,'w') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write('Norm of electric field\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('POINTS {:} double\n'.format(len(nodetags)))
        for i in range(len(nodetags)):
            f.write('{:} {:} {:}\n'.format(nodecoords[3*i],
                                           nodecoords[3*i+1],
                                           nodecoords[3*i+2]))
        f.write('\n')
        f.write('CELLS {:} {:}\n'.format(ncells, size_cells))
        for i,Type in enumerate(elementTypes):
            if Type==1:
                for j in range(int(len(elementnodeTags[i])/2)):
                    f.write('2 {:} {:}\n'.format(int(elementnodeTags[i][j*2]-1),
                                                 int(elementnodeTags[i][j*2+1]-1)))
            elif Type==2:
                for j in range(int(len(elementnodeTags[i])/3)):
                    f.write('3 {:} {:} {:}\n'.format(int(elementnodeTags[i][j*3]-1),
                                             int(elementnodeTags[i][j*3+1]-1),
                                             int(elementnodeTags[i][j*3+2]-1)))
            elif Type==4:
                for j in range(int(len(elementnodeTags[i])/4)):
                    f.write('4 {:} {:} {:} {:}\n'.format(int(elementnodeTags[i][j*4]-1),
                                             int(elementnodeTags[i][j*4+1]-1),
                                             int(elementnodeTags[i][j*4+2]-1),
                                             int(elementnodeTags[i][j*4+3]-1)))
            elif Type==15:
                for j in range(len(elementnodeTags[i])):
                    f.write('1 {:}\n'.format(int(elementnodeTags[i][j]-1)))
        f.write('\n')
        f.write('CELL_TYPES {:}\n'.format(ncells))
        for i,Type in enumerate(elementTypes):
            if Type==1:
                for j in range(len(elementTags[i])):
                    f.write('3\n')
            elif Type==2:
                for j in range(len(elementTags[i])):
                    f.write('5\n')
            elif Type==15:
                for j in range(len(elementTags[i])):
                    f.write('1\n')
            elif Type==4:
                for j in range(len(elementTags[i])):
                    f.write('10\n')
        f.write('\n')

if __name__=='__main__':
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    
    model = gmsh.model
    
    width=1
    height=1
    
    points=[model.occ.addPoint(-width/2.,-height/2.,0, meshSize=0.2),
            model.occ.addPoint(-width/2.,height/2.,0, meshSize=0.2),
            model.occ.addPoint(width/2.,height/2.,0, meshSize=0.2),
            model.occ.addPoint(width/2.,-height/2.,0, meshSize=0.2)]
    lines=[model.occ.addLine(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
    loop=model.occ.addCurveLoop(lines)
    rect1=model.occ.addPlaneSurface([loop])
    model.occ.synchronize()
    model.mesh.generate(2)
    elementTypes, elementTags, elementnodeTags = model.mesh.getElements()
    nodetags, nodecoords, _ = model.mesh.getNodes()
    gmsh.write("export.vtk")
    gmsh.finalize()
    
    write_mesh_vtk('exportbis.vtk',  elementTypes, elementTags, elementnodeTags,
                       nodetags, nodecoords)