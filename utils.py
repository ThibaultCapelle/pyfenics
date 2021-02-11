#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:38:41 2021

@author: usera
"""

import dolfin

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
            f.write('{:} {:} {:}\n'.format(float(nodecoords[3*i]),
                                           float(nodecoords[3*i+1]),
                                           float(nodecoords[3*i+2])))
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

class Expression(dolfin.UserExpression):
    
    def __init__(self, marker, **kwargs):
        self.marker=marker
        super().__init__(**kwargs)
        pass
    
    def eval_cell(self, values, x, cell):
        values[0]=self.marker[cell.index]