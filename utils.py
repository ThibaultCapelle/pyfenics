#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:38:41 2021

@author: usera
"""

import dolfin, gmsh
import numpy as np

#This function should be called within gmsh


class GMSH_Mesh:
    
    def __init__(self, physical_dictionary=None):
        self.elementTypes, self.elementTags, self.elementnodeTags = gmsh.model.mesh.getElements()
        self.nodetags, self.nodecoords, _ = gmsh.model.mesh.getNodes()
        self.physical_dictionary=physical_dictionary
        self.cell_physical=self.get_cell_physical()
        
    def get_cell_physical(self, dim=3):
        if self.physical_dictionary is not None:
            material_names, material_tags =([k for k in self.physical_dictionary.keys()],
                                            [v for v in self.physical_dictionary.values()])
            cell_physical=dict()
            model=gmsh.model
            for material_name, material_tag in zip(material_names, material_tags):
                cell_physical[material_name]=dict()
                for e in model.getEntitiesForPhysicalGroup(dim, material_tag):
                    elementTypes, elementTags, elementnodeTags = model.mesh.getElements(dim=dim, tag=e)
                    for i, tag in enumerate(elementTags[0]):
                        cell_physical[material_name][tag]=[elementnodeTags[0][4*i],
                                                             elementnodeTags[0][4*i+1],
                                                             elementnodeTags[0][4*i+2],
                                                             elementnodeTags[0][4*i+3]]
            return cell_physical
        else:
            return None
        
class DOLFIN_Mesh:
    
    def __init__(self, geometry=None, physical_dictionary=None):
        if geometry is None:
            self.geometry=GMSH_Mesh(physical_dictionary=physical_dictionary)
        else:
            self.geometry=geometry
        self.cell_physical=self.geometry.cell_physical
        self.create_mesh()
        
    def create_mesh(self):
        self.mesh=dolfin.Mesh()
        editor=dolfin.MeshEditor()
        editor.open(self.mesh, "tetrahedron", 3, 3)
        editor.init_vertices_global(len(self.geometry.nodetags), len(self.geometry.nodetags))
        self.nodemap, self.nodemap_inv, self.cellmap, self.cellmap_inv = (dict(),
                                                                          dict(),
                                                                          dict(),
                                                                          dict())
        for i, tag in enumerate(self.geometry.nodetags):
            self.nodemap[i]=tag
            self.nodemap_inv[tag]=i
            editor.add_vertex(i, [self.geometry.nodecoords[3*i],
                                    self.geometry.nodecoords[3*i+1],
                                    self.geometry.nodecoords[3*i+2]])
        index_tetra=np.where(self.geometry.elementTypes==4)[0][0]
        editor.init_cells_global(len(self.geometry.elementTags[index_tetra]),
                                 len(self.geometry.elementTags[index_tetra]))
        if self.cell_physical is None:
            self.cell_physical = dict()
            self.cell_physical['No material']=dict()
            for i, tag in enumerate(self.geometry.elementTags[index_tetra]):
                self.cell_physical['No material'][tag]=[self.geometry.elementnodeTags[0][4*i],
                                                   self.geometry.elementnodeTags[0][4*i+1],
                                                   self.geometry.elementnodeTags[0][4*i+2],
                                                   self.geometry.elementnodeTags[0][4*i+3]]
        i=0
        for material in self.cell_physical.keys():
            for tag, nodes in self.cell_physical[material].items():
                self.cellmap[i]=tag
                self.cellmap_inv[tag]=i
                editor.add_cell(i, [self.nodemap_inv[node] for node in nodes])
                i+=1
        editor.close()
    
    def write_vtk(self, filename):
        ncells, size_cells=0, 0
        for i,Type in enumerate(self.geometry.elementTypes):
            ncells+=len(self.geometry.elementTags[i])
            if Type==1:
                size_cells+=3*len(self.geometry.elementTags[i])
            elif Type==2:
                size_cells+=4*len(self.geometry.elementTags[i])
            elif Type==4:
                size_cells+=5*len(self.geometry.elementTags[i])
            elif Type==15:
                size_cells+=2*len(self.geometry.elementTags[i])
        with open(filename,'w') as f:
            f.write('# vtk DataFile Version 2.0\n')
            f.write('Norm of electric field\n')
            f.write('ASCII\n')
            f.write('DATASET UNSTRUCTURED_GRID\n')
            f.write('POINTS {:} double\n'.format(len(self.geometry.nodetags)))
            for i in range(len(self.geometry.nodetags)):
                f.write('{:} {:} {:}\n'.format(float(self.geometry.nodecoords[3*i]),
                                               float(self.geometry.nodecoords[3*i+1]),
                                               float(self.geometry.nodecoords[3*i+2])))
            f.write('\n')
            f.write('CELLS {:} {:}\n'.format(ncells, size_cells))
            for i,Type in enumerate(self.geometry.elementTypes):
                if Type==1:
                    for j in range(int(len(self.geometry.elementnodeTags[i])/2)):
                        f.write('2 {:} {:}\n'.format(int(self.geometry.elementnodeTags[i][j*2]-1),
                                                     int(self.geometry.elementnodeTags[i][j*2+1]-1)))
                elif Type==2:
                    for j in range(int(len(self.geometry.elementnodeTags[i])/3)):
                        f.write('3 {:} {:} {:}\n'.format(int(self.geometry.elementnodeTags[i][j*3]-1),
                                                 int(self.geometry.elementnodeTags[i][j*3+1]-1),
                                                 int(self.geometry.elementnodeTags[i][j*3+2]-1)))
                elif Type==4:
                    for j in range(int(len(self.geometry.elementnodeTags[i])/4)):
                        f.write('4 {:} {:} {:} {:}\n'.format(int(self.geometry.elementnodeTags[i][j*4]-1),
                                                 int(self.geometry.elementnodeTags[i][j*4+1]-1),
                                                 int(self.geometry.elementnodeTags[i][j*4+2]-1),
                                                 int(self.geometry.elementnodeTags[i][j*4+3]-1)))
                elif Type==15:
                    for j in range(len(self.geometry.elementnodeTags[i])):
                        f.write('1 {:}\n'.format(int(self.geometry.elementnodeTags[i][j]-1)))
            f.write('\n')
            f.write('CELL_TYPES {:}\n'.format(ncells))
            for i,Type in enumerate(self.geometry.elementTypes):
                if Type==1:
                    for j in range(len(self.geometry.elementTags[i])):
                        f.write('3\n')
                elif Type==2:
                    for j in range(len(self.geometry.elementTags[i])):
                        f.write('5\n')
                elif Type==15:
                    for j in range(len(self.geometry.elementTags[i])):
                        f.write('1\n')
                elif Type==4:
                    for j in range(len(self.geometry.elementTags[i])):
                        f.write('10\n')
            f.write('\n')

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