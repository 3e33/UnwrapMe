# Copyright (C) 2023-2026 Elias V.
# SPDX-License-Identifier: GPL-3.0-only

import bpy
import bmesh
from bpy_extras import mesh_utils

import numpy as np
import itertools
import functools
import random
import math
from heapq import heappush, heappop, heapify

print("-------------------")

#### A* ####

class Node():
    __slots__ = ('bmVert',
                 'cost',
                 'parentNode',
                 'travelledDistance',
                 'bmEdge')

    def __init__(self,
                 bmVert,
                 cost = 0,
                 parentNode = None,
                 travelledDistance = 0,
                 bmEdge = None):
        self.cost = cost
        self.parentNode = parentNode
        self.bmVert = bmVert
        self.bmEdge = bmEdge
        self.travelledDistance = travelledDistance

    def __lt__(self, other):
        return self.cost < other.cost

def getShortestPath(startVert, endVert, bmEdges, precalcs):
    fringes = [Node(startVert)]
    visited = set()
    pathEdges = []

    while fringes:
        currentNode = heappop(fringes)

        if(currentNode.bmVert == endVert):
            while(currentNode.parentNode):
                pathEdges.append(currentNode.bmEdge)
                currentNode = currentNode.parentNode

            return pathEdges

        edges = [edge for edge in currentNode.bmVert.link_edges
                 if edge in bmEdges]

        if(not edges):
            return None

        edgeLengths = np.array([precalcs.edgeLengths[edge.index] for edge in edges])

        finalDistances = np.array([edge.other_vert(currentNode.bmVert).co
                                   for edge in edges])
        finalDistances = np.array(endVert.co) - finalDistances
        finalDistances = np.sqrt(np.sum(finalDistances * finalDistances, axis = 1))

        costs = currentNode.travelledDistance + edgeLengths + finalDistances
        nodes = [Node(edges[c].other_vert(currentNode.bmVert),
                      costs[c],
                      currentNode,
                      currentNode.travelledDistance + edgeLengths[c],
                      edges[c])
                 for c in range(len(costs))]

        for n in nodes:
            if(n.bmEdge.index not in visited):
                heappush(fringes, n)
                visited.add(n.bmEdge.index)

    return None

############

class Precalcs():
    __slots__ = ("faceAreas",
                 "centroids",
                 "edgeLengths")

    def __init__(self):
        self.faceAreas = []
        self.centroids = []
        self.edgeLengths = []

def initObjectState():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')

def initBlenderState():
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

def getActiveMesh():
    obj = bpy.context.edit_object
    mesh = obj.data

    return mesh

def getBMesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    return bm
    
def opMergeCharts():
    bpy.ops.mesh.select_linked(delimit={'SEAM'})
    bpy.ops.mesh.mark_seam(clear=True)
    bpy.ops.mesh.region_to_loop()
    bpy.ops.mesh.mark_seam(clear=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')

def opSeamFaces():
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.region_to_loop()
    bpy.ops.mesh.mark_seam(clear=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='FACE')

def getSeamWalk(edge, vert, allowedEdges, limitEdges = False):
    if(not edge.seam):
        return None

    edges = set()
    edges.add(edge)

    endVert = vert
    while True:
        nextEdge = [e for e in endVert.link_edges
                    if e not in edges and
                    e.seam == True and
                    (limitEdges and e in allowedEdges)]

        if(not nextEdge or
           len(nextEdge) > 1):
            break

        edges.add(nextEdge[0])
        endVert = nextEdge[0].other_vert(endVert)

    return [edges, endVert]


def opStraightenSeams(mesh, bm):
    selectedEdges = set(e for e in bm.edges
                        if e.select == True)
    seams = set(e for e in selectedEdges
                if e.seam == True)

    seamVerts = set(v for e in seams
                    for v in e.verts)
    for v in seamVerts:
        vs = [e for e in v.link_edges
              if e.seam]

        if(len(vs) > 2):
            selectedEdges = selectedEdges.union(vs)

    endPoints = []
    paths = []
    while(seams):
        edge = next(iter(seams))

        walked, endVert = getSeamWalk(edge, edge.verts[0], selectedEdges, True)
        walked2, endVert2 = getSeamWalk(edge, edge.verts[1], selectedEdges, True)

        path = walked.union(walked2)
        seams = seams - path

        endPoints.append([endVert, endVert2])
        paths.append(path)

    for p in paths:
        if((len(p) + len(set(v for e in p
                             for v in e.verts))) % 2 == 0):
            continue;

        for e in p:
            e.seam = False

    precalcs = Precalcs()
    precalcs.edgeLengths = [e.calc_length() for e in bm.edges]
    for i in range(len(endPoints)):
        if(len(endPoints[i]) < 2):
            continue
        
        path = getShortestPath(endPoints[i][0], endPoints[i][1], selectedEdges, precalcs)

        if(path):
            for e in path:
                e.seam = True

    bpy.ops.object.mode_set(mode='OBJECT')
    bm.to_mesh(mesh)
    bpy.ops.object.mode_set(mode='EDIT')
