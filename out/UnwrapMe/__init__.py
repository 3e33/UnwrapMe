# Copyright (C) 2023-2026 Elias V.
# SPDX-License-Identifier: GPL-3.0-only

import bpy
from bpy_extras import mesh_utils
import os
import sys
from pathlib import Path

from . import updater
from . import core

def getAddonFolderPath():
    return Path(os.path.dirname(os.path.realpath(__file__)))

def showPopup(icon, message):
    def popup(self, context):
        self.layout.label(text=message)
    
    bpy.context.window_manager.popup_menu(popup,
                                          title="UnwrapMe Message",
                                          icon=icon)

class UnwrapMeUpdate(bpy.types.Operator):
    bl_idname = "addon.upwrap_me_update"
    bl_label = "Check for updates"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        import ssl
        import tomllib

        ssl._create_default_https_context = ssl._create_unverified_context

        latestURL = updater.getLatestReleaseURL("3e33", "UnwrapMe")
        if(latestURL == None):
            showPopup("ERROR",
                      "Problem getting addon release URL.")

            return {'CANCELLED'}
            
        addonFolderPath = getAddonFolderPath()
        latestVersion = updater.getTupleFromString(latestURL.split("/")[-1])

        with open(addonFolderPath / "blender_manifest.toml", "rb") as f:
            data = tomllib.load(f)
        localVersion = tuple(map(int, data.get("version").split(".")))

        if(latestVersion > localVersion):
            showPopup("INFO",
                      "There's a new update!")
        else:
            showPopup("INFO",
                      "No new updates found.")

        return {'FINISHED'}

class UnwrapMeOpenUrl(bpy.types.Operator):
    bl_idname = "addon.upwrapme_open_url"
    bl_label = "OpenUrl"
    bl_options = {"INTERNAL"}
    bl_description = """
Will open your web browser."""

    url: bpy.props.StringProperty(name="")

    def execute(self, context):
        import webbrowser
        
        webbrowser.open(self.url)
        
        return {'FINISHED'}

def dPresetsUpdate(self, context):
    props = context.active_object.unwrapMeProps
    if (props.dPresets == "1"):
        props.maxError = 0.01

    if (props.dPresets == "2"):
        props.maxError = 0.1

    if (props.dPresets == "3"):
        props.maxError = 0.2

    if (props.dPresets == "4"):
        props.maxError = 0.3

    if (props.dPresets == "5"):
        props.maxError = 0.4

def presetsUpdate(self, context):
    props = context.active_object.unwrapMeProps
    if (props.presets == "single_flat"):
        props.growFromSelected = True
        props.fillHoles = False
        props.maxError = 0.01
        props.maxIterations = 1
        props.extraCuts = False
        # props.autoMerge = False

    if (props.presets == "single_curved"):
        props.growFromSelected = True
        props.fillHoles = False
        props.maxError = 0.20
        props.maxIterations = 2
        props.extraCuts = True
        # props.autoMerge = False

    if (props.presets == "fill_all"):
        props.growFromSelected = False
        props.startFromSelected = True
        props.fillHoles = True
        props.maxError = 0.10
        props.maxIterations = 15
        props.extraCuts = True
        # props.autoMerge = True

    if (props.presets == "mechanical"):
        props.growFromSelected = True
        props.fillHoles = True
        props.maxError = 0.10
        props.maxIterations = 15
        props.extraCuts = True
        # props.autoMerge = False

    if (props.presets == "extra"):
        props.growFromSelected = True
        props.fillHoles = False
        props.maxError = 100
        props.maxIterations = 5
        props.extraCuts = True
        # props.autoMerge = False

class GrowCharts(bpy.types.Operator):
    bl_idname = "uv.grow_charts"
    bl_label = "Grow Charts"
    bl_options = {'REGISTER', 'UNDO_GROUPED'}
    bl_description = """
Grow UV charts and create seams for UV unwrapping"""

    def execute(self, context):
        if(self.options.is_repeat):
            return {'PASS_THROUGH'}

        from . import fastcore

        core.initBlenderState()
        mesh = core.getActiveMesh()
        bm = core.getBMesh(mesh)

        props = context.active_object.unwrapMeProps
        fastcore.growCharts(bm, props)

        bpy.ops.object.mode_set(mode='OBJECT')
        bm.to_mesh(mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.ui_units_x = 18

        props = context.active_object.unwrapMeProps
        layout.prop(props, 'presets')
        layout.prop(props, 'growFromSelected')
        if(props.growFromSelected == False):
            layout.prop(props, 'chartsNumber')
            layout.prop(props, 'startFromSelected')

        layout.prop(props, 'maxError')
        layout.prop(props, 'dPresets', expand = True)
        layout.prop(props, 'maxIterations')

        layout.prop(props, 'fillHoles')
        layout.prop(props, 'extraCuts')
        # layout.prop(props, 'autoMerge')

        layout.prop(props, 'sharpAngle')


    def invoke(self, context, event):
        props = context.active_object.unwrapMeProps
        props.presets = "0"

        return context.window_manager.invoke_props_dialog(self)

class UnwrapMeProps(bpy.types.PropertyGroup):
    maxError: bpy.props.FloatProperty(name="Developability",
                                      default=0.10,
                                      step=1.0,
                                      description="""
Lower means less distortion but smaller charts""")

    fillHoles: bpy.props.BoolProperty(name="Fill Surrounding",
                                      default = True,
                                      description = """
Create extra charts to fill empty areas around the initial charts""")

    maxIterations: bpy.props.IntProperty(name="Lloyd Iterations Limit",
                                         default=15,
                                         step=1,
                                         description="""
In case of nonconvergence (it can go forever) we limit iterations.""")

    chartsNumber: bpy.props.IntProperty(name="Charts Amount",
                                        default=15,
                                        step=1,
                                        description = """
Usually better to have too many than too few to cover the entire mesh""")

    growFromSelected: bpy.props.BoolProperty(name="Grow From Selected Face Clusters",
                                             default = False,
                                             description = """
Start growing from selected face clusters, otherwise starting faces will be chosen using farthest-first traversal""")

    startFromSelected: bpy.props.BoolProperty(name="Start From Selected Face Cluster",
                                              default = True,
                                              description = """
Charts grow from initially selected face cluster, otherwise starting face will be random""")

    extraCuts: bpy.props.BoolProperty(name="Cut Cylinders",
                                              default = True,
                                              description = """
Cut extra seams for cylindrical surfaces""")

#     autoMerge: bpy.props.BoolProperty(name="Auto Merge",
#                                       default = True,
#                                       description = """
# Automatically attempt to merge similar charts.""")

    presets: bpy.props.EnumProperty(name="Presets",
                                    items = [("0",
                                              "None",
                                              ""),
                                             ("single_flat",
                                              "Single, Flat",
                                              ""),
                                             ("single_curved",
                                              "Single, Curved",
                                              ""),
                                             ("fill_all",
                                              "Fill, Organic",
                                              ""),
                                             ("mechanical",
                                              "Fill, Mechanical, Angle Limit",
                                              ""),
                                             ("extra",
                                              "Extra Cuts Only",
                                              "")],
                                    update=presetsUpdate)

    dPresets: bpy.props.EnumProperty(name=" ",
                                     items = [("1",
                                               "Flat",
                                               ""),
                                              ("2",
                                               "2",
                                               ""),
                                              ("3",
                                               "3",
                                               ""),
                                              ("4",
                                               "4",
                                               ""),
                                              ("5",
                                               "5",
                                               "")],
                                     update=dPresetsUpdate)

    sharpAngle: bpy.props.IntProperty(name="Angle Limit",
                                      default = 90,
                                      soft_min = 0,
                                      soft_max = 90,
                                      description = """
Charts won't grow past this edge angle""")

    # curvature props
    curvatureStrength: bpy.props.FloatProperty(name = "Strength",
                                               default = 0,
                                               precision = 2,
                                               step = 1,
                                               soft_min = 0,
                                               description = """
Selection strength""")

    curvatureSharpAngle: bpy.props.IntProperty(name="Sharp Angle",
                                               default = 25,
                                               soft_min = 0,
                                               soft_max = 90,
                                               description = """
Selection won't grow past this edge angle""")
    

class OpenUnwrapMePie(bpy.types.Operator):
    bl_idname = "uv.unwrap_me_pie"
    bl_label = "Unwrap Me"
    bl_description = """
Open the Unwrap Me pie menu"""

    def execute(self, context):
        bpy.ops.wm.call_menu_pie(name="VIEW3D_MT_unwrapme")

        return {'FINISHED'}

class MergeCharts(bpy.types.Operator):
    bl_idname = "uv.merge_charts"
    bl_label = "Merge Charts"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Merge charts together by removing the seams between them"""

    def execute(self, context):
        core.opMergeCharts()

        return {'FINISHED'}

class StraightenSeams(bpy.types.Operator):
    bl_idname = "uv.straighten_seams"
    bl_label = "Straighten Seams"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Straighten the seam inside the selected area of faces / edges"""

    def execute(self, context):
        core.initBlenderState()
        mesh = core.getActiveMesh()
        bm = core.getBMesh(mesh)

        try:
            core.opStraightenSeams(mesh, bm)
        except:
            pass

        return {'FINISHED'}

class SeamFaces(bpy.types.Operator):
    bl_idname = "uv.seam_faces"
    bl_label = "Seam Faces Boundary"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Seam boundary of selected faces"""

    def execute(self, context):
        core.opSeamFaces()

        return {'FINISHED'}

class RemoveUVOverlaps(bpy.types.Operator):
    bl_idname = "uv.remove_overlaps"
    bl_label = "Remove UV Chart Overlaps"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Will detect simple inner-chart overlaps and separate them into their own UV islands"""

    def execute(self, context):
        if(self.options.is_repeat):
            return {'PASS_THROUGH'}

        from . import fastcore

        core.initBlenderState()
        mesh = core.getActiveMesh()
        bm = core.getBMesh(mesh)

        for f in bm.faces:
            f.select = False

        import time
        tdata = time.perf_counter()
        ttotal = time.perf_counter()
        
        uv_layer = bm.loops.layers.uv.verify()
        uvFaces = fastcore.polyList()
        for f in bm.faces:
            polygon = fastcore.poly()

            for l in f.loops:
                uv = l[uv_layer].uv
                polygon.addPoint(uv[0], uv[1])

            polygon.addPoint(f.loops[0][uv_layer].uv[0],
                             f.loops[0][uv_layer].uv[1])

            uvFaces.append(polygon)

        print(f"uv data transferred in {time.perf_counter() - tdata:0.4f}s")

        bpy.ops.object.mode_set(mode='OBJECT')
        islands = mesh_utils.mesh_linked_uv_islands(mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        faceIndexes = fastcore.getOverlappingClusters(bm,
                                                      uvFaces,
                                                      islands)

        for i in faceIndexes:
            bm.faces[i].select = True

        bpy.ops.object.mode_set(mode='OBJECT')
        bm.to_mesh(mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.select_all(action='SELECT')

        if(bpy.app.version[0] >= 3 and
           bpy.app.version[1] >= 6):
            bpy.ops.uv.pack_islands(margin=0.001, shape_method='CONVEX')
        else:
            bpy.ops.uv.pack_islands(margin=0.001)

        bpy.ops.uv.select_all(action='DESELECT')

        print(f"Total time: {time.perf_counter() - ttotal:0.4f}s")

        return {'FINISHED'}

class SelectByCurvature(bpy.types.Operator):
    bl_idname = "uv.select_by_curvature"
    bl_label = "Select By Curvature"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Select linked faces by curvature"""

    mesh = None
    bm = None
    bridgeContext = None

    def execute(self, context):
        from . import fastcore

        if(not self.options.is_repeat):
            self.bridgeContext = fastcore.getBridgeContext()

            core.initObjectState()
            self.mesh = core.getActiveMesh()
            self.bm = core.getBMesh(self.mesh)

        props = context.active_object.unwrapMeProps
        fastcore.selectByCurvature(self.bm,
                                   self.bridgeContext,
                                   props.curvatureStrength,
                                   props.curvatureSharpAngle)

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bm.to_mesh(self.mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        props = context.active_object.unwrapMeProps
        layout.prop(props, 'curvatureStrength')
        layout.prop(props, 'curvatureSharpAngle')

class UnwrapMeRegionClusteringProps(bpy.types.PropertyGroup):
    mergeStrength: bpy.props.FloatProperty(default = 0,
                                           soft_min = 0,
                                           precision = 2,
                                           step = 1,
                                           description = """
Controls the threshold that decides if two regions should be merged together""")

    mergeAngleLimit: bpy.props.IntProperty(default = 40,
                                           soft_min = 0,
                                           soft_max = 90,
                                           description = """
Regions will not merge over this angle""")

    curvatureWeight: bpy.props.FloatProperty(default = 0,
                                             soft_min = 0,
                                             soft_max = 1,
                                             precision = 2,
                                             step = 1,
                                             description = """
Impact of curvature on the segmentation, from 0 to 1""")

    postprocessing: bpy.props.IntProperty(default = 0,
                                          soft_min = 0,
                                          description = """
Used to fix small gaps.
The number indicates how many times border faces are expanded,
similar to Select More""")

class RegionClustering(bpy.types.Operator):
    bl_idname = "uv.region_clustering"
    bl_label = "Region Clustering"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Segments a mesh by clustering similar regions"""

    mesh = None
    bm = None
    bridgeContext = None

    def execute(self, context):
        from . import fastcore

        if(not self.options.is_repeat):
            self.bridgeContext = fastcore.clustering.getBridgeContext()

            core.initObjectState()
            self.mesh = core.getActiveMesh()
            self.bm = core.getBMesh(self.mesh)

        props = context.active_object.unwrapMeRegionClusteringProps
        result = fastcore.clustering.regionClustering(self, props)

        if (result < 0):
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bm.to_mesh(self.mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

    def draw(self, context):
        props = context.active_object.unwrapMeRegionClusteringProps

        layout = self.layout
        layout.prop(props, 'mergeStrength', text="Strength")
        layout.prop(props, 'mergeAngleLimit', text="Angle Limit")
        layout.prop(props, 'curvatureWeight', text="Curvature Weight")
        layout.prop(props, 'postprocessing', text="Postprocessing Steps")

class UnwrapMeAutoSeamProps(bpy.types.PropertyGroup):
    maxError: bpy.props.FloatProperty(name="Max Error",
                                      default=0.05,
                                      step=1.0,
                                      description="""
The maximum allowed error when fitting the actual 3D surface to a developable proxy.
Lower values produce less distortion when unwrapping but more charts""")

    iterations: bpy.props.IntProperty(name="Relaxation Steps",
                                      default=5,
                                      step=1,
                                      description="""
The maximum number of Lloyd iterations to do between each new chart insertion""")

    looseGeometry: bpy.props.BoolProperty(name="Loose Geometry",
                                          default = False,
                                          description = """
Use if the model has disconnected geometry so that seams are generated for all loose pieces""")

    manualSelect: bpy.props.BoolProperty(name="Manual Selection Mode",
                                         default = False,
                                         update = lambda self, context: (setattr(self, "looseGeometry", False) if self.manualSelect else None),
                                         description = """
Clustering will start from selected faces""")

    clearSeams: bpy.props.BoolProperty(name="Clear Seams",
                                       default = True,
                                       description = """
Clear existing seams""")

class UnwrapMeAutoSeam(bpy.types.Operator):
    bl_idname = "uv.auto_seam"
    bl_label = "Auto Seam"
    bl_options = {'REGISTER', 'UNDO_GROUPED'}
    bl_description = """
Generate UV seams"""

    def execute(self, context):
        if(self.options.is_repeat):
            return {'PASS_THROUGH'}

        from . import fastcore

        core.initObjectState()
        self.mesh = core.getActiveMesh()
        self.bm = core.getBMesh(self.mesh)

        props = context.active_object.unwrapMeAutoSeamProps
        result = fastcore.autoSeam.autoSeam(self, props)

        if (result < 0):
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bm.to_mesh(self.mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.ui_units_x = 18

        props = context.active_object.unwrapMeAutoSeamProps

        layout.prop(props, 'maxError')
        layout.prop(props, 'iterations')
        row = layout.row()
        row.prop(props, 'looseGeometry')
        layout.prop(props, 'manualSelect')
        if (props.manualSelect == True):
            layout.prop(props, 'clearSeams')
            row.enabled = False

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

class UnwrapMeAutoSplit(bpy.types.Operator):
    bl_idname = "uv.auto_split"
    bl_label = "Auto Split"
    bl_options = {'REGISTER', 'UNDO_GROUPED'}
    bl_description = """
Split cylindrical surfaces with extra seams"""

    def execute(self, context):
        if(self.options.is_repeat):
            return {'PASS_THROUGH'}

        from . import fastcore

        core.initObjectState()
        self.mesh = core.getActiveMesh()
        self.bm = core.getBMesh(self.mesh)

        result = fastcore.autoSplit.autoSplit(self)

        if (result < 0):
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bm.to_mesh(self.mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

class UnwrapMeEdgeWalkProps(bpy.types.PropertyGroup):
    searchAngle: bpy.props.IntProperty(default = 45,
                                       soft_min = 0,
                                       soft_max = 90,
                                       description = """
Max search angle for finding next edge""")

class UnwrapMeEdgeWalk(bpy.types.Operator):
    bl_idname = "uv.edge_walk"
    bl_label = "Edge Walk"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """
Walk along an edge loop while creating or erasing a seam"""

    mesh = None
    bm = None
    bridgeContext = None

    def execute(self, context):
        from . import fastcore

        if(not self.options.is_repeat):
            self.bridgeContext = fastcore.edgeWalk.getBridgeContext()

            core.initObjectState()
            self.mesh = core.getActiveMesh()
            self.bm = core.getBMesh(self.mesh)

        props = context.window_manager.unwrapMeEdgeWalkProps
        result = fastcore.edgeWalk.edgeWalk(self, props)

        if (result < 0):
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bm.to_mesh(self.mesh)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

    def draw(self, context):
        props = context.window_manager.unwrapMeEdgeWalkProps

        layout = self.layout
        layout.prop(props, 'searchAngle', text="Max Search Angle")

class VIEW3D_MT_unwrapme(bpy.types.Menu):
    bl_label = "Unwrap Me"

    def draw(self, context):
        layout = self.layout

        pie = layout.menu_pie()

        # left
        col = pie.column()
        col.operator(MergeCharts.bl_idname)
        col.operator(UnwrapMeEdgeWalk.bl_idname)
        col.operator(SeamFaces.bl_idname)
        col.operator(SelectByCurvature.bl_idname)
        col.operator(StraightenSeams.bl_idname)

        # right
        col = pie.column()
        col.operator(GrowCharts.bl_idname)
        col.operator(RemoveUVOverlaps.bl_idname)

        # bottom
        row = pie.row()
        rowCol = row.column()
        rowCol.operator(UnwrapMeAutoSeam.bl_idname)
        rowCol.operator(RegionClustering.bl_idname)
        rowCol.operator(UnwrapMeAutoSplit.bl_idname)

        # top
        # row = pie.row()
        # rowCol = row.column()

class UnwrapMePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    bought: bpy.props.BoolProperty(
        name="I already purchased it.",
        default=False)

    def draw(self, context):
        layout = self.layout

        layout.operator(UnwrapMeUpdate.bl_idname)

        buyBox = layout.box()
        if(not self.bought):
            buyBox.label(text="--- Buy Unwrap Me ---", icon="HEART")

            props = buyBox.operator(UnwrapMeOpenUrl.bl_idname, text="Blender Market", icon="URL")
            props.url = "https://blendermarket.com/creators/shiftatechnologies"

            buyBox.prop(self, "bought", icon="HEART")

        else:
            buyBox.prop(self, "bought", text="Thank you.", icon="HEART")

    
def unwrapmeMenu(self, context):
    layout = self.layout

    layout.separator()
    layout.operator(OpenUnwrapMePie.bl_idname)

classes = [
    UnwrapMePreferences,
    UnwrapMeUpdate,
    UnwrapMeProps,
    GrowCharts,
    OpenUnwrapMePie,
    MergeCharts,
    SeamFaces,
    StraightenSeams,
    VIEW3D_MT_unwrapme,
    RemoveUVOverlaps,
    UnwrapMeOpenUrl,
    SelectByCurvature,
    RegionClustering,
    UnwrapMeRegionClusteringProps,
    UnwrapMeAutoSeamProps,
    UnwrapMeAutoSeam,
    UnwrapMeAutoSplit,
    UnwrapMeEdgeWalkProps,
    UnwrapMeEdgeWalk
]

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Object.unwrapMeProps = bpy.props.PointerProperty(type=UnwrapMeProps)
    bpy.types.Object.unwrapMeRegionClusteringProps = bpy.props.PointerProperty(type=UnwrapMeRegionClusteringProps)
    bpy.types.Object.unwrapMeAutoSeamProps = bpy.props.PointerProperty(type=UnwrapMeAutoSeamProps)
    bpy.types.WindowManager.unwrapMeEdgeWalkProps = bpy.props.PointerProperty(type=UnwrapMeEdgeWalkProps)

    bpy.types.VIEW3D_MT_uv_map.append(unwrapmeMenu)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)

    del bpy.types.Object.unwrapMeProps
    del bpy.types.Object.unwrapMeRegionClusteringProps
    del bpy.types.Object.unwrapMeAutoSeamProps
    del bpy.types.WindowManager.unwrapMeEdgeWalkProps
    bpy.types.VIEW3D_MT_uv_map.remove(unwrapmeMenu)
