#!/usr/bin/env python

import sys
import vtk
import PyQt5
import numpy as np
import os

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QFrame, QCheckBox, QGridLayout, \
    QHBoxLayout, QPushButton, QLabel, QSizePolicy, QSpacerItem, QToolButton, QStyleFactory, \
    QVBoxLayout, QWidget, QButtonGroup, QSlider, QMenuBar, QDockWidget, QMainWindow, QScrollArea, QFileDialog, QAction
from PyQt5.QtOpenGL import QGLFormat
from PyQt5.QtWidgets import QSizePolicy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


from itertools import product
from vtk.util import numpy_support
import vtk

from PyQt5.QtOpenGL import QGLFormat

from qt_utils_rangeslider import *

import warnings
import math

import vtk_utils


from matplotlib import pyplot as plt
from matplotlib.image import imread as imread


from matplotlib.collections import QuadMesh, LineCollection, PatchCollection



def calculate_viewpoint_bounds(v_al, actual_size, renderer):

    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()

    # bounds_indices = np.asarray([[0,0,0],[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    bounds_indices = np.asarray([[0,2,4],[0,2,5], [0,3,4], [0,3,5], [1,2,4], [1,2,5], [1,3,4], [1,3,5]])
    
    viewport_bounds = np.zeros(4)
    viewport_bounds[:2] = np.inf
    viewport_bounds[2:] = -1

    for bound_index in bounds_indices: 
        coordinate.SetValue(v_al[bound_index[0]], v_al[bound_index[1]], v_al[bound_index[2]])
        viewCoord = coordinate.GetComputedViewportValue(renderer)

        if viewCoord[0] < viewport_bounds[0]:
            viewport_bounds[0] = viewCoord[0]
        if viewCoord[1] < viewport_bounds[1]:
            viewport_bounds[1] = viewCoord[1]

        if viewCoord[0] > viewport_bounds[2]:
            viewport_bounds[2] = viewCoord[0]
        if viewCoord[1] > viewport_bounds[3]:
            viewport_bounds[3] = viewCoord[1]



    viewport_bounds /= np.asarray([actual_size[0], actual_size[1], actual_size[0], actual_size[1]])

    return viewport_bounds


def bounds_to_viewpoint(v_al, actual_size, renderer):

    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()

    # bounds_indices = np.asarray([[0,0,0],[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    bounds_indices = np.asarray([[0,2,4],[0,2,5], [0,3,4], [0,3,5], [1,2,4], [1,2,5], [1,3,4], [1,3,5]])
    
    viewpoint_bounds = np.zeros([8, 2])
    for b, bound_index in enumerate(bounds_indices): 
        coordinate.SetValue(v_al[bound_index[0]], v_al[bound_index[1]], v_al[bound_index[2]])
        viewpoint_bounds[b,:] = coordinate.GetComputedViewportValue(renderer)

    viewpoint_bounds /= np.asarray([actual_size[0], actual_size[1]])

    return viewpoint_bounds


def interpolate_grid(grid, subsampling=10):

    grid_origin = grid[0,0,:]

    grid_offsets = (grid - grid_origin)/subsampling

    grid_persp_i = grid_offsets[1,1,:] - grid_offsets[1,0,:]
    grid_persp_j = grid_offsets[1,1,:] - grid_offsets[0,1,:]

    new_grid = np.zeros([subsampling + 1, subsampling + 1, 2])

    for i in range(subsampling + 1):
        for j in range(subsampling + 1):
            new_grid[i,j,:] = grid_origin + (i*grid_offsets[1,0,:]) + (j*grid_offsets[0,1,:]) + (i * grid_persp_i) + (j * grid_persp_j)

    return new_grid


def make_mpl_axes(axes_lims=[[0,1], [0,1], [0,1]], xlabel='X', ylabel='Y', zlabel='Z', savename='mpl.pgf', im=None, figsize=(6,6), cam_params=[30,30,30], edgecolor='k', focal_length=1, viewport_bounds=[0,0,1,1], dots=None):

    aspect = np.asarray([axes_lims[1]-axes_lims[0], axes_lims[3]-axes_lims[2], axes_lims[5]-axes_lims[4]])

    plt.rcParams["figure.autolayout"] = False
    fig = plt.figure(figsize=figsize)
    plt.axis('equal')

    # ax = fig.add_axes(viewport_bounds, projection='3d', box_aspect=aspect)


    # # ax = plt.axes(projection='3d', box_aspect=aspect)
    # # ax = plt.axes(projection='3d')

    # ax.view_init(azim=cam_params[0], elev=cam_params[1], roll=cam_params[2])
    # ax.set_proj_type('persp', focal_length=focal_length)

    # print('viewport bounds', viewport_bounds)

    # # ax.set_position(viewport_bounds, which='actual')
    # ax.patch.set_linewidth(2)
    # ax.patch.set_edgecolor('cornflowerblue')
    # # ax.set_xlabel(xlabel)
    # # ax.set_ylabel(ylabel)
    # # ax.set_zlabel(zlabel)
    # ax.set_xlim(axes_lims[0], axes_lims[1])
    # ax.set_ylim(axes_lims[2], axes_lims[3])
    # ax.set_zlim(axes_lims[4], axes_lims[5])

    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    # ax.xaxis.pane.set_edgecolor(edgecolor)
    # ax.yaxis.pane.set_edgecolor(edgecolor)
    # ax.zaxis.pane.set_edgecolor(edgecolor)

    # plt.tick_params(top=False, bottom=False, left=False, right=False,
    #             labelleft=False, labelbottom=False)
    
    
    ax2 = fig.add_axes([0, 0, 1, 1])
    ax2.patch.set_alpha(0.00)
    ax2.set_axis_off()
    
    if dots is not None:

        xp = np.zeros((2,2,2))
        xp[0,0,:] = dots[0,:]
        xp[0,1,:] = dots[1,:]
        xp[1,0,:] = dots[2,:]
        xp[1,1,:] = dots[3,:]

        xp = interpolate_grid(xp)

        x_plane = QuadMesh(xp, edgecolors='k', facecolors='w')

        yp = np.zeros((2,2,2))
        yp[0,0,:] = dots[0,:]
        yp[0,1,:] = dots[4,:]
        yp[1,0,:] = dots[2,:]
        yp[1,1,:] = dots[6,:]
        yp = interpolate_grid(yp)
        y_plane = QuadMesh(yp, edgecolors='k', facecolors='w')
       
        zp = np.zeros((2,2,2))
        zp[0,0,:] = dots[0,:]
        zp[0,1,:] = dots[1,:]
        zp[1,0,:] = dots[4,:]
        zp[1,1,:] = dots[5,:]
        zp = interpolate_grid(zp)
        z_plane = QuadMesh(zp, edgecolors='k', facecolors='w')

        xp2 = np.zeros((2,2,2))
        xp2[0,0,:] = dots[4,:]
        xp2[0,1,:] = dots[5,:]
        xp2[1,0,:] = dots[6,:]
        xp2[1,1,:] = dots[7,:]
        x_plane_2 = QuadMesh(xp2, edgecolors='k', facecolors='w')

        yp2 = np.zeros((2,2,2))
        yp2[0,0,:] = dots[1,:]
        yp2[0,1,:] = dots[5,:]
        yp2[1,0,:] = dots[3,:]
        yp2[1,1,:] = dots[7,:]
        y_plane_2 = QuadMesh(yp2, edgecolors='k', facecolors='w')

        zp2 = np.zeros((2,2,2))
        zp2[0,0,:] = dots[2,:]
        zp2[0,1,:] = dots[3,:]
        zp2[1,0,:] = dots[6,:]
        zp2[1,1,:] = dots[7,:] 
        z_plane_2 = QuadMesh(zp2, edgecolors='k', facecolors='w')


        ax2.add_collection(x_plane)
        ax2.add_collection(y_plane)
        ax2.add_collection(z_plane)
        # for dot in dots:
        #     rect = Circle(dot, radius=0.005, color="blue", zorder=10)
        #     ax2.add_patch(rect)



    if im is not None:
        imax = fig.add_axes([0, 0, 1, 1])
        # imax = fig.add_axes([0.05, 0.05, 0.9, 0.9])


        imax.set_axis_off()
        imax.imshow(im, aspect="equal")

        # rect = Rectangle((0.5, 0.5), width=0.05, height=0.05, color="red", zorder=10)
        # imax.add_patch(rect)

        # imax.patch.set_linewidth(2)
        # imax.patch.set_edgecolor('green')

    # actual_fucking_position = ax.get_position()

    # print('fucking position', actual_fucking_position)
    # ax.set_position(viewport_bounds, which='actual')
    # actual_fucking_position = ax.get_position()


    # print('fucking position', actual_fucking_position)
    # plt.tight_layout()
    plt.savefig(savename, dpi=600)
    plt.show()

def export_geometry(source, savename=None, filetype='gltf', verbose=True):

    if filetype == 'vtkjs':

        savename = savename[:-6]

        if verbose:

            print('writing VTKJS to %s' % savename)
        # warnings.warn('This export function is broken for JS files, I\'m not sure what\'s wrong')

        # pass

        exporter = vtk.vtkJSONRenderWindowExporter()

        exporter.SetDebug(True)
        exporter.GlobalWarningDisplayOn()

        exporter.GetArchiver().SetArchiveName(savename)
        # exporter.GetArchiver().SetArchiveName('/home/fraser/Videos/LiverView/dodec_export3.js')
        exporter.SetRenderWindow(source)

        exporter.Write()
        exporter.Update()

        # scene_name = os.path.split(savename)[1]
        # savename = savename + ".vtkjs"
        # try:
        #     import zlib
        #     import zipfile
        #     compression = zipfile.ZIP_DEFLATED
        # except:
        #     compression = zipfile.ZIP_STORED
        # zf = zipfile.ZipFile(savename, mode='w')
        # for dir_name, _, file_list in os.walk(savename):
        #     for fname in file_list:
        #         full_path = os.path.join(dir_name, fname)
        #         rel_path = '%s/%s' % (scene_name,
        #                                 os.path.relpath(full_path, savename))
        #         zf.write(full_path, arcname=rel_path, compress_type=compression)
        # zf.close()

        if verbose:
            print('file written')

    if filetype == 'js':
        if verbose:
            print('writing JS to %s' % savename)
        # warnings.warn('This export function is broken for JS files, I\'m not sure what\'s wrong')

        # pass

        exporter = vtk.vtkJSONRenderWindowExporter()

        exporter.SetDebug(True)
        exporter.GlobalWarningDisplayOn()

        exporter.GetArchiver().SetArchiveName(savename)
        # exporter.GetArchiver().SetArchiveName('/home/fraser/Videos/LiverView/dodec_export3.js')
        exporter.SetRenderWindow(source)

        exporter.Write()
        exporter.Update()

        if verbose:
            print('file written')

        # exporter = vtk.vtkJSONRenderWindowExporter()
        # partitioned_archiver = vtk.vtkPythonArchiver()
        # partitioned_archiver.SetArchiveName(savename)
        # # partitioned_archiver.OpenArchive()
        # exporter.SetArchiver(partitioned_archiver)
        # exporter.SetRenderWindow(source)
        # print('JS', dir(exporter), exporter)
        # # exporter.SetFileName('%s.jsre' % savename)
        # exporter.Write()
        # exporter.Update()
        # # partitioned_archiver.CloseArchive()
        # if verbose:
        #     print('file written')

    if filetype == 'wrl':
        if verbose:
            print('writing WRL to %s' % savename)
        exporter = vtk.vtkVRMLExporter()
        exporter.SetRenderWindow(source)
        exporter.SetFileName('%s.wrl' % savename)
        exporter.Write()
        exporter.Update()
        if verbose:
            print('file written')

    elif filetype == 'obj':
        if verbose:
            print('writing OBJ to %s' % savename)
        exporter = vtk.vtkOBJExporter()
        exporter.SetRenderWindow(source)
        exporter.SetFilePrefix(savename)
        exporter.Write()
        exporter.Update()
        if verbose:
            print('file written')

    elif filetype == 'pdf':
        if verbose:
            print('writing PDF to %s' % savename)
        exporter = vtk.vtkPDFExporter()
        exporter.SetRenderWindow(source)
        exporter.SetFileName(savename)
        exporter.Update()
        exporter.Write()

        print(exporter)
        print(dir(exporter))

        if verbose:
            print('file written')

    elif filetype == 'svg':
        if verbose:
            print('writing SVG to %s' % savename)
        exporter = vtk.vtkSVGExporter()
        exporter.SetRenderWindow(source)
        exporter.SetFileName(savename)
        exporter.Write()
        exporter.Update()
        if verbose:
            print('file written')

    elif filetype == 'gl2ps':
        if verbose:
            print('writing GL2PS to %s' % savename)
        exporter = vtk.vtkGL2PSExporter()
        exporter.SetRenderWindow(source)
        exporter.SetFilePrefix(savename)
        exporter.Write()
        exporter.Update()
        if verbose:
            print('file written')

    elif filetype == 'x3d':
        if verbose:
            print('writing x3D to %s' % savename)
        exporter = vtk.vtkX3DExporter()
        exporter.SetInput(source)
        exporter.SetFileName(savename)
        exporter.Update()
        exporter.Write()
        if verbose:
            print('file written')

    elif filetype == 'pov':
        if verbose:
            print('writing POV to %s' % savename)
        exporter = vtk.vtkPOVExporter()
        exporter.SetInput(source)
        exporter.SetFileName(savename)
        exporter.Update()
        exporter.Write()
        if verbose:
            print('file written')

    elif filetype == 'pvwgl':
        if verbose:
            print('writing PVWGL to %s' % savename)
        exporter = vtk.vtkPVWebGLExporter()
        exporter.SetInput(source)
        exporter.SetFileName(savename)
        exporter.Update()
        exporter.Write()
        if verbose:
            print('file written')

    elif filetype == 'gltf':
        if verbose:
            print('writing GLTF to %s' % savename)
        try:
            exporter = vtk.vtkGLTFExporter()
        except AttributeError:
            print('Gltf exporting is not supported in your version of VTK, try updating')
        exporter.SetInput(source)
        exporter.InlineDataOn()
        exporter.SetFileName(savename)
        exporter.Update()
        exporter.Write()
        if verbose:
            print('file written')

    elif filetype == 'png':
        render_scale = 4
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetScale(render_scale)
        w2if.SetInput(source)
        w2if.Update()
        if verbose:
            print('writing data to %s, upscaled by %d' % (savename, render_scale))
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(savename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
        if verbose:
            print('file written')
        source.Render()

    elif filetype == 'pgf':

        tempname = 'temp_overlay.png'
        temp_image_size = 1000

        renderer = source.GetRenderers().GetFirstRenderer()
        renderer.SetBackgroundAlpha(0.0)
        renderer.GradientBackgroundOff()
        renderer.SetUseDepthPeeling(1)
        renderer.SetOcclusionRatio(0)
        renderer.Modified()
        source.SetAlphaBitPlanes(1) 
        source.Modified()

        dpi = source.GetDPI()
        actual_size = source.GetActualSize()

        print(actual_size)
        aspect_ratio = actual_size[0]/actual_size[1]

        render_scale = int(np.ceil(temp_image_size/actual_size[0]))


        physical_size = np.asarray(actual_size)/dpi

        total_3D_actors = renderer.GetActors().GetNumberOfItems()

        actors = renderer.GetActors()

        for i in range(total_3D_actors):
            actor = actors.GetItemAsObject(i)

            if isinstance(actor, vtk.vtkCubeAxesActor):

                actor.SetVisibility(False)


        if verbose:
            print('writing PGF to %s' % savename)
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInputBufferTypeToRGBA()
        w2if.SetScale(render_scale)
        w2if.SetInput(source)
        w2if.ReadFrontBufferOff()
        w2if.Update()
        if verbose:
            print('writing data to %s, upscaled by %d' % (savename, render_scale))
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(tempname)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

        source.Render()


        im = imread(tempname)


        for i in range(total_3D_actors):

                actor = renderer.GetActors().GetItemAsObject(i)

                if isinstance(actor, vtk.vtkCubeAxesActor):

                    # xtitle = actor.GetXTitle()
                    # ytitle = actor.GetYTitle()
                    # ztitle = actor.GetZTitle() 


                    # Elements:
                    # 4.44795 0 0 0 
                    # 0 3.73205 0 0 
                    # 0 0 -2.72046 -85.8674 
                    # 0 0 -1 0 
        
                    # 

                    print(actor, dir(actor))

                    


                    xtitle = actor.GetZTitle()
                    ytitle = actor.GetXTitle()
                    ztitle = actor.GetYTitle()

                    v_al = np.asarray(actor.GetBounds())

                    mpl_axes_limits = np.asarray([v_al[4], v_al[5], v_al[0], v_al[1], v_al[2], v_al[3]])

                    axes_camera = actor.GetCamera()
                    axes_camera.OrthogonalizeViewUp()

                    parallel_scale = axes_camera.GetParallelScale()/10

                    print('parallel scale', parallel_scale)

                    clipping_range = axes_camera.GetClippingRange()

                    viewport_bounds = calculate_viewpoint_bounds(v_al, actual_size, renderer)

                    # print(viewport_bounds)

                    mpl_axes_viewpoint_pos = np.asarray([viewport_bounds[0], viewport_bounds[1], viewport_bounds[2]-viewport_bounds[0], viewport_bounds[3]-viewport_bounds[1]])

                    # transform_matrix = axes_camera.GetCompositeProjectionTransformMatrix(aspect_ratio, clipping_range[0], clipping_range[1])

                    # print(transform_matrix.GetData(), dir(transform_matrix))

                    direction_of_projection = axes_camera.GetViewPlaneNormal()
                    azimuth = direction_of_projection[0] * (180/np.pi)
                    elevation = direction_of_projection[1] * (180/np.pi)
                    roll = axes_camera.GetRoll()

                    camera_params = [azimuth, elevation, roll]

                    # angle = 2*atan((h/2)/d)  #where h is the height of the RenderWindow (measured by holding a ruler up to your screen) and d is the distance from your eyes to the screen. 

                    actor.SetVisibility(True)
                    
                    source.Render()

                    dots = bounds_to_viewpoint(v_al, actual_size, renderer)

                    make_mpl_axes(axes_lims=mpl_axes_limits, xlabel=xtitle, ylabel=ytitle, zlabel=ztitle, im=im, figsize=physical_size, savename=savename, cam_params=camera_params, focal_length=parallel_scale, viewport_bounds=mpl_axes_viewpoint_pos, dots=dots)




 
        if verbose:
            print('file written')
    
    else:
        print('Filetype (%s) not supported in list of exporters' % savename)


def selectionCallback(caller, event):

    sel = caller.GetCurrentSelection()
    node0 = sel.GetNode(0)
    node0_field_type = node0.GetFieldType()
    sel_list0 = caller.GetCurrentSelection().GetNode(0).GetSelectionList()
    node1 = sel.GetNode(1)
    node1_field_type = node1.GetFieldType()
    sel_list1 = caller.GetCurrentSelection().GetNode(1).GetSelectionList()

    # print(sel_list0, sel_list1)

    if (sel_list0.GetNumberOfTuples() > 0):
        # printFieldType(node0_field_type)
        for ii in range(sel_list0.GetNumberOfTuples()):
            print(sel_list0.GetValue(ii))

    if (sel_list1.GetNumberOfTuples() > 0):
        # printFieldType(node1_field_type)
        for ii in range(sel_list1.GetNumberOfTuples()):
            print(sel_list1.GetValue(ii))


# class MyInteractorStyle(vtk.vtkInteractorStyleRubberBand3D):
class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """[VTK class definitions to allow for more natural camera movement (trackpad style)
        and keybindings for extra functionality (camera movement, printscreens etc.)]

    Arguments:
      expects: renderWindowInteractor, render_camera, renderWindow
    """

    def __init__(self, parent, camera, renderer):
        self.parent = parent
        self.camera = camera
        self.renderer = renderer

        self.verbose = False
        self.auto_up = True
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        self.AddObserver('AnnotationChangedEvent', selectionCallback)
        # self.AutoAdjustCameraClippingRange(True)

    def camera_zoom_in(self, step=2):

        old = np.asarray(self.camera.GetPosition())

        new = old *0.9

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        # print(dir(self.renderer))

        # self.renderer.ResetCameraClippingRange()

        self.renderer.Render()
        return


    def camera_zoom_out(self, step=2):

        old = np.asarray(self.camera.GetPosition())

        new = old * 1.1

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        # self.renderer.ResetCameraClippingRange()

        self.renderer.Render()
        return



    def rotate_clockwise(self, step=2):

        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        if x_coord >= 0:
            if y_coord >= 0:
                x_sign = -1
                y_sign = 1
            else:
                x_sign = 1
                y_sign = 1
        elif y_coord >= 0:
            x_sign = -1
            y_sign = -1
        else:
            x_sign = 1
            y_sign = -1

        scale = 0.05

        del_x = np.abs(y_coord) * scale * x_sign
        del_y = np.abs(x_coord) * scale * y_sign
        hypot_1 = math.hypot(old[0], old[1])

        new_x =  old[0] + del_x
        new_y =  old[1] + del_y

        hypot_2 = math.hypot(new_x, new_y)

        rescale = hypot_1/hypot_2

        new = old
        new[0] = (new_x * rescale)
        new[1] = (new_y * rescale)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_anticlockwise(self):


        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        if x_coord > 0:
            if y_coord > 0:
                x_sign = 1
                y_sign = -1
            else:
                x_sign = -1
                y_sign = -1
        elif y_coord > 0:
            x_sign = 1
            y_sign = 1
        else:
            x_sign = -1
            y_sign = 1

        scale = 0.05

        del_x = np.abs(y_coord) * scale * x_sign
        del_y = np.abs(x_coord) * scale * y_sign
        hypot_1 = math.hypot(old[0], old[1])

        new_x =  old[0] + del_x
        new_y =  old[1] + del_y

        hypot_2 = math.hypot(new_x, new_y)

        rescale = hypot_1/hypot_2

        new = old
        new[0] = (new_x * rescale)
        new[1] = (new_y * rescale)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_upclockwise(self):


        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        # print(x_coord, y_coord)

        d_coord = math.hypot(x_coord, y_coord)

        z_coord = old[2]

        if z_coord >= 0:
           z_sign = 1
           d_sign = -1
        else:
            z_sign = 1
            d_sign = 1

        scale = 0.05

        del_z = np.abs(d_coord) * scale * z_sign
        del_d = np.abs(z_coord) * scale * d_sign
        hypot_1 = math.hypot(d_coord, z_coord)

        new_z =  z_coord + del_z
        new_d =  d_coord + del_d

        hypot_2 = math.hypot(new_z, new_d)
        rescale = hypot_1/hypot_2

        new = old
        new[2] = (new_z * rescale)
        new_d = (new_d * rescale)

        rescale_2 = new_d/d_coord

        new[0] = (x_coord * rescale_2)
        new[1] = (y_coord * rescale_2)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_downclockwise(self):

        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        d_coord = math.hypot(x_coord, y_coord)

        z_coord = old[2]

        if z_coord >= 0:
           z_sign = -1
           d_sign = +1
        else:
            z_sign = -1
            d_sign = -1

        scale = 0.05

        del_z = np.abs(d_coord) * scale * z_sign
        del_d = np.abs(z_coord) * scale * d_sign
        hypot_1 = math.hypot(d_coord, z_coord)

        new_z =  z_coord + del_z
        new_d =  d_coord + del_d

        hypot_2 = math.hypot(new_z, new_d)
        rescale = hypot_1/hypot_2

        new = old
        new[2] = (new_z * rescale)
        new_d = (new_d * rescale)

        rescale_2 = new_d/d_coord

        new[0] = (x_coord * rescale_2)
        new[1] = (y_coord * rescale_2)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def screenshot(self):

        w2if = vtk.vtkWindowToImageFilter()
        # print(dir(w2if))
        w2if.SetScale(4)
        w2if.SetInput(self.renderer)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("screenshot.png")
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

    def keyPressEvent(self, obj, event):
        key = str(self.parent.GetKeySym())
        # check here for available keypresses supported by Qt & VTK
        # https://github.com/Kitware/VTK/blob/master/GUISupport/Qt/QVTKInteractorAdapter.cxx
        if key == 'Left':
            self.rotate_clockwise()
        if key == 'Right':
            self.rotate_anticlockwise()
        if key == 'Up':
            self.rotate_upclockwise()
        if key == 'Down':
            self.rotate_downclockwise()
        if key == 'c':
            self.screenshot()
        if key == 'equal':
            self.camera_zoom_in()
        if key == 'minus':
            self.camera_zoom_out()
        if key == 'o':
            self.switch_rendering_mode()
        return


    def left_button_press_event(self, obj, event):
        if self.verbose:
            print("left Button pressed")
        self.OnLeftButtonDown()
        return


    def left_button_release_event(self, obj, event):
        if self.verbose:
            print("left Button released")
        self.OnLeftButtonUp()
        return


    def middle_button_press_event(self, obj, event):
        if self.verbose:
            print("Middle Button pressed")
        self.OnMiddleButtonDown()
        return


    def middle_button_release_event(self, obj, event):
        if self.verbose:
            print("Middle Button released")
        self.OnMiddleButtonUp()
        return

    def switch_rendering_mode(self):
        # print(dir(self.camera))
        projection_mode = self.camera.GetParallelProjection()
        if projection_mode == 1:
            self.camera.SetParallelProjection(False)
            projection_mode = self.camera.GetParallelProjection()
            # print(projection_mode)
            print('Camera set to Perspective Projection')
        else:
            self.camera.SetParallelProjection(True)
            projection_mode = self.camera.GetParallelProjection()
            # print(projection_mode)
            print('Camera set to Parallel Projection')

        self.renderer.Render()


class Widget(QMainWindow):
    def __init__(self,
                 locations=None,
                 data=None,
                 actor_dict=None,
                 edge_data=None,
                 data_type='basis',
                 glyph_type='cube',
                 glyph_scale=1,
                 indicator_type='crystal',
                 cubemap=None,
                 parent=None,
                 slicing_vals=None,
                 title=None,
                 caption_dict=None,
                 show_camera_controls=False,
                 use_SSAO=False):
        super(Widget, self).__init__(parent=parent)

        f = QGLFormat()
        # print(dir(f))
        print(f)
        f.setSampleBuffers(True)  # turn on antialiasing
        f.setAlpha(True)
        QGLFormat.setDefaultFormat(f)

        self.range_dict = dict()
        self.title_text = title
        self.data_backup = data
        self.data_type = data_type
        self.indices = None
        self.indicator_type = indicator_type
        self.caption_dict = caption_dict
        self.show_camera_controls = show_camera_controls
        self.use_SSAO = use_SSAO
        try:
            self.indices = np.arange(0, locations.shape[0])
        except:
            pass
        self.locations_backup = locations
        self.data = data
        self.edge_data = edge_data
        self.slicing_vals = slicing_vals
        self.cubemap_path = cubemap

        # actor_dict_1D = dict()

        if actor_dict is not None:

            # print('List of actors supplied')

            # for keyname in actor_dict.keys():
            #     try:
            #         for vnum, value in enumerate(actor_dict[keyname]):
            #             # actor_list_1D.append(value)
            #             new_key = "%s-%d" % (keyname, vnum)
            #             actor_dict_1D.update({new_key: actor_dict[keyname][vnum]})

            #     except:
            #         # actor_list_1D.append(actor_dict[keyname])
            #         actor_dict_1D.update({keyname: actor_dict[keyname]})

            # # self.actor_list = actor_list_1D

            # actor_dict = None
            # self.actor_dict = actor_dict_1D
            # self.actor_visibility = np.ones(len(self.actor_dict.keys()))

            # print('List of actors supplied')

            # for keyname in actor_dict.keys():
            #     try:
            #         print(type(actor_dict[keyname]))
            #         for vnum, value in enumerate(actor_dict[keyname]):
            #             # actor_list_1D.append(value)
            #             new_key = "%s-%d" % (keyname, vnum)
            #             actor_dict_1D.update({new_key: actor_dict[keyname][vnum]})

            #     except:
            #         # actor_list_1D.append(actor_dict[keyname])
            #         actor_dict_1D.update({keyname: actor_dict[keyname]})

            # self.actor_list = actor_list_1D

            # actor_dict = None
            self.actor_dict = actor_dict
            # self.actor_visibility = np.ones(len(self.actor_dict.keys()))

        else:
            self.actor_dict = None
            self.actor_list = None

        self.glyph_type = glyph_type
        self.glyph_scale = glyph_scale

        if locations is not None:
            self.locations = locations
            self.make_location_dict()
            self.scalar_size = None
            self.scalar_property = list(data.keys())[0]
            self.scalar_bounds = [np.min(list(data.values())[0]), np.max(list(data.values())[0])]

        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(QApplication.palette())

        self.layoutUI()
        self.setWindowTitle('LiverView Visualisation Window')

    def make_location_dict(self):

        try:
            self.data.update({'X': self.locations[:, 0]})
            self.data.update({'Y': self.locations[:, 1]})
            self.data.update({'Z': self.locations[:, 2]})
        except:
            pass

    def set_camera(self, btn):
        vtk_utils.set_camera(self.camera, btn.text())
        self.camera.Modified()
        self.iren.Render()

    def camera_reset(self):
        self.ren.ResetCamera()
        self.iren.Render()

    def camera_auto_reset_func(self, state):

        if state == QtCore.Qt.Checked:
            print('Autoreset Enabled')
            self.camera_auto_reset = True
        else:
            print('Autoreset Disabled')
            self.camera_auto_reset = False

    def camera_perspective_func(self, state):

        if state == QtCore.Qt.Checked:
            print('Using Perspective Rendering')
            self.camera_perspective = True
        else:
            print('Using Orthographic Rendering')
            self.camera_perspective = False

        self.rerender()

    def SSAO_func(self, state):

        if state == QtCore.Qt.Checked:
            print('SSAO Enabled')
            self.use_SSAO = True

        else:
            print('SSAO Disabled')
            self.use_SSAO = False

        self.rerender()

    def SSAO_blur_func(self, state):

        if state == QtCore.Qt.Checked:
            print('SSAO Blur Enabled')
            self.SSAO_blur = True

        else:
            print('SSAO Blur Disabled')
            self.SSAO_blur = False

        self.rerender()

    def SSAO_valuechange(self, value, verbose=False):

        if verbose:
            print('value:', value)

        self.SSAO_radius_override = True

        if np.abs(value - self.SSAO_radius) > 10:

            self.SSAO_radius = value

            self.rerender()

        self.SSAO_radius_override = False

    def SSAO_kernel_valuechange(self, value, verbose=False):

        if verbose:
            print('value:', value)

        if np.abs(value - self.SSAO_kernel_size) > 10:

            self.SSAO_kernel_size = value

            self.rerender()

    def tree_update_checkbox_state(self, state):

        if state == QtCore.Qt.Checked:
            print('Auto-update Enabled')
            self.autoupdate = True

        else:
            print('Auto-update Disabled')
            self.autoupdate = False

    def tree_updater(self):
        if self.autoupdate:
            self.rerender()
        else:
            pass

    def valuechange(self, value, images, direction, verbose=False):

        slicer = images  # self.actor_dict[key_name]
        extents = np.asarray(slicer.GetDisplayExtent())

        if direction == 'x':
            if verbose:
                print('x slice set to', value)
            extents[0] = value
            extents[1] = value
            slicer.SetDisplayExtent(extents)
            self.slider_render()

        if direction == 'y':
            if verbose:
                print('y slice set to', value)
            extents[2] = value
            extents[3] = value
            slicer.SetDisplayExtent(extents)
            self.slider_render()

        if direction == 'z':
            if verbose:
                print('z slice set to', value)
            extents[4] = value
            extents[5] = value
            slicer.SetDisplayExtent(extents)
            self.slider_render()

    def update_scalar(self, text, poly_actor, poly_name, verbose=True):

        # self.scalar_property = text

        # if self.use_subset:
        #     scalar_vals = np.nan_to_num(self.data[text])[self.indices]
        # else:
        #     scalar_vals = np.copy(scalar_dict[text])

            # np.nan_to_num(self.data[self.scalar_property])

        # print(scalar_vals)

        # scalar_color = vtk.vtkDoubleArray()
        # scalar_color.SetNumberOfComponents(1)
        # scalar_color.SetName(text)
        # scalar_data = poly_actor.GetMapper().GetInput().GetPointData()

        # print(scalar_data)

        # data_copy = vtk.vtkPointData()

        # data_copy.DeepCopy(scalar_data)

        # scalar_color = data_copy.GetScalars(text)

        # scalar_color = poly_actor.GetMapper().GetInput().GetPointData().GetScalars(text)
        self.polydata_dict[poly_name]['Current_scalar'] = text

        indices = np.copy(self.polydata_dict[poly_name]['Range_indices'])

        scalar_vals = np.copy(self.polydata_dict[poly_name]['scalar_values'][self.polydata_dict[poly_name]['Current_scalar']])

        try:
            scalar_vals = scalar_vals[indices]
        except IndexError:
            print('ERROR INDICES:', indices)

        locations = np.copy(self.polydata_dict[poly_name]['locations'])

        if verbose:
            print('scalar set to:', text)
            print('scalars', scalar_vals)
            print('shape;', scalar_vals.shape)
            print('locations:', locations.shape)

        # scalar_color_save = vtk.vtkFloatArray()

        # scalar_color_save.DeepCopy(scalar_color)

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(locations))

        # print(scalar_color)
        scalar_color = vtk.vtkFloatArray()

        scalar_color.SetArray(numpy_support.numpy_to_vtk(scalar_vals), scalar_vals.shape[0], 1)

        # scalar_vals = numpy_support.vtk_to_numpy(scalar_color)

        # scalar_size = poly_actor.GetMapper().GetInput().GetPointData().GetScalars(text).GetSize()
        # print('+++++++++++++++++++++++++++++++++', scalar_size)
        # scalars_per_glyph = scalar_size/scalar_vals.shape[0]

        # scalar_vals = np.repeat(scalar_vals, scalars_per_glyph)

        # for pointId in range(len(scalar_vals)*8):
        #     try:
        #         scalar_color.InsertNextValue(scalar_vals[pointId])
        #     except IndexError:
        #         continue
        poly_actor.GetMapper().GetInput().SetPoints(points)
        poly_actor.GetMapper().GetInput().GetPointData().SetScalars(scalar_color)
        poly_actor.GetMapper().GetInput().GetPointData().Modified()
        poly_actor.GetMapper().SetScalarRange(np.nanmin(scalar_vals), np.nanmax(scalar_vals))
        # self.colorbar.GetScalarBarActor().SetTitle(self.scalar_property)
        # self.colorbar.GetScalarBarActor().SetLookupTable(self.poly_actor.GetMapper().GetLookupTable())
        scalar_bounds = [np.nanmin(scalar_vals), np.nanmax(scalar_vals)]

        # self.range_dict[poly_name][0] = text
        # self.range_dict[poly_name][1] = scalar_bounds

        # print(poly_actor.GetMapper().GetInput())

        self.polydata_dict[poly_name]['range_slider'].setMin(scalar_bounds[0])
        self.polydata_dict[poly_name]['range_slider'].setMax(scalar_bounds[1])
        self.polydata_dict[poly_name]['range_slider'].setRange(scalar_bounds[0], scalar_bounds[1])
        self.polydata_dict[poly_name]['range_slider'].update()

        self.iren.Render()

    def reset_range(self, click, poly_actor, poly_name, verbose=True):

        if verbose:
            print('in reset mode')

        locations = np.copy(self.polydata_dict[poly_name]['locations_backup']).astype(np.float32)
        locations = np.mean(locations.reshape(-1, 24, 3), axis=1)
        # self.polydata_dict[poly_name]['locations'] = locations

        self.polydata_dict[poly_name]['Range_indices'] = np.ones(self.polydata_dict[poly_name]['Range_indices'].shape[0])

        # data = self.data_backup
        if verbose:
            print('Current scalar:', self.polydata_dict[poly_name]['Current_scalar'])

        data = self.polydata_dict[poly_name]['scalar_values'][self.polydata_dict[poly_name]['Current_scalar']]
        data = np.mean(data.reshape(-1, 24, 1), axis=1)

        if verbose:
            print('data shape:', data.shape)

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(locations))

        # for location in locations:
        #     points.InsertNextPoint(location[0], location[1], location[2])

        # scalar_color = vtk.vtkFloatArray()
        # scalar_color.SetNumberOfComponents(1)
        # scalar_color.SetName("SignedDistances")

        scalar_color = vtk.vtkFloatArray()
        scalar_color.SetArray(numpy_support.numpy_to_vtk(data), data.shape[0], 0)

        if verbose:
            print('check one')
        # for pointId in range(points.GetNumberOfPoints()):
        #     try:
        #         scalar_color.InsertNextValue(data[pointId])
        #     except:
        #         scalar_color.InsertNextValue(0)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        polydata.GetPointData().SetScalars(scalar_color)

        glyph_type = self.glyph_type
        glyph_scale = self.glyph_scale
        fixed_voxel_size = True

        if glyph_type == 'cube':
            glyphSource = vtk.vtkCubeSource()
        elif glyph_type == 'sphere':
            glyphSource = vtk.vtkSphereSource()

        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
        glyph3D.SetInputData(polydata)
        glyph3D.SetScaleFactor(glyph_scale)
        if fixed_voxel_size:
            glyph3D.SetInputArrayToProcess(0, 0, 0, 0, 'RTData')
        glyph3D.Update()

        if verbose:
            print('check two')

        # self.range_dict[poly_name][1] = [np.min(data), np.max(data)]

        scalar_bounds = [np.min(data), np.max(data)]
        self.polydata_dict[poly_name]['range_slider'].setMin(scalar_bounds[0])
        self.polydata_dict[poly_name]['range_slider'].setMax(scalar_bounds[1])
        self.polydata_dict[poly_name]['range_slider'].setRange(scalar_bounds[0], scalar_bounds[1])
        self.polydata_dict[poly_name]['range_slider'].update()

        if verbose:
            print('check three')

        poly_actor.GetMapper().SetInputConnection(glyph3D.GetOutputPort())

        if verbose:
            print('check four')

        poly_actor.GetMapper().Modified()
        self.ren.Render()

        if verbose:
            print('reset complete')

    def get_points_in_range(self, click, poly_actor, poly_name, verbose=True):

        limits = self.polydata_dict[poly_name]['range_slider'].getRange()
        # print(limits)
        if verbose:
            print('current scalar:', self.polydata_dict[poly_name]['Current_scalar'])

        # scalar_vals = poly_actor.GetMapper().GetInput().GetPointData().GetArray(self.range_dict[poly_name][0])
        # scalar_vals = numpy_support.vtk_to_numpy(scalar_vals)
        scalar_vals = self.polydata_dict[poly_name]['scalar_values'][self.polydata_dict[poly_name]['Current_scalar']]

        if verbose:
            print('check one:', scalar_vals.shape)
        # print(scalar_vals)

        indices = np.logical_and((scalar_vals >= limits[0]), (scalar_vals <= limits[1]))
        # print(indices)
        # self.polydata_dict[poly_name]['Range_indices'] = np.logical_and(self.polydata_dict[poly_name]['Range_indices'], indices)
        self.polydata_dict[poly_name]['Range_indices'] = indices

        if verbose:
            print('check two:', indices, indices.shape)

        # locations = poly_actor.GetMapper().GetInput().GetPoints().GetData()
        # locations = numpy_support.vtk_to_numpy(locations)

        selected_locations = np.copy(self.polydata_dict[poly_name]['locations'][self.polydata_dict[poly_name]['Range_indices']].astype(np.float32))

        selected_locations = np.mean(selected_locations.reshape(-1, 24, 3), axis=1)

        selected_data = np.copy(scalar_vals[self.polydata_dict[poly_name]['Range_indices']].astype(np.float32))

        selected_data = np.mean(selected_data.reshape(-1, 24, 1), axis=1)

        if verbose:
            print('check three', selected_data.shape, selected_locations.shape)

            print('check four', selected_data)

            print('check five', selected_data.dtype)

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(selected_locations))

        scalar_color = vtk.vtkFloatArray()
        scalar_color.SetArray(numpy_support.numpy_to_vtk(selected_data), selected_data.shape[0], 0)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        polydata.GetPointData().SetScalars(scalar_color)

        glyph_type = self.glyph_type
        glyph_scale = self.glyph_scale
        fixed_voxel_size = True

        if glyph_type == 'cube':
            glyphSource = vtk.vtkCubeSource()
        elif glyph_type == 'sphere':
            glyphSource = vtk.vtkSphereSource()

        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
        glyph3D.SetInputData(polydata)
        glyph3D.SetScaleFactor(glyph_scale)
        if fixed_voxel_size:
            glyph3D.SetInputArrayToProcess(0, 0, 0, 0, 'RTData')
        glyph3D.Update()

        poly_actor.GetMapper().SetInputConnection(glyph3D.GetOutputPort())
        poly_actor.GetMapper().SetScalarRange(limits[0], limits[1])
        poly_actor.GetMapper().Modified()

        self.polydata_dict[poly_name]['locations'] = numpy_support.vtk_to_numpy(poly_actor.GetMapper().GetInput().GetPoints().GetData())

        self.polydata_dict[poly_name]['Range_indices'] = indices

        self.iren.Render()

    def slider_change(self):

        slider_val = str(self.slider.value())

    def hide_all_actors(self):

        root = self.listWidget.invisibleRootItem()
        child_count = root.childCount()

        temp_disable_autoupdate = False

        if self.autoupdate:
            self.autoupdate = False
            temp_disable_autoupdate = True

        for index in range(child_count):
            item = root.child(index)

            for c_index in range(item.childCount()):

                # print(item.child(c_index).checkState(0))

                item.child(c_index).setCheckState(0, Qt.Unchecked)
        #             self.actor_visibility[child_index] = int(1)
        #         else:
        #             self.actor_visibility[child_index] = int(0)

        #         child_index += 1

        # for anum, actor in enumerate(self.actor_dict.values()):
        #     actor.SetVisibility(int(self.actor_visibility[anum]))

        # for index in range(self.listWidget.count()):
        #     self.listWidget.item(index).setCheckState(Qt.Unchecked)

        self.rerender()

        if temp_disable_autoupdate:
            self.autoupdate = True

    def show_all_actors(self):
        # for index in range(self.listWidget.count()):
        #     self.listWidget.item(index).setCheckState(Qt.Checked)

        root = self.listWidget.invisibleRootItem()
        child_count = root.childCount()

        temp_disable_autoupdate = False

        if self.autoupdate:
            self.autoupdate = False
            temp_disable_autoupdate = True

        for index in range(child_count):
            item = root.child(index)

            for c_index in range(item.childCount()):

                # print(item.child(c_index).checkState(0))

                item.child(c_index).setCheckState(0, Qt.Checked)

        self.rerender()

        if temp_disable_autoupdate:
            self.autoupdate = True

    def rerender(self):

        # self.actor_visibility = np.zeros(self.listWidget.count())

        root = self.listWidget.invisibleRootItem()
        child_count = root.childCount()

        child_index = 0

        for index in range(child_count):
            item = root.child(index)

            # print(type(item))
            # # for grandchild in item.children():
            # #     print(grandchild)

            # print(item.checkState(0))

            # print(dir(item))

            # print(item.childCount())

            for c_index in range(item.childCount()):

                # print(item.child(c_index).checkState(0))

                if item.child(c_index).checkState(0) == Qt.Checked:
                    self.actor_visibility[child_index] = int(1)
                else:
                    self.actor_visibility[child_index] = int(0)

                child_index += 1

        for anum, actor in enumerate(self.actor_dict.values()):
            actor.SetVisibility(int(self.actor_visibility[anum]))

        # print(self.actor_visibility)

        # for index in range(self.listWidget.count()):
        #     if self.listWidget.item(index).checkState() == Qt.Checked:
        #         self.actor_visibility[index] = int(1)
        #     else:
        #         self.actor_visibility[index] = int(0)

        # for anum, actor in enumerate(self.actor_dict.values()):
        #     actor.SetVisibility(int(self.actor_visibility[anum]))

        # print()

        # print('box radius', b_r)

        if self.use_SSAO:

            # print(sorted(dir(self.ren)))

            # bounds = np.asarray(self.ren.ComputeVisiblePropBounds())

            # b_r = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

            # if self.SSAO_radius_override:

            #     b_r = self.SSAO_radius

            # # print('USING SCREEN SPACE AMBIENT OCCLUSION')

            # occlusion_radius = b_r * 0.1
            # occlusion_bias = b_r * 0.001

            # # print(occlusion_radius)

            # basicPasses = vtk.vtkRenderStepsPass()
            # translucentpass = vtk.vtkTranslucentPass()
            # volumepass = vtk.vtkVolumetricPass()
            # depthpass = vtk.vtkDepthPeelingPass()
            # depthpass.SetTranslucentPass(translucentpass)

            # passes = vtk.vtkRenderPassCollection()
            # passes.AddItem(basicPasses)
            # passes.AddItem(translucentpass)
            # passes.AddItem(volumepass)
            # passes.AddItem(depthpass)

            # seq = vtk.vtkSequencePass()
            # seq.SetPasses(passes)

            # ssao = vtk.vtkSSAOPass()
            # ssao.SetRadius(occlusion_radius)
            # ssao.SetDelegatePass(seq)
            # ssao.SetBias(occlusion_bias)
            # ssao.SetBlur(self.SSAO_blur)
            # ssao.SetKernelSize(self.SSAO_kernel_size)

            # self.ren.SetPass(ssao)

            # # self.ren.SetPass(ssao)

            # self.ren.SetUseDepthPeeling(1)
            # self.ren.SetOcclusionRatio(0.2)
            # self.ren.SetMaximumNumberOfPeels(100)
            # # self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)

            bounds = np.asarray(self.ren.ComputeVisiblePropBounds())

            b_r = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

            occlusion_radius = b_r * 0.1
            occlusion_bias = b_r * 0.001

            lightsP = vtk.vtkLightsPass()
            opaqueP = vtk.vtkOpaquePass()
            translucentP = vtk.vtkTranslucentPass()
            volumeP = vtk.vtkVolumetricPass()

            collection = vtk.vtkRenderPassCollection()
            collection.AddItem(lightsP)

            # print(vtk.VTK_VERSION_NUMBER, vtk.vtkVersion().GetVTKMinorVersion(), vtk.vtkVersion().GetVTKBuildVersion())

            # glowP = vtk.vtkOutlineGlowPass()
            # collection.AddItem(glowP)

            # opaque passes
            ssaoCamP = vtk.vtkCameraPass()
            ssaoCamP.SetDelegatePass(opaqueP)

            ssaoP = vtk.vtkSSAOPass()
            ssaoP.SetRadius(occlusion_radius)
            ssaoP.SetDelegatePass(ssaoCamP)
            ssaoP.SetBias(occlusion_bias)
            ssaoP.SetBlur(True)
            ssaoP.SetKernelSize(256)

            collection.AddItem(ssaoP)

            # translucent and volumic passes
            ddpP = vtk.vtkDualDepthPeelingPass()
            ddpP.SetTranslucentPass(translucentP)
            ddpP.SetVolumetricPass(volumeP)
            collection.AddItem(ddpP)

            overP = vtk.vtkOverlayPass()
            collection.AddItem(overP)




            sequence = vtk.vtkSequencePass()
            sequence.SetPasses(collection)

            fxaaP = vtk.vtkOpenGLFXAAPass()
            fxaaP.SetDelegatePass(sequence)

            camP = vtk.vtkCameraPass()
            camP.SetDelegatePass(fxaaP)

            self.ren.SetPass(camP)

        else:
            pass
            # basicPasses = vtk.vtkRenderStepsPass()
            # fxaaP = vtk.vtkOpenGLFXAAPass()
            # fxaaP.SetDelegatePass(basicPasses)
            # self.ren.GetRenderWindow().SetAlphaBitPlanes(True)
            # self.ren.SetPass(fxaaP)
            # self.ren.SetUseDepthPeeling(True)
            # self.ren.SetOcclusionRatio(1)
            # self.ren.SetMaximumNumberOfPeels(100)

            # basicPasses = vtk.vtkRenderStepsPass()
            # translucentpass = vtk.vtkTranslucentPass()
            # depthpass = vtk.vtkDualDepthPeelingPass()
            # depthpass.SetTranslucentPass(translucentpass)
            # volumepass = vtk.vtkVolumetricPass()

            lightsP = vtk.vtkLightsPass()
            opaqueP = vtk.vtkOpaquePass()
            translucentP = vtk.vtkTranslucentPass()
            volumeP = vtk.vtkVolumetricPass()

            collection = vtk.vtkRenderPassCollection()
            collection.AddItem(lightsP)

            # opaque passes
            ssaoCamP = vtk.vtkCameraPass()
            ssaoCamP.SetDelegatePass(opaqueP)
            collection.AddItem(ssaoCamP)

            ddpP = vtk.vtkDualDepthPeelingPass()
            ddpP.SetTranslucentPass(translucentP)
            ddpP.SetVolumetricPass(volumeP)
            collection.AddItem(ddpP)

            overP = vtk.vtkOverlayPass()
            collection.AddItem(overP)

            sequence = vtk.vtkSequencePass()
            sequence.SetPasses(collection)

            fxaaP = vtk.vtkOpenGLFXAAPass()
            fxaaP.SetDelegatePass(sequence)

            camP = vtk.vtkCameraPass()
            camP.SetDelegatePass(fxaaP)

            self.ren.SetPass(camP)

            # passes = vtk.vtkRenderPassCollection()
            # passes.AddItem(basicPasses)
            # passes.AddItem(translucentpass)
            # passes.AddItem(volumepass)
            # passes.AddItem(depthpass)

            # seq = vtk.vtkSequencePass()
            # seq.SetPasses(passes)

            # fxaaP = vtk.vtkOpenGLFXAAPass()
            # fxaaP.SetDelegatePass(seq)

            self.ren.GetRenderWindow().SetAlphaBitPlanes(True)

            self.ren.SetUseDepthPeeling(True)
            self.ren.SetOcclusionRatio(1)
            self.ren.SetMaximumNumberOfPeels(100)

        self.iren.ReInitialize()


        self.camera.SetParallelProjection(not self.camera_perspective)

        if self.camera_auto_reset:
            self.camera_reset()

        # print('dir:', dir(self.iren.GetPicker()))

        # print('Selected:', self.iren.GetPickingManager().GetNumberOfPickers())

        picked_actor = self.iren.GetPicker().GetActor()

        if picked_actor is not None:

            print('actor:', self.iren.GetPicker().GetActor())

            print('actor properties:', dir(self.iren.GetPicker().GetActor().GetProperty()))

            print('actor options:', dir(self.iren.GetPicker().GetActor()))

            print('actor name:', self.iren.GetPicker().GetActor().__vtkname__)

            print('class name:', self.iren.GetPicker().GetActor().GetClassName())

            # self.iren.GetPicker().GetActor().GetProperty().SetOpacity(0.5)

    def slider_render(self):

        self.iren.ReInitialize()

        self.show()

    def render(self, verbose=False):

        # f = QGLFormat()
        # f.setSampleBuffers(True) # turn on antialiasing
        # QGLFormat.setDefaultFormat(f)

        try:

            # cube_path = '../earth'
            if self.cubemap_path is None:
                self.cubemap_path = 'data/cubemap/lab'
            cube_path = self.cubemap_path  # 'data/cubemap/wires_cold'

            cubemap = vtk_utils.ReadCubeMap(cube_path, '/', '.png', 2)
            self.ren.UseImageBasedLightingOn()
            self.ren.SetEnvironmentTexture(cubemap)

        except:
            print('raytracing failed to initialise, are you on VTK >=9.0?')

        self.vtkWidget = QVTKRenderWindowInteractor(self.vtk_frame)

        self.vtkWidget.showFullScreen()

        # print('widget dir', dir(self.vtkWidget))

        # print(type(self.vtkWidget))

        self.vtkWidget.acceptDrops()

        self.setCentralWidget(self.vtkWidget)
        self.camera = self.ren.GetActiveCamera()

        # self.camera.SetParallelProjection(True)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        # self.vtkWidget.GetRenderWindow().SetSize( 920,1080)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.vtkWidget.GetRenderWindow().SetMultiSamples(1)
        self.vtkWidget.GetRenderWindow().LineSmoothingOn()
        self.vtkWidget.GetRenderWindow().PointSmoothingOn()

        self.vtkWidget.GetRenderWindow().PolygonSmoothingOn()  # this?

        self.iren.SetInteractorStyle(MyInteractorStyle(self.vtkWidget, self.camera, self.vtkWidget.GetRenderWindow()))

        # self.iren.GetInteractorStyle().OnDropFiles()

        self.iren.GetPickingManager().EnabledOn()

        self.propPicker = vtk.vtkPropPicker()
        self.pointPicker = vtk.vtkPointPicker()

        self.iren.GetPickingManager().AddPicker(self.propPicker)
        self.iren.GetPickingManager().AddPicker(self.pointPicker)

        self.ren.GradientBackgroundOn()

        self.ren.SetBackground(1, 1, 1)
        self.ren.SetBackground2(1, 1, 1)

        if self.data is not None:
            self.poly_actor = vtk_utils.add_polydata(self.locations,
                                                     self.data,
                                                     glyph_type=self.glyph_type,
                                                     opacity=1,
                                                     fixed_voxel_size=True,
                                                     glyph_scale=self.glyph_scale,
                                                     colormap='viridis')

            self.add_scalar_pane(self.poly_actor, 'PolyData')
            axes_actor = vtk_utils.add_axes(self.poly_actor, self.ren, axes_type='cartesian', axes_placement='outer')
            self.ren.AddActor(self.poly_actor)
            self.ren.AddActor(axes_actor)
            self.colorbar = add_colorbar(self.poly_actor, self.iren, list(self.data.keys())[0])

        if self.title_text is not None:

            self.title = vtk_utils.add_text(self.title_text, self.iren)

        if self.indicator_type is not None:
            # self.indicator = vtk_utils.add_indicator(self.iren, marker_type=self.indicator_type)
            pass

        # self.balloonRep = vtk.vtkBalloonRepresentation()
        # self.balloonRep.SetBalloonLayoutToImageRight()

        # self.balloonWidget = vtk.vtkBalloonWidget()
        # self.balloonWidget.SetInteractor(self.iren)
        # self.balloonWidget.SetRepresentation(self.balloonRep)

        self.scalar_widgets = dict()

        if self.actor_dict is not None:
            for anum, key_name in enumerate(self.actor_dict.keys()):

                if type(self.actor_dict[key_name]) == vtk.vtkScalarBarActor:

                    # print(self.actor_visibility[anum])
                    # self.actor_dict[key_name].SetDragable(False)

                    scalar_widget = add_colorbar(self.actor_dict[key_name], self.iren)

                    self.scalar_widgets.update({key_name: scalar_widget})

                if isinstance(self.actor_dict[key_name], dict):
                    for sub_actor in self.actor_dict[key_name].values():

                        if type(sub_actor) == vtk.vtkScalarBarActor:

                            # sub_actor.SetDragable(self.actor_visibility[anum])

                            scalar_widget = add_colorbar(sub_actor, self.iren)

                        self.ren.AddActor(sub_actor)

                else:
                    if self.actor_visibility[anum]:

                        # print(dir(self.actor_dict[key_name]))
                        self.ren.AddActor(self.actor_dict[key_name])
                        # self.balloonWidget.AddBalloon(self.actor_dict[key_name], key_name)

        self.ren.ResetCamera()

        # self.balloonWidget.EnabledOn()
        self.vtkWidget._Iren.ConfigureEvent()
        self.vtkWidget.update()


        # try:
        #     rtp = vtk.vtkOSPRayPass()
        #     self.ren.SetPass(rtp)
        #     print('Using OSPRAY Rendering')
        # except AttributeError:

        #     pass

        # renderer = vtk.vtkRenderer()

        # useOspray = True

        # if(useOspray):
        #     print("Using ospray")
        #     osprayPass= vtk.vtkOSPRayPass()
        #     self.ren.SetPass(osprayPass)

        #     osprayNode=vtk.vtkOSPRayRendererNode()
        #     osprayNode.SetSamplesPerPixel(4,self.ren)
        #     osprayNode.SetAmbientSamples(4,self.ren)
        #     osprayNode.SetMaxFrames(4, self.ren)
        # else:
        #     print("Using opengl")

        self.ren.UseFXAAOn()

        self.show()

        self.vtkWidget.GetRenderWindow().Render()

        if self.show_camera_controls:
            self.cam_widget = vtk_utils.add_camera_widget(self.ren, self.iren)

        # self.spline_widget = vtk_utils.add_spline_widget(self.iren)

        if self.caption_dict is not None:
            self.caption_widgets = dict()
            total_keys = len(self.caption_dict.keys())
            print('Adding captions')
            scalar = 1/total_keys
            for k_num, keyname in enumerate(self.caption_dict.keys()):

                offset = 1000  # k_num * scalar

                # print(offset)

                self.caption_widgets[keyname] = vtk_utils.add_caption_widget(self.iren, caption_text=keyname, position=self.caption_dict[keyname], caption_position1=[offset, offset])

        # self.caption_widget = vtk_utils.add_caption_widget(self.iren)

        if verbose:
            print(self.ren)

    def add_help_frame(self, frame_1):

        label_box = QLabel()

        label_box.setText("\nF: Fly to the picked point \nP: Perform a pick operation.\nR: Reset the camera view.\nS: Render as surfaces. \nW: Render as wireframe.")

        frame_1.addWidget(label_box)

    def add_plot_frame(self):

        # label_box = QLabel()

        # label_box.setText("\nF: Fly to the picked point \nP: Perform a pick operation.\nR: Reset the camera view.\nS: Render as surfaces. \nW: Render as wireframe.")

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        data = np.load('test_superpulse.npy')
        sc.axes.plot(data.T.flatten())

        # plot_utils.plot_superpulse(data, fig=sc, show=True)

        sc.show()
        self.verticalLayout.addWidget(sc)

        # self.show()

    def add_slider_frame(self, images, key_name):

        # slider_ranges = np.asarray(self.actor_dict[key_name].GetBounds())
        slider_ranges = np.asarray(images.GetBounds())

        # slider_bounds = [self.actor_dict[key_name].GetSliceNumberMin(), self.actor_dict[key_name].GetSliceNumberMax()]
        # slider_initial = self.actor_dict[key_name].GetSliceNumber()
        slider_initial = images.GetSliceNumber()

        direction = slider_ranges[1::2] - slider_ranges[::2]
        direction = np.argwhere(direction == 0)[0][0]

        # print(dir(images))

        # print(images.GetXRange())
        # print(images.GetYRange())
        # print(images.GetZRange())

        # print(images.GetSliceNumber())
        # print(images.GetSliceNumberMax())
        # print(images.GetSliceNumberMin())

        bounds = [images.GetSliceNumberMin(), images.GetSliceNumberMax()]
        # bounds = [self.actor_dict[key_name].GetWholeZMin(), self.actor_dict[key_name].GetWholeZMax()]

        # print('bounds', bounds)
        coordinates = ['x', 'y', 'z']

        l7 = QLabel("Slicing Control - %s - %s" % (key_name, coordinates[direction]))
        l7.setAlignment(Qt.AlignCenter)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(bounds[0])
        slider.setMaximum(bounds[1])
        midpoint = int((bounds[0] + bounds[1])/2)
        slider.setValue(midpoint)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(5)
        slider.valueChanged.connect(lambda value, key_name=key_name, c=coordinates[direction]: self.valuechange(value, images=images, direction=c))

        self.slider_Frame = QFrame(self)
        # self.slider_Frame.setStyleSheet("background-color: White;")
        self.slider_Frame.setFrameShape(QFrame.StyledPanel)
        self.slider_Frame.setFrameShadow(QFrame.Raised)
        # self.slider_Frame.setMaximumWidth(self.panel_width)
        self.slider_Layout = QVBoxLayout(self.slider_Frame)

        self.slider_Layout.addWidget(l7)
        self.slider_Layout.addWidget(slider)
        self.verticalLayout.addWidget(self.slider_Frame)

    def add_scalar_pane(self, polyactor, poly_name):

        from vtk.util import numpy_support

        try:
            polydata = polyactor.GetMapper().GetInput().GetPointData()
        except AttributeError:
            return
        # polydata = polyactor.GetMapper().GetInput().GetPoints()
        # print(poly_name)

        total_arrays = polydata.GetNumberOfArrays()

        # print('TOTAL ARRAYS:', total_arrays)

        scalar_dict = dict()
        scalar_subdict = dict()
        scalar_dict.update({'Current_scalar': None})

        initial_name = None

        if total_arrays > 0:

            locations = polyactor.GetMapper().GetInput().GetPoints().GetData()
            locations = numpy_support.vtk_to_numpy(locations)

            for iterator in range(total_arrays):

                try:
                    ptdata = polydata.GetArray(iterator)
                    ptdataname = polydata.GetArrayName(iterator)
                    # print('DATA NAME -', ptdataname)

                except:
                    print('NO ARRRAY')
                    return

                if ptdata is not None:

                    ptdata = numpy_support.vtk_to_numpy(ptdata)

                    if ptdata.ndim == 1:
                        scalar_subdict.update({ptdataname: ptdata})
                        if iterator == 0:
                            initial_name = ptdataname
                            # print(ptdata)
                            scalar_bounds = [np.nanmin(ptdata), np.nanmax(ptdata)]
                        scalar_dict['Current_scalar'] = ptdataname

            scalar_dict.update({'scalar_values': scalar_subdict})
            scalar_dict.update({'locations_backup': locations})
            scalar_dict.update({'locations': locations})
            scalar_dict.update({'Range_indices': np.ones(locations.shape[0])})

            self.polydata_dict.update({poly_name: scalar_dict})

        else:
            return

        if initial_name is None:
            # pass
            # print('NO VIABLE ARRRAYS')
            return

        self.scalarFrame = QFrame(self)
        # self.scalarFrame.setStyleSheet("background-color: White;")
        self.scalarFrame.setFrameShape(QFrame.StyledPanel)
        self.scalarFrame.setFrameShadow(QFrame.Raised)
        # self.scalarFrame.setMaximumWidth(self.panel_width)

        self.exitverticalLayout = QVBoxLayout(self.scalarFrame)

        l3 = QLabel()
        l3.setText("%s Scalar Range" % poly_name)
        l3.setAlignment(Qt.AlignCenter)
        self.exitverticalLayout.addWidget(l3)

        comboBox = PyQt5.QtWidgets.QComboBox(self)

        for key_item in scalar_dict['scalar_values'].keys():
            comboBox.addItem(str(key_item))

        comboBox.activated[str].connect(lambda text, poly_actor=polyactor, poly_name=poly_name: self.update_scalar(text, poly_actor, poly_name))

        self.exitverticalLayout.addWidget(comboBox)

        rs = QRangeSlider()

        # slider_dict_name = '%s_slider_dict'

        # glyph_scale = polyactor.GetMapper().GetInputData().GetScaleFactor()

        # print('SCALE________________', glyph_scale)

        self.polydata_dict[poly_name].update({'range_slider': rs})
        # self.polydata_dict[poly_name].update({'slider_ranges':scalar_bounds})

        self.polydata_dict[poly_name]['range_slider'].setMin(scalar_bounds[0])
        self.polydata_dict[poly_name]['range_slider'].setMax(scalar_bounds[1])
        self.polydata_dict[poly_name]['range_slider'].setRange(scalar_bounds[0], scalar_bounds[1])

        # self.polydata_dict[poly_name].update({slider_dict_name:range_dict})

        self.exitverticalLayout.addWidget(self.polydata_dict[poly_name]['range_slider'])
        self.RangeBtn = QPushButton("Apply Range", self.scalarFrame)
        self.exitverticalLayout.addWidget(self.RangeBtn)
        self.RangeBtn.clicked.connect(lambda click, poly_actor=polyactor, poly_name=poly_name: self.get_points_in_range(click, poly_actor, poly_name))

        self.resetBtn = QPushButton("Reset Range", self.scalarFrame)
        self.exitverticalLayout.addWidget(self.resetBtn)
        self.resetBtn.clicked.connect(lambda click, poly_actor=polyactor, poly_name=poly_name: self.reset_range(click, poly_actor, poly_name))
        self.verticalLayout.addWidget(self.scalarFrame)

    def qt_export_scene(self, savename=None):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        savename, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;3D Models (*.wrl *.pov *.pvwgl *.obj *x3d *.gltf *.js *.vtkjs);;2D Images (*.png *.pdf *.svg *.gl2ps)", options=options)
        if savename:
            # print(savename)
            filetype = savename.split('.')[-1]
            export_geometry(self.vtkWidget.GetRenderWindow(), savename, filetype.lower())

    def layoutUI(self):

        self.ren = vtk.vtkRenderer()
        self.ren.UseFXAAOn()
        self.ren.SetUseDepthPeeling(1)
        self.ren.SetMaximumNumberOfPeels(32)


        self.polydata_dict = dict() # a dictionary for holding all the data for polydata frames, super important
        self.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        self.rightFrame = QFrame(self)
        # self.rightFrame.setFrameShape(QFrame.StyledPanel)
        # self.rightFrame.setFrameShadow(QFrame.Raised)
        # self.rightFrame.setGeometry(300, 300, 1280, 800)
        self.panel_width = 300
        # self.rightFrame.setMaximumWidth(self.panel_width * 2)
        self.use_subset = False

        menubar_save = QAction('&Export', self)
        menubar_save.setShortcut('Ctrl+E')
        menubar_save.triggered.connect(self.qt_export_scene)

        menubar_rerender_view = QAction('&Rerender', self)
        menubar_rerender_view.setShortcut('Ctrl+R')
        menubar_rerender_view.triggered.connect(self.rerender)

        menubar_show_all = QAction('&Show all', self)
        menubar_show_all.setShortcut('Ctrl+S')
        menubar_show_all.triggered.connect(self.show_all_actors)

        menubar_hide_all = QAction('&Hide all', self)
        menubar_hide_all.setShortcut('Ctrl+H')
        menubar_hide_all.triggered.connect(self.hide_all_actors)

        bar = QMenuBar()
        file_menu = bar.addMenu("File")
        file_menu.addAction("New")
        file_menu.addAction(menubar_save)

        file_menu.addAction("Quit")
        edit_menu = bar.addMenu("Edit")
        view_menu = bar.addMenu("View")
        view_menu.addAction(menubar_rerender_view)
        view_menu.addAction(menubar_show_all)
        view_menu.addAction(menubar_hide_all)
        help_menu = bar.addMenu("Help")



        self.verticalLayout = QVBoxLayout(self.rightFrame)
        self.verticalLayout.addWidget(bar)

        self.docker1 = QDockWidget(self)
        self.docker1.setMinimumWidth(self.panel_width)
        # self.docker1.setMaximumWidth(2 * self.panel_width)
        self.docker1.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        self.scroll = QScrollArea()

        # self.scroll.setMinimumWidth(self.panel_width)
        # self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.vtk_frame = QFrame(self)

        self.verticalLayout.setSpacing(5)

        self.CameraFrame = QFrame(self)
        # self.CameraFrame.setStyleSheet("background-color: White;")
        self.CameraFrame.setMaximumHeight(600)
        self.CameraLayout = QVBoxLayout(self.CameraFrame)

        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(2)
        l2 = QLabel()
        l2.setText("Camera Control")
        l2.setAlignment(Qt.AlignCenter)

        self.camera_buttons = QButtonGroup()

        coords = list(product((0, 1, 2), (0, 1, 2)))
        coords.append((3, 1))
        labels = ['NW', 'N', 'NE', 'W', 'Top', 'E', 'SW', 'S', 'SE', 'Bottom']
        for num, coord in enumerate(coords):
            x, y = coord
            button = QPushButton(self.CameraFrame)
            button.setFixedSize(60, 60)
            button.setText(labels[num])
            button.setCheckable(True)
            self.camera_buttons.addButton(button)
            self.gridLayout.addWidget(button, x, y)

        self.camera_buttons.buttonClicked.connect(self.set_camera)
        self.CameraLayout.addWidget(l2)
        self.CameraLayout.addLayout(self.gridLayout)

        self.camera_reset_Btn = QPushButton("Reset Camera", self.CameraFrame)
        self.camera_reset_Btn.clicked.connect(self.camera_reset)
        self.camera_reset_Btn.setMaximumWidth(180)
        self.CameraLayout.addWidget(self.camera_reset_Btn)

        self.export_Btn = QPushButton("Export Scene", self.CameraFrame)
        self.export_Btn.setMaximumWidth(180)

        self.CameraLayout.addWidget(self.export_Btn)

        self.add_help_frame(self.CameraLayout)

        # self.add_plot_frame(self.CameraLayout)

        self.camera_auto_reset = True
        self.camera_checkbox = QCheckBox("Auto Reset Camera?", self)
        self.camera_checkbox.setCheckState(QtCore.Qt.Checked)
        self.camera_checkbox.stateChanged.connect(self.camera_auto_reset_func)
        self.camera_checkbox.setMaximumWidth(250)
        self.CameraLayout.addWidget(self.camera_checkbox)

        self.camera_perspective = True
        self.camerap_checkbox = QCheckBox("Use Perspective Rendering?", self)
        self.camerap_checkbox.setCheckState(QtCore.Qt.Checked)
        self.camerap_checkbox.stateChanged.connect(self.camera_perspective_func)
        self.camerap_checkbox.setMaximumWidth(250)
        self.CameraLayout.addWidget(self.camerap_checkbox)

        # spacerItem2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.CameraLayout.addItem(spacerItem2)

        self.verticalLayout.addWidget(self.CameraFrame)

        self.SSAOFrame = QFrame(self)
        # self.SSAOFrame.setStyleSheet("background-color: White;")
        # self.SSAOFrame.setMaximumWidth(self.panel_width)
        self.SSAOFrame.setMaximumHeight(600)
        self.SSAOLayout = QVBoxLayout(self.SSAOFrame)

        l3 = QLabel()
        l3.setText("Ambient Occlusion")
        l3.setAlignment(Qt.AlignCenter)

        self.SSAOLayout.addWidget(l3)

        self.SSAO_checkbox = QCheckBox("Use Ambient Occlusion?", self)
        if self.use_SSAO:
            self.SSAO_checkbox.setCheckState(QtCore.Qt.Checked)
        else:
            self.SSAO_checkbox.setCheckState(QtCore.Qt.Unchecked)
        self.SSAO_checkbox.stateChanged.connect(self.SSAO_func)
        # self.SSAO_checkbox.setMaximumWidth(250)
        self.SSAOLayout.addWidget(self.SSAO_checkbox)

        self.SSAO_blur_checkbox = QCheckBox("Blur AO?", self)
        self.SSAO_blur_checkbox.setCheckState(QtCore.Qt.Checked)
        self.SSAO_blur_checkbox.stateChanged.connect(self.SSAO_blur_func)
        # self.SSAO_blur_checkbox.setMaximumWidth(250)
        self.SSAO_blur = True
        self.SSAOLayout.addWidget(self.SSAO_blur_checkbox)

        SSAO_slider = QSlider(Qt.Horizontal)
        SSAO_slider.setMinimum(0)
        SSAO_slider.setMaximum(500)
        midpoint = 250
        SSAO_slider.setValue(midpoint)
        SSAO_slider.setTickPosition(QSlider.TicksBelow)
        SSAO_slider.setTickInterval(20)
        SSAO_slider.valueChanged.connect(lambda value: self.SSAO_valuechange(value))

        self.SSAO_radius_override = False
        self.SSAO_radius = 250

        l4 = QLabel()
        l4.setText("Radius Override")
        l4.setAlignment(Qt.AlignCenter)

        self.SSAOLayout.addWidget(l4)

        self.SSAOLayout.addWidget(SSAO_slider)

        SSAO_kernel_slider = QSlider(Qt.Horizontal)
        SSAO_kernel_slider.setMinimum(0)
        SSAO_kernel_slider.setMaximum(2048)
        midpoint = 250
        self.SSAO_kernel_size = midpoint
        SSAO_kernel_slider.setValue(midpoint)
        SSAO_kernel_slider.setTickPosition(QSlider.TicksBelow)
        SSAO_kernel_slider.setTickInterval(20)
        SSAO_kernel_slider.valueChanged.connect(lambda value: self.SSAO_kernel_valuechange(value))

        l5 = QLabel()
        l5.setText("Kernel Size")
        l5.setAlignment(Qt.AlignCenter)
        self.SSAOLayout.addWidget(l5)

        self.SSAOLayout.addWidget(SSAO_kernel_slider)

        # spacerItem2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.CameraLayout.addItem(spacerItem2)

        self.verticalLayout.addWidget(self.SSAOFrame)

        check_list = list()

        self.slider_ranges = list()
        # has_slices = False

        if self.actor_dict is not None:

            self.scalar_2_Frame = QFrame(self)
            # self.scalar_2_Frame.setStyleSheet("background-color: White;")

            # self.scalar_2_Frame.setMaximumWidth(self.panel_width)

            self.scalarverticalLayout = QVBoxLayout(self.scalar_2_Frame)

            self.verticalLayout.addWidget(self.scalar_2_Frame)

            l1 = QLabel()
            l1.setText("Actor Visibility")
            l1.setAlignment(Qt.AlignCenter)

            self.scalarverticalLayout.addWidget(l1)

            self.listWidget = QtWidgets.QTreeWidget()

            self.slider_keys = list()

            axes_dict = dict()
            axes_keys = list()
            scalarbar_dict = dict()
            actor_dict_1D = dict()

            iterator = 0

            for key_name in self.actor_dict:

                check_list.append(key_name)

                if isinstance(self.actor_dict[key_name], dict):

                    # print('item is dict')

                    parent = QtWidgets.QTreeWidgetItem(self.listWidget)
                    parent.setText(0, key_name)
                    parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

                    # print(dir(parent))



                    for subkey in self.actor_dict[key_name]:

                        child = QtWidgets.QTreeWidgetItem(parent)
                        child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                        child.setText(0, subkey)
                        child.setCheckState(0, Qt.Checked)
                        # child.stateChanged.connect(self.rerender())

                        if subkey in actor_dict_1D.keys():

                            warnings.warn('Actor has the same key as one in the complete dict, this will mess up rendering and visibility!')

                        actor_dict_1D.update({subkey: self.actor_dict[key_name][subkey]})

                        iterator += 1
                        # try:
                        if (self.actor_dict[key_name][subkey].GetClassName()) == 'vtkImageActor':

                            self.add_slider_frame(self.actor_dict[key_name][subkey], key_name)

                        # print(self.actor_dict[key_name][subkey].GetMapper().GetClassName())

                        if self.actor_dict[key_name][subkey].GetMapper().GetClassName() in ['vtkOpenGLPolyDataMapper', 'vtkDataSetMapper'] :

                            # print(self.actor_dict[key_name])
                            self.add_scalar_pane(self.actor_dict[key_name][subkey], key_name)

                            if key_name[0] == '<':
                                axes_actor = vtk_utils.add_axes(self.actor_dict[key_name][subkey], self.ren, axes_type='cartesian', axes_placement='outer')
                                axes_key = '%s-%s-axes' % (key_name, subkey)
                                axes_dict.update({axes_key: axes_actor})
                                axes_keys.append(axes_key)
                            if key_name[0] == '(':
                                axes_actor = vtk_utils.add_axes(self.actor_dict[key_name][subkey], self.ren, axes_type='polar', axes_placement='outer')
                                axes_key = '%s-%s-axes' % (key_name, subkey)
                                axes_dict.update({axes_key: axes_actor})
                                axes_keys.append(axes_key)

                            if key_name[0] == '[':
                                bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name][subkey], title=subkey, return_widget=False)
                                bar_name = 'colorbar_' + key_name + '_' + subkey
                                scalarbar_dict.update({bar_name: bar_actor})

                            if key_name[1] == '[':
                                bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name][subkey], title=subkey, return_widget=False)
                                bar_name = 'colorbar_' + key_name + '_' + subkey
                                scalarbar_dict.update({bar_name: bar_actor})

                        # except AttributeError:
                        #     pass

                    continue

                elif isinstance(self.actor_dict[key_name], list):

                    # print('item is list')

                    parent = QtWidgets.QTreeWidgetItem(self.listWidget)
                    parent.setText(0, key_name)
                    parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

                    for s_num in range(len(self.actor_dict[key_name])):

                        child = QtWidgets.QTreeWidgetItem(parent)
                        child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                        subname = key_name + '_' + str(s_num)
                        child.setText(0, subname)
                        child.setCheckState(0, Qt.Checked)

                        # print('list item', subname)

                        actor_dict_1D.update({subname: self.actor_dict[key_name][s_num]})

                        iterator += 1

                        if (self.actor_dict[key_name][s_num].GetClassName()) == 'vtkImageActor':

                            self.add_slider_frame(self.actor_dict[key_name][s_num], key_name)

                        try:

                            if self.actor_dict[key_name][s_num].GetMapper().GetClassName() in ['vtkOpenGLPolyDataMapper', 'vtkDataSetMapper'] :
                                # print(key_name, 'is polydata -----------------------------')
                                # print(self.actor_dict[key_name])
                                self.add_scalar_pane(self.actor_dict[key_name][s_num], key_name)
                                if key_name[0] == '<':
                                    axes_actor = vtk_utils.add_axes(self.actor_dict[key_name][s_num], self.ren, axes_type='cartesian', axes_placement='outer')
                                    axes_key = '%s-%i-axes' % (key_name, s_num)
                                    axes_dict.update({axes_key: axes_actor})
                                    axes_keys.append(axes_key)
                                if key_name[0] == '(':
                                    axes_actor = vtk_utils.add_axes(self.actor_dict[key_name][s_num], self.ren, axes_type='polar', axes_placement='outer')
                                    axes_key = '%s-%i-axes' % (key_name, s_num)
                                    axes_dict.update({axes_key: axes_actor})
                                    axes_keys.append(axes_key)

                                if key_name[0] == '[':
                                    bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name][s_num], title=key_name, return_widget=False)
                                    bar_name = 'colorbar_' + key_name + '_' + str(s_num)
                                    scalarbar_dict.update({bar_name: bar_actor})

                                if key_name[1] == '[':
                                    bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name][s_num], title=key_name, return_widget=False)
                                    bar_name = 'colorbar_' + key_name + '_' + str(s_num)
                                    scalarbar_dict.update({bar_name: bar_actor})

                        except AttributeError:

                            print('Actor has no attribute GetClassName')

                    continue

                else:

                    iterator += 1

                    actor_dict_1D.update({key_name: self.actor_dict[key_name]})

                    parent = QtWidgets.QTreeWidgetItem(self.listWidget)
                    parent.setText(0, key_name)
                    parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setText(0, key_name)
                    child.setCheckState(0, Qt.Checked)
                try:
                    if (self.actor_dict[key_name].GetClassName()) == 'vtkImageActor':

                        self.add_slider_frame(key_name)
                except AttributeError:
                    print(self.actor_dict[key_name])

                # try:
                # print('ACTOR +++++++++++++', key_name, self.actor_dict[key_name].GetMapper().GetClassName())
                if self.actor_dict[key_name].GetMapper().GetClassName() in ['vtkOpenGLPolyDataMapper', 'vtkDataSetMapper'] :
                    # print(key_name, 'is polydata -----------------------------')
                    # print(self.actor_dict[key_name])
                    self.add_scalar_pane(self.actor_dict[key_name], key_name)
                    if key_name[0] == '<':
                        axes_actor = vtk_utils.add_axes(self.actor_dict[key_name], self.ren, axes_type='cartesian', axes_placement='outer')
                        axes_key = '%s-axes' % key_name
                        axes_dict.update({axes_key: axes_actor})
                        axes_keys.append(axes_key)
                    if key_name[0] == '(':
                        axes_actor = vtk_utils.add_axes(self.actor_dict[key_name], self.ren, axes_type='polar', axes_placement='outer')
                        axes_key = '%s-axes' % key_name
                        axes_dict.update({axes_key: axes_actor})
                        axes_keys.append(axes_key)

                    if key_name[0] == '[':
                        bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name], title=key_name, return_widget=False)
                        bar_name = 'colorbar_' + key_name
                        scalarbar_dict.update({bar_name: bar_actor})

                    if key_name[1] == '[':
                        bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name], title=key_name, return_widget=False)
                        bar_name = 'colorbar_' + key_name
                        scalarbar_dict.update({bar_name: bar_actor})

                

                # except AttributeError:

                #     print('actor does not have a mapper? is it 2D?')

                #     print(type(self.actor_dict[key_name]))

                #     print(self.actor_dict[key_name])

                #     continue
                
                try:
                    # print('ACTOR +++++++++++++', key_name, self.actor_dict[key_name].GetMapper().GetClassName())
                    if self.actor_dict[key_name].GetMapper().GetClassName() == 'vtkDataSetMapper':
                        # print(key_name, 'is polydata -----------------------------')
                        # print(self.actor_dict[key_name])
                        self.add_scalar_pane(self.actor_dict[key_name], key_name)
                        if key_name[0] == '<':
                            axes_actor = vtk_utils.add_axes(self.actor_dict[key_name], self.ren, axes_type='cartesian', axes_placement='outer')
                            axes_key = '%s-axes' % key_name
                            axes_dict.update({axes_key: axes_actor})
                            axes_keys.append(axes_key)
                        if key_name[0] == '(':
                            axes_actor = vtk_utils.add_axes(self.actor_dict[key_name], self.ren, axes_type='polar', axes_placement='outer')
                            axes_key = '%s-axes' % key_name
                            axes_dict.update({axes_key: axes_actor})
                            axes_keys.append(axes_key)

                        if key_name[0] == '[':
                            bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name], title=key_name, return_widget=False)
                            bar_name = 'colorbar_' + key_name
                            scalarbar_dict.update({bar_name: bar_actor})

                        if key_name[1] == '[':
                            bar_actor = vtk_utils.add_colorbar(self.actor_dict[key_name], title=key_name, return_widget=False)
                            bar_name = 'colorbar_' + key_name
                            scalarbar_dict.update({bar_name: bar_actor})

                    

                except AttributeError:

                    print('actor does not have a mapper? is it 2D?')

                    # plotWidget = vtk.vtkXYPlotWidget()

                    # print(self.actor_dict[key_name])

                    print(type(self.actor_dict[key_name]))

                    # plotWidget.SetXYPlotActor(self.actor_dict[key_name])

                    continue



            self.actor_dict = actor_dict_1D

            if len(axes_dict.keys()) != 0:

                parent = QtWidgets.QTreeWidgetItem(self.listWidget)
                parent.setText(0, 'Axes')
                parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

                for axes_name in axes_dict.keys():

                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)

                    child.setText(0, axes_name)
                    child.setCheckState(0, Qt.Checked)

                    iterator += 1

            if len(scalarbar_dict.keys()) != 0:

                parent = QtWidgets.QTreeWidgetItem(self.listWidget)
                parent.setText(0, 'Scalar Bars')
                parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

                for bar_name in scalarbar_dict.keys():

                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)

                    child.setText(0, bar_name)
                    child.setCheckState(0, Qt.Checked)

                    iterator += 1

            self.actor_visibility = np.ones(iterator)

            self.actor_dict.update(axes_dict)
            self.actor_dict.update(scalarbar_dict)

            check_list = check_list + axes_keys

            self.listWidget.setMinimumHeight(400)

            self.autoupdate_checkbox = QCheckBox("Autoupdate on change?", self)
            self.autoupdate_checkbox.setCheckState(QtCore.Qt.Checked)
            self.autoupdate_checkbox.stateChanged.connect(self.tree_update_checkbox_state)          
            self.autoupdate = True

            self.scalarverticalLayout.addWidget(self.listWidget)
            self.scalarverticalLayout.addWidget(self.autoupdate_checkbox)

            self.actor_button_frame = QFrame(self)
            # self.actor_button_frame.setStyleSheet("background-color: white;")

            self.horizontalLayout = QHBoxLayout(self.actor_button_frame)

            self.hideBtn = QPushButton("Hide all", self.actor_button_frame)
            self.hideBtn.clicked.connect(self.hide_all_actors)

            self.horizontalLayout.addWidget(self.hideBtn)

            self.showBtn = QPushButton("Show all", self.actor_button_frame)
            self.showBtn.clicked.connect(self.show_all_actors)

            self.horizontalLayout.addWidget(self.showBtn)

            self.renderBtn = QPushButton("Redisplay", self.actor_button_frame)
            self.renderBtn.clicked.connect(self.rerender)

            self.horizontalLayout.addWidget(self.renderBtn)
            self.scalarverticalLayout.addWidget(self.actor_button_frame)

        self.scroll.setWidget(self.rightFrame)
        self.docker1.setWidget(self.scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.docker1)

        self.render()

        try:
            self.listWidget.itemChanged.connect(self.tree_updater)
        except AttributeError:
            print('list of actors not used, disabling autoupdate')

        self.export_Btn.clicked.connect(self.qt_export_scene)



def add_colorbar(actor,
                 interactor=None,
                 title='Values',
                 orientation='vertical',
                 return_widget=True,
                 total_ticks=10,
                 background=True,
                 opacity=1.0):
    """[Adds a colorbar to the interactor]

    Arguments:
        interactor {[VTK Interactor]} -- [the interactor used by VTK]
        title {[string]} -- [the title of the colorbar]
        mapper {[VTK mapper]} -- [the VTK mapper used to grab color values]

    Keyword Arguments:
        orientation {str} -- [orientation of the colorbar] (default: {'vertical'})
        return_widget {bool} -- [returns the colorbar as a widget instead] (default: {True})

    Returns:
        [type] -- [description]
    """

    if type(actor) == vtk.vtkScalarBarActor:

        scalar_bar = actor

    else:

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(actor.GetMapper().GetLookupTable())
        scalar_bar.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
        scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()
        scalar_bar.GetLabelTextProperty().ItalicOff()
        # scalar_bar.GetLabelTextProperty().ShadowOff()
        # scalar_bar.GetLabelTextProperty().FrameOn()
        scalar_bar.GetLabelTextProperty().SetShadowOffset(0,0)
        scalar_bar.GetLabelTextProperty().GetShadowColor((1,1,1))#.SetValue(1,1,1)
        # scalar_bar.GetLabelTextProperty().SetBackgroundOpacity(0.5)
        # scalar_bar.GetLabelTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        # print('SHADOW', scalar_bar.GetLabelTextProperty().GetShadow())

        scalar_bar.SetTitle(title)
        scalar_bar.GetTitleTextProperty().SetColor(0.0, 0.0, 0.0)
        # scalar_bar.SetUnconstrainedFontSize(40)
        scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
        scalar_bar.GetTitleTextProperty().ShadowOff()
        scalar_bar.GetTitleTextProperty().ItalicOff()
        scalar_bar.AnnotationTextScalingOn()
        scalar_bar.SetVerticalTitleSeparation(10)

        # scalar_bar.SetOrientationToHorizontal()
        # scalar_bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        # scalar_bar.GetPositionCoordinate().SetValue(.2,.05)
       
        # scalar_bar.SetWidth( 0.7 )
        # scalar_bar.SetHeight( 0.1 )
        # scalar_bar.SetPosition( 0.1, 0.01 )
        # scalar_bar.SetTextPad(10)
        # scalar_bar.DrawFrameOn()
        # scalar_bar.GetFrameProperty().SetOpacity(0.5)
        # scalar_bar.GetFrameProperty().SetLineWidth(10)
        # scalar_bar.SetMaximumWidthInPixels(200)
        # scalar_bar.GetTitleTextProperty().SetBackgroundOpacity(0.5)
        # scalar_bar.GetTitleTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        # scalar_bar.GetTitleTextProperty().SetOrientation(90)

        # print(dir(scalar_bar))
        if orientation == 'Horizontal':
            scalar_bar.SetOrientationToHorizontal()
        else:
            scalar_bar.SetOrientationToVertical()
        scalar_bar.SetWidth(0.2)
        # scalar_bar.SetHeight(0.1)
        scalar_bar.SetVisibility(1)
        scalar_bar.SetNumberOfLabels(total_ticks)
        scalar_bar.UseOpacityOn()
        if background:
            scalar_bar.DrawBackgroundOn()
            scalar_bar.GetBackgroundProperty().SetOpacity(opacity)
        # scalar_bar.SetWidth(10)
        scalar_bar.SetBarRatio(0.85)

        # print(scalar_bar)

        # scalar_bar.GetPositionCoordinate().SetValue(0.05, 0.01)

    # create the scalar_bar_widget
    if return_widget:
        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetInteractor(interactor)
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.On()
        scalar_bar_widget.ResizableOn()

        return scalar_bar_widget

    return scalar_bar



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure()  # figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


def render_data(locations=None, data=None, edge_data=None, actor_dict=None, glyph_type='cube', glyph_scale=1, data_type='basis', cubemap=None, slicing_vals=None, title=None, show_camera_controls=False, use_SSAO=False, caption_dict=None, verbose=False):
    app = QApplication(['LiverView Visualiser'])
    app.setWindowIcon(QtGui.QIcon('window_icon.png'))
    if locations is None:
        if data is not None:
            x_vals = data['Euclidean X']
            y_vals = data['Euclidean Y']
            z_vals = data['Euclidean Z']

            locations = (np.vstack([x_vals, y_vals, z_vals]).T)

    if data is None:
        if locations is not None:
            data = dict()
            data.update({'Euclidean X': locations[:,0]})
            data.update({'Euclidean Y': locations[:,1]})
            data.update({'Euclidean Z': locations[:,2]})


    if verbose:
        print('Rendering data')

    render_window = Widget(locations, data=data, actor_dict=actor_dict, edge_data=edge_data, glyph_type=glyph_type,
                           glyph_scale=glyph_scale, data_type=data_type, slicing_vals=slicing_vals, title=title, cubemap=cubemap,
                           show_camera_controls=show_camera_controls, use_SSAO=use_SSAO, caption_dict=caption_dict)
    render_window.show()

    if verbose:
        print('Data rendered')
    app.exec_()


if __name__ == '__main__':

    locations = np.random.rand(100,3) * 100

    render_data(locations=locations)
