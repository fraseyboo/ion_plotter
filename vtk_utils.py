import vtk
from vtk.util import numpy_support
import math
from tqdm import tqdm, trange
import numpy as np
import matplotlib

try:
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
except:
    print('Could not Latex-ify labels ')


print('Current backend', matplotlib.pyplot.get_backend())

conflicting_backends = ['QtAgg', 'Qt5Agg']

if matplotlib.pyplot.get_backend() in conflicting_backends:
    try:
        matplotlib.use('GTK3Agg')
        print('backend set to GTK3Agg to avoid conflict with VTK plot')
    except:
        print('Tried to switch to GTK3, failed')
        print('Currently using:', matplotlib.get_backend())



def set_camera(camera, position):
    pos_dict = {'N': np.array([0, 1, 0]),
                'E': np.array([1, 0, 0]),
                'S': np.array([0, -1, 0]),
                'W': np.array([-1, 0, 0]),
                'Top': np.array([0, 0, 1]),
                'Bottom': np.array([0, 0, -1]),
                'NE': np.array([0.707, 0.707, 0]),
                'NW': np.array([0.707, -0.707, 0]),
                'SE': np.array([-0.707, 0.707, 0]),
                'SW': np.array([-0.707, -0.707, 0])}

    pos = np.asarray(camera.GetPosition())
    cent = np.asarray(camera.GetFocalPoint())

    # print(dir(camera))

    rad = np.sqrt(np.sum(np.square(pos-cent)))

    new_pos = pos_dict[position]*rad

    camera.SetPosition(new_pos)
    up = np.asarray([0, 0, 1])
    # camera.SetViewUp(pos_dict[position])
    camera.SetViewUp(up)

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


def make_colormap(c_range=(0,1), color='viridis', cnum=100, invert=False, verbose=False, dark_mode=False):


    # Possible values are: 

    # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r,
    # BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu,
    # GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r,
    # PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r,
    # PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r,
    # Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
    # RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r,
    # Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu,
    # YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r,
    # autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r,
    # cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,
    # cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray,
    # gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow,
    # gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot,
    # gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r,
    # inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral,
    # nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r,
    # prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring,
    # spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b,
    # tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r,
    # twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r

    if verbose:
        if color is None:
            print('No colormap supplied, defaulting to Viridis')
        else:
            print('Making colormap:', color)

    if verbose:
        print('Using Matplotlib color tables')

    try:
        mpl_data = matplotlib.cm.get_cmap(color)
    except ValueError:
        try:
            color = color[5:]
            print(color)
            mpl_data = matplotlib.cm.get_cmap(color)
            dark_mode = True
            print('Using Dark Mode')
        except ValueError:
            mpl_data = matplotlib.cm.get_cmap('viridis')
    cm_data = np.zeros([cnum, 3])
    for i in range(cnum):
        cm_data[i,:] = mpl_data(i/cnum)[0:3]

    colormap = np.asarray(cm_data)

    if dark_mode:
        colormap = colormap * [0.1, 0.1, 0.1]

    if invert:
        if verbose:
            print('Using inverted direction')
        colormap = np.flip(colormap, axis=0)

    scale = np.linspace(c_range[0], c_range[1], num=colormap.shape[0])

    colormap = np.vstack([scale, colormap.T]).T

    if verbose:
        print('Colormap made')

    return colormap


def make_LUT(colormap='viridis', invert=False, verbose=False, c_range=(0,1), nan_color=(1,0,0,1), scale_type='linear'):

    colormap = make_colormap(c_range=c_range, color=colormap, invert=invert, verbose=verbose)

    colorSeries = vtk.vtkColorSeries()
    colorSeries.SetNumberOfColors(colormap.shape[0])
    try:
        for cnum, color in enumerate(colormap):
            color = (255*color[1:4]).astype(int) 
            vcolor = vtk.vtkColor3ub(color)
            colorSeries.SetColor(cnum, vcolor)
        lut = vtk.vtkLookupTable()
        colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
        lut.SetNanColor(nan_color)
    except TypeError:
        for cnum, color in enumerate(colormap):
            color = (255*color[1:4]).astype(int) 
            vcolor = vtk.vtkColor3ub(color[0], color[1], color[2])
            colorSeries.SetColor(cnum, vcolor)
        lut = vtk.vtkLookupTable()
        colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
        lut.SetNanColor(nan_color)
        lut.SetRange(c_range)

    if scale_type == 'linear':

        lut.SetScaleToLinear()

    elif scale_type == 'logarithmic':

        lut.SetScaleToLog10()

    return lut



def add_polydata(locations,
                 scalar_dict=None,
                 initial_key=None,
                 renderer=None,
                 glyph_type='cube',
                 glyph_scale=2,
                 fixed_voxel_size=True,
                 colormap='viridis',
                 opacity=1.0,
                 colors=None,
                 resolution=24,
                 verbose=False):
    """[Creates a polydata actor from a set of coordinates & scalars]

    Arguments:
        locations {[numpy array (N,3)]} -- [the locations of the points]
        scalar_vals {[dictionary]} -- [the scalar values in a key:value dictionary]

    Keyword Arguments:
        glyph_type {str} -- [the shape of the glyph used, can be sphere] (default: {'cube'})
        glyph_scale {int} -- [the default scale of the glyph] (default: {2})
        fixed_voxel_size {bool} -- [bool if you want the glyphs to scale] (default: {True})
        colormap {[type]} -- [colormap of the polydata] (default: {None})
        opacity {float} -- [polydata opacity, use with caution as it affects rendering] (default: {1.0})
        resolution {int} -- [the rendering resolution of the sphere glyph] (default: {6})
    """

    if verbose:
        print('Input shape:', locations.shape)

    if np.ndim(locations) < 2:
        print('input array appears to be 1D, shape:', locations.shape,  'size:', locations.size)
        if locations.size == 3:
            locations = locations.reshape([-1,3])
        elif locations.size == 2:
            locations = np.vstack([locations, 0])
            locations = locations.reshape([-1,3])
        else:
            locations = locations[:3].reshape([-1,3])

    if locations.shape[1] > 3:
        print('3D point cloud expected, got', locations.shape)
        locations = locations[:,:3]
        print('Points truncated to', locations.shape)

    elif locations.shape[1] == 2:
        print('3D point cloud expected, got', locations.shape)
        locations = np.hstack((locations, np.zeros((locations.shape[0],1))))
        print('Points padded to', locations.shape)
    

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(locations.astype(np.float)))
    # for location in locations:
    #     points.InsertNextPoint(location[0], location[1], location[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    total_points = points.GetNumberOfPoints()

    if scalar_dict is not None:
        if isinstance(scalar_dict, np.ndarray):
            if verbose:
                print('array specified as scalars, shape', scalar_dict.shape, 'total points expected:', total_points)

            scalar_array = scalar_dict

            scalar_dict = dict()

            if np.ndim(scalar_array) == 2:

                # print('Array is 2D')

                if scalar_array.shape[0] == total_points:
                    # print('scalar array is tall')
                    for s_num, scalar in enumerate(scalar_array.T):
                        # print(scalar.shape)
                        key_name = 'array_%i' % s_num
                        scalar_dict.update({key_name: scalar})

                elif scalar_array.shape[1] == total_points:
                    # print('scalar array is wide')
                    for s_num, scalar in enumerate(scalar_array):
                        # print(scalar.shape)
                        key_name = 'array_%i' % s_num
                        scalar_dict.update({key_name: scalar})


            elif np.ndim(scalar_array) == 1:

                # print('Array is 1D')

                scalar_dict.update({'scalar_0':scalar_array})
            # for scalar in scalar_array:
            #     print(scalar.shape)

            else:
                scalar_dict = dict({'value': np.zeros(locations.shape[0])})


            # scalar_dict = dict({'value': np.zeros(locations.shape[0])})
    else: 
        scalar_dict = dict({'value': np.zeros(locations.shape[0])})

    for key_name in scalar_dict.keys():

        # print('Key name', key_name)

        scalar_color = vtk.vtkFloatArray()
        scalar_color.SetNumberOfComponents(1)
        scalar_color.SetName(key_name)
        scalar_vals = scalar_dict[key_name]

        # print(np.nanmin(scalar_vals), np.nanmax(scalar_vals))

        for pointId in range(total_points):
            try:
                scalar_color.InsertNextValue(scalar_vals[pointId])
            except:
                scalar_color.InsertNextValue(0)
                break

        polydata.GetPointData().AddArray(scalar_color)

    # print('saved_keyname', key_name)

    polydata.GetPointData().SetScalars(scalar_color)
    polydata.GetPointData().SetActiveScalars(key_name)

    if colors is not None:

        Colors = vtk.vtkUnsignedCharArray()
        Colors.SetNumberOfComponents(3)
        Colors.SetName("Colors")
        if isinstance(colors, str):
            print('Singular named color for set assumed, google \'VTK named colors\' for options')
            if colors[:5] == 'dark_':
                print('Dark mode enabled')
                colors = colors[5:]
                single_color = (np.asarray(vtk.vtkNamedColors().GetColor3d(colors)) * 0.3)
            else:
                single_color = vtk.vtkNamedColors().GetColor3d(colors)


            single_color = np.asarray((single_color[0], single_color[1], single_color[2] )) * 255

            # print(single_color)
            for _ in range(total_points):
                Colors.InsertNextTuple3(*single_color)

        elif np.ndim(colors) == 1:
            print('Singular  RGB [0-255] color triple for set assumed')
            for _ in range(total_points):
                Colors.InsertNextTuple3(*colors)

        else:
            print('List of RGB [0-255] color triples for set assumed')
            for color in colors:
                # print(color)
                Colors.InsertNextTuple3(*color)

        polydata.GetPointData().SetScalars(Colors)
        polydata.Modified()



    if initial_key is not None:

        scalar_color = vtk.vtkFloatArray()
        scalar_color.SetNumberOfComponents(1)
        scalar_color.SetName(key_name)
        scalar_vals = scalar_dict[initial_key]

        for pointId in range(total_points):
            try:
                scalar_color.InsertNextValue(scalar_vals[pointId])
            except:
                scalar_color.InsertNextValue(0)
                break

        polydata.GetPointData().SetScalars(scalar_color)


    if glyph_type == 'sphere':
        glyphSource = vtk.vtkSphereSource()
        glyphSource.SetPhiResolution(resolution)
        glyphSource.SetThetaResolution(resolution)
    elif glyph_type == 'tetrahedron':
        glyphSource = vtk.vtkPlatonicSolidSource()
        glyphSource.SetSolidTypeToTetrahedron()
    elif glyph_type == 'octahedron':
        glyphSource = vtk.vtkPlatonicSolidSource()
        glyphSource.SetSolidTypeToOctahedron()
    elif glyph_type == 'icosahedron':
        glyphSource = vtk.vtkPlatonicSolidSource()
        glyphSource.SetSolidTypeToIcosahedron()
    elif glyph_type == 'dodecahedron':
        glyphSource = vtk.vtkPlatonicSolidSource()
        glyphSource.SetSolidTypeToDodecahedron()
    else: # default to cube
        glyphSource = vtk.vtkCubeSource()

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceConnection(glyphSource.GetOutputPort())
    glyph3D.SetInputData(polydata)

    if fixed_voxel_size:
        if verbose:
            print('Fixed voxel size')
        glyph3D.SetScaleFactor(glyph_scale)
        glyph3D.SetInputArrayToProcess(0, 0, 0, 0, 'RTData')
        glyph3D.SetColorModeToColorByScalar()
    else:
        if verbose:
            print('Scalar voxel size')
        glyph3D.ClampingOff()
        glyph3D.SetScaleModeToScaleByScalar()
        # glyph3D.SetInputArrayToProcess(0, 0, 0, Colors, key_name)
        glyph3D.SetScaleFactor(glyph_scale)
        
    glyph3D.Update()

    lut = make_LUT(colormap=colormap)

    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(glyph3D.GetOutputPort())
    try:
        mapper.SetScalarRange(np.nanmin(scalar_vals), np.nanmax(scalar_vals))
    except ValueError:
        mapper.SetScalarRange(0, 1)
    mapper.SetLookupTable(lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetOpacity(opacity)

    if renderer is None:
        return actor  # , mesh_actor, volume_test]
    else:
        renderer.AddActor(actor)




def add_polyhedron(vertices, faces, labels=None, offset=[0, 0, 0], scalars=None, secondary_offset=None, rotation=None, opacity=1.0, verbose=False, mesh_color='black', color_map='viridis', c_range=None, representation='surface', interpolate_scalars=True):

    colors = vtk.vtkNamedColors()

    points = vtk.vtkPoints()
    visited = list()
    narrowed_vertices = list()

    if labels is not None:
        if verbose:
            print('assuming points need relabelling reduction')
        for face in faces:
            for input_vertex in face:

                if input_vertex not in visited:

                    vertex = vertices[np.argwhere(labels == input_vertex), :].flatten() + offset
                    narrowed_vertices.append(vertex)
                    visited.append(input_vertex)

        narrowed_vertices = np.asarray(narrowed_vertices)
        visited = np.asarray(visited)


        new_faces = list()

        for face in faces:
            new_face = [np.argwhere(visited == face[0]), np.argwhere(visited == face[1]), np.argwhere(visited == face[2])]
            new_faces.append(new_face)

        faces = np.squeeze(np.asarray(new_faces))

    else:
        narrowed_vertices = np.asarray(vertices) + offset

    for vertex in narrowed_vertices:

        points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

    cell_array = vtk.vtkCellArray()

    if faces.shape[1] == 3:

        for face in faces:

            Triangle = vtk.vtkTriangle()
            Triangle.GetPointIds().SetId(0,face[0])
            Triangle.GetPointIds().SetId(1,face[1])
            Triangle.GetPointIds().SetId(2,face[2])
            cell_array.InsertNextCell(Triangle)

    elif faces.shape[1] == 4:

        for face in faces:

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0,face[0])
            quad.GetPointIds().SetId(1,face[1])
            quad.GetPointIds().SetId(2,face[2])
            quad.GetPointIds().SetId(3,face[3])
            cell_array.InsertNextCell(quad)

    else:

        for face in faces:

            cell = vtk.vtkPolygon()

            for p_num, point in enumerate(face):

                cell.GetPointIds().SetId(p_num, point)

            cell_array.InsertNextCell(cell)


    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cell_array)

    # Create a mapper and actor
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))

    if scalars is not None:

        if c_range is None:
            c_range = (np.min(scalars), np.max(scalars))
            
        lut = make_LUT(colormap=color_map, c_range=c_range, nan_color=(0,1,0,1))

        actor.GetMapper().SetLookupTable(lut)
        actor.GetMapper().SetScalarRange(c_range)
        actor.GetMapper().GetInput().GetPointData().SetScalars(numpy_support.numpy_to_vtk(scalars))

        actor.GetMapper().SetInterpolateScalarsBeforeMapping(interpolate_scalars)


    actor.GetProperty().SetOpacity(opacity)

    if representation == 'wireframe':

        actor.GetProperty().SetRepresentationToWireframe()

    return actor



def add_stl(filename,
             renderer=None,
             opacity=1.0,
             specular=0.1,
             ambient=0.0,
             scale=(1, 1, 1),
             translate=(0, 0, 0),
             rotate=(0, 0, 0),
             mesh_color="blue",
             use_wireframe=False,
             scale_then_translate=False,
             smoothing_passes=0,
             max_subdivisions=0
             ):
    """[Used to load in standard STL files and translate them]

    Arguments:
        filename {[string]} -- [the filename of the STL]

    Keyword Arguments:
        opacity {float} -- [opacity of the object] (default: {1.0})
        scale {tuple} -- [x,y,z scale] (default: {(1, 1, 1)})
        translate {tuple} -- [x,y,z shift] (default: {(0, 0, 0)})
        rotate {tuple} -- [rotation in degrees] (default: {(0, 0, 0)})
        mesh_color {str} -- [color of the mesh, taken from VTK colors] (default: {"blue"})
        use_wireframe {bool} -- [renders the wireframe instead] (default: {False})
    """

    colors = vtk.vtkNamedColors()

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    transform = vtk.vtkTransform()
    transform.RotateX(rotate[0])
    transform.RotateY(rotate[1])
    transform.RotateZ(rotate[2])

    if scale_then_translate:
        transform.Scale(scale)
        transform.Translate(translate)
    else:
        transform.Translate(translate)
        transform.Scale(scale)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    if max_subdivisions > 0:

        print('subdividing mesh, may take a while')

        subdivider = vtk.vtkAdaptiveSubdivisionFilter()
        subdivider.SetMaximumNumberOfPasses(max_subdivisions)
        subdivider.SetInputConnection(transformFilter.GetOutputPort())
        subdivider.Update()
        transformFilter = subdivider

    if smoothing_passes > 0:

        smooth_loop = vtk.vtkSmoothPolyDataFilter()
        smooth_loop.SetNumberOfIterations(smoothing_passes)
        smooth_loop.SetRelaxationFactor(0.1)
        smooth_loop.BoundarySmoothingOn()
        smooth_loop.SetInputConnection(transformFilter.GetOutputPort())
        smooth_loop.Update()
        mapper = vtk.vtkPolyDataMapper()

        mapper.SetInputConnection(smooth_loop.GetOutputPort())
    else:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if mesh_color is not None:
        if isinstance(mesh_color, str):
            actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
        else:
            actor.GetProperty().SetColor(mesh_color[0], mesh_color[1], mesh_color[2])
    actor.GetProperty().SetOpacity(opacity)

    actor.GetProperty().SetSpecular(specular)
    actor.GetProperty().SetSpecularPower(80.0)
    actor.GetProperty().SetAmbient(ambient)

    actor.GetProperty().SetInterpolationToGouraud()

    if use_wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    if renderer is None:
        return actor
    else:
        renderer.AddActor(actor)

def add_polygon(position=[0,0,0], vertices=None, polygon='hexagon', mesh_color='red', rotation_offset=0, opacity=1, verbose=True):


    if vertices is None:

        if verbose:
            print('No custom vertices supplied, drawing', polygon, 'with center:', position, 'rotation:', rotation_offset)

        polygon_dict = dict(triangle=3, square=4, pentagon=5, hexagon=6, heptagon=7, octagon=8, nonagon=9, decagon=10)

        if polygon is not None:

            segments = polygon_dict[polygon]


        polygon_angle = 360/segments

        vertices = np.zeros([segments, 3])
        
        for wedge in range(segments):

            vertex_angle = (wedge*polygon_angle) + rotation_offset

            x_pos = math.sin(math.radians(vertex_angle)) + position[0]
            y_pos = math.cos(math.radians(vertex_angle)) + position[1]
            z_pos = position[2]


            vertices[wedge, :] = [x_pos, y_pos, z_pos] 

    points = vtk.vtkPoints()

    for vertex in vertices:

        # print(vertex)

        points.InsertNextPoint(vertex)

    polygon = vtk.vtkPolygon()

    polygon.GetPointIds().SetNumberOfIds(vertices.shape[0])

    for vnum in range(vertices.shape[0]):
        polygon.GetPointIds().SetId(vnum, vnum)

    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(mesh_color))
    actor.GetProperty().SetOpacity(opacity)

    return actor

def add_lines(locations,
              weights=None,
              use_euclidean_as_weight=True,
              renderer=None,
              opacity=1.0,
              color=None,
              colormap='viridis',
              render_type='line',
              tube_radius=0.2,
              tube_resolution=6,
              line_width=1,
              verbose=False):
    """[Adds a series of lines to the renderer in a similar manner to add_arrows(),
        The lines are rendered in 2D and are much faster to compute]

    Arguments:
        [edge_data] {Numpy 2D array, (N,6)} -- [a 2D array containing the starts & ends of all lines]
        [renderer] {VTK renderer} -- the renderer used by VTK
    Returns:
        [renderer] {VTK renderer} -- the same renderer with lines added

    """

    if locations.shape[1] == 3:
        locations = np.reshape(np.asarray([(x,y) for x,y in zip(locations[:-1], locations[1:])]), [-1, 6])

    colors = vtk.vtkNamedColors()

    if verbose:
        print(locations.shape)
    linesPolyData = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    iterator = 0
    pts = vtk.vtkPoints()
    if locations.shape[1] == 6:
        if verbose:
            print('Reading rowwise')
        for row in tqdm(locations, desc='Adding Lines', disable=not verbose):

            p0 = row[0:3]
            p1 = row[3:6]

            # print(p0, p1)

            pts.InsertNextPoint(p0)
            pts.InsertNextPoint(p1)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, iterator)

            line.GetPointIds().SetId(1, iterator+1)

            iterator += 2

            lines.InsertNextCell(line)

        linesPolyData.SetLines(lines)

        linesPolyData.SetPoints(pts)

        line_mapper = vtk.vtkPolyDataMapper()

        if vtk.VTK_MAJOR_VERSION <= 5:
            line_mapper.SetInput(linesPolyData)
        else:
            line_mapper.SetInputData(linesPolyData)

        if ((weights is None) and (use_euclidean_as_weight is True)):
            if verbose:
                print('Using Euclidean distance as weights')
            weights = np.linalg.norm(locations[:, :3]-locations[:, 3:], axis=1)

        if weights is not None:
            scalar_color = vtk.vtkFloatArray()
            scalar_color.SetNumberOfComponents(1)
            scalar_color.SetName("SignedDistances")

            for pointId in range(lines.GetNumberOfCells()):
                scalar_color.InsertNextValue(weights[pointId])

            linesPolyData.GetCellData().SetScalars(scalar_color)

            line_mapper.SetScalarRange(np.min(weights), np.max(weights))

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetOpacity(opacity)
        line_actor.GetProperty().SetLineWidth(line_width)

        if color is not None:
            line_actor.GetProperty().SetColor(colors.GetColor3d(color))
        else:
            lut = make_LUT(colormap=colormap)
            line_mapper.SetLookupTable(lut)

        if render_type == 'tube':
            if verbose:
                print('Rendering as tubes, may take a while')
            tubefilter = vtk.vtkTubeFilter()
            tubefilter.SetInputData(linesPolyData)
            tubefilter.SetRadius(tube_radius)
            tubefilter.SetNumberOfSides(tube_resolution)
            tubemapper = vtk.vtkPolyDataMapper()
            tubemapper.SetInputConnection(tubefilter.GetOutputPort())

            if weights is not None:

                tubemapper.SetScalarRange(np.min(weights), np.max(weights))

            # else:
            if color is None:
                tubemapper.SetLookupTable(lut)

            line_actor = vtk.vtkActor()
            line_actor.SetMapper(tubemapper)
            line_actor.GetProperty().SetOpacity(opacity)
            
            if color is not None:
                line_actor.GetProperty().SetColor(colors.GetColor3d(color))
        if renderer is None:
            return line_actor
        else:
            renderer.AddActor(line_actor)

    else:
        if verbose:
            print(locations.shape)
            print('Reading row pairwise')
        for i in range(locations.shape[0] - 1):
            arrow_start, arrow_end = locations[i, :], locations[(i + 1), :]
            arrow = add_line(arrow_start, arrow_end, return_line=True)
            renderer.AddActor(arrow)

    return renderer


def add_line(startPoint, endPoint, return_line=False, linewidth=1, linecolor='Black'):
    """[Creates a singular line actor or the linesource]

    Arguments:
        startPoint {[Numpy array (3)]} -- [start point of line]
        endPoint {[Numpy array (3)]} -- [end point of line]

    Keyword Arguments:
        return_line {bool} -- [returns the line as a source, used if multiple lines are added
                                see add_lines() for implementation] (default: {True})

    Returns:
        [actor] {VTK Actor} -- [A VTK actor for the line]
    """

    if return_line:

        pts = vtk.vtkPoints()
        pts.InsertNextPoint(startPoint)
        pts.InsertNextPoint(endPoint)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)

        return (line, pts)

    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(startPoint)
    lineSource.SetPoint2(endPoint)
    lineSource.SetResolution(1)
    

    # Visualize
    colors = vtk.vtkNamedColors()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(lineSource.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetColor(colors.GetColor3d(linecolor))

    return actor

 

def add_labels(locations, scalar_vals, precision=1, renderer=None, label_color=[1, 1, 1], text_color=[0, 0, 0], opacity=0.5):

    points = vtk.vtkPoints()
    label = vtk.vtkStringArray()
    label.SetName('label')

    # print(type(scalar_vals))

    if scalar_vals.dtype == np.dtype('U24'):

        print('Labels appear to be text')

    for lnum, location in enumerate(locations):
        points.InsertNextPoint(location[0], location[1], location[2])
        try:
            label_val = round(scalar_vals[lnum], precision)
        except IndexError:
            label_val = 0
            # label_val = scalar_vals[lnum]
        if precision == 0:
            label_val = int(label_val)
        label.InsertNextValue(str(label_val))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(label)

    hier = vtk.vtkPointSetToLabelHierarchy()
    hier.SetInputData(polydata)
    hier.SetLabelArrayName('label')
    hier.GetTextProperty().SetColor(text_color)

    lmapper = vtk.vtkLabelPlacementMapper()
    lmapper.SetInputConnection(hier.GetOutputPort())

    lmapper.SetShapeToRoundedRect()
    lmapper.SetBackgroundColor(label_color)
    lmapper.SetBackgroundOpacity(opacity)
    lmapper.SetMargin(3)

    lactor = vtk.vtkActor2D()
    lactor.SetMapper(lmapper)

    if renderer is None:
        return lactor
    else:
        renderer.AddActor(lactor)




def add_axes(source_object=None,
              source_renderer=None,
              line_width=10,
              bounds=None,
              axes_type='cartesian',
              axes_units='mm',
              tick_location='outside',
              minor_ticks=False,
              axes_placement='outer',
              grid_placement='outer',
              flat_labels=True,
              sticky_axes=False,
              draw_grid_planes=False):
    """[Adds axes to an actor]

    Arguments:
        source_object {VTK Actor} -- [The actor you want the Axes to be bound to]
        source_renderer {VTK renderer} -- [the renderer used in VTK]

    Returns:
        Axes {VTK widget} -- [The axes specified]
    """

    if axes_type == 'cartesian':
        cubeAxesActor = vtk.vtkCubeAxesActor()

        if source_object is not None:
            cubeAxesActor.SetBounds(source_object.GetBounds())
        else:
            cubeAxesActor.SetBounds(bounds)

        
        if source_renderer is not None:
            cubeAxesActor.SetCamera(source_renderer.GetActiveCamera())

        cubeAxesActor.GetProperty().SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetXTitle('X-Axis')
        cubeAxesActor.SetXUnits(axes_units)
        cubeAxesActor.GetXAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(0).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(0).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetYTitle('Y-Axis')
        cubeAxesActor.SetYUnits(axes_units)
        cubeAxesActor.GetYAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetZTitle('Z-Axis')
        cubeAxesActor.SetZUnits(axes_units)
        cubeAxesActor.GetZAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.DrawXGridlinesOn()
        cubeAxesActor.DrawYGridlinesOn()
        cubeAxesActor.DrawZGridlinesOn()

        cubeAxesActor.SetUseBounds(True)

        cubeAxesActor.GetXAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetYAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetZAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)

        if vtk.VTK_MAJOR_VERSION > 5:
            if grid_placement == 'outer':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_FURTHEST)
            elif grid_placement == 'inner':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_CLOSEST)
            elif grid_placement == 'all':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_ALL)

        if minor_ticks:
            cubeAxesActor.XAxisMinorTickVisibilityOn()
            cubeAxesActor.YAxisMinorTickVisibilityOn()
            cubeAxesActor.ZAxisMinorTickVisibilityOn()

        else:
            cubeAxesActor.XAxisMinorTickVisibilityOff()
            cubeAxesActor.YAxisMinorTickVisibilityOff()
            cubeAxesActor.ZAxisMinorTickVisibilityOff()

        if tick_location == 'inside':
            cubeAxesActor.SetTickLocationToInside()
        elif tick_location == 'outside':
            cubeAxesActor.SetTickLocationToOutside()
        elif tick_location == 'both':
            cubeAxesActor.SetTickLocationToBoth()

        if axes_placement == 'outer':
            cubeAxesActor.SetFlyModeToOuterEdges()
            cubeAxesActor.SetTickLocationToOutside()
        elif axes_placement == 'inner':
            cubeAxesActor.SetFlyModeToClosestTriad()
        elif axes_placement == 'furthest':
            cubeAxesActor.SetFlyModeToFurthestTriad()
        elif axes_placement == 'all':
            cubeAxesActor.SetFlyModeToStaticEdges()

        # cubeAxesActor.SetUse2DMode(flat_labels)
        # cubeAxesActor.SetUseTextActor3D(True)
        cubeAxesActor.SetStickyAxes(sticky_axes)
        cubeAxesActor.SetCenterStickyAxes(False)

        cubeAxesActor.GetProperty().SetLineWidth(line_width)
        # cubeAxesActor.Update()
        cubeAxesActor.GetProperty().RenderLinesAsTubesOn()
        # cubeAxesActor.GetProperty().SetEdgeVisibility(True)
        cubeAxesActor.GetProperty().SetInterpolationToPhong()
        cubeAxesActor.GetProperty().VertexVisibilityOn()
        cubeAxesActor.SetUse2DMode(True)



        x_properties = cubeAxesActor.GetTitleTextProperty(0)

        # print(dir(x_properties))


        x_properties.BoldOn()
        x_properties.ItalicOn()
        x_properties.SetFontSize(20)
        x_properties.SetLineOffset(50)
        x_properties.SetVerticalJustificationToTop()

        # print(x_properties)

        y_properties = cubeAxesActor.GetTitleTextProperty(1)
        y_properties.BoldOn()
        y_properties.ItalicOn()
        y_properties.SetFontSize(20)
        y_properties.SetLineOffset(50)

        z_properties = cubeAxesActor.GetTitleTextProperty(2)
        z_properties.BoldOn()
        z_properties.ItalicOn()
        z_properties.SetFontSize(20)
        z_properties.SetLineOffset(50)
    

        if draw_grid_planes:
            cubeAxesActor.DrawXGridpolysOn()
            cubeAxesActor.DrawYGridpolysOn()
            cubeAxesActor.DrawZGridpolysOn()


        cubeAxesActor.SetUseTextActor3D(2)
        cubeAxesActor.SetUseTextActor3D(1)
        cubeAxesActor.SetUseTextActor3D(0)
        cubeAxesActor.SetUse2DMode(True)
        cubeAxesActor.Modified()





        return cubeAxesActor

    elif axes_type == 'polar':
        pole = [0, 0, 0]
        polaxes = vtk.vtkPolarAxesActor()
        polaxes.SetMinimumAngle(0.)
        polaxes.SetMaximumAngle(360.)
        polaxes.SetSmallestVisiblePolarAngle(1.)
        polaxes.SetUse2DMode(flat_labels)

        polaxes.SetAutoSubdividePolarAxis(True)
        if source_renderer is not None:
            polaxes.SetCamera(source_renderer.GetActiveCamera())
        polaxes.SetPolarLabelFormat("%6.1f")
        polaxes.GetSecondaryRadialAxesProperty().SetColor(1., 0., 0.)
        polaxes.GetSecondaryRadialAxesTextProperty().SetColor(0., 0., 1.)
        polaxes.GetPolarArcsProperty().SetColor(1., 0., 0.)
        polaxes.GetSecondaryPolarArcsProperty().SetColor(0., 0., 1.)
        polaxes.GetPolarAxisProperty().SetColor(0., 0, 0.)
        polaxes.GetPolarAxisTitleTextProperty().SetColor(0., 0., 0.)
        polaxes.GetPolarAxisLabelTextProperty().SetColor(0., 0., 0.)
        polaxes.SetEnableDistanceLOD(True)

        if minor_ticks:
            polaxes.SetAxisMinorTickVisibility(1)

        polaxes.SetScreenSize(6.)

        if source_object is not None:
            object_bounds = np.asarray(source_object.GetBounds())
            del_x = object_bounds[1] - object_bounds[0]
            del_y = object_bounds[3] - object_bounds[2]

            max_bound = max(del_x, del_y)
            polaxes.SetMaximumRadius(max_bound/2)

        else:
            polaxes.SetBounds(bounds)

        return polaxes



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

        # print(dir(scalar_bar))
        if orientation == 'Horizontal':
            scalar_bar.SetOrientationToHorizontal()
        else:
            scalar_bar.SetOrientationToVertical()
        # scalar_bar.SetWidth(width)
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


def generate_delaunay_space(reference_actor, opacity=0.5, mesh_color='Blue'):

    colors = vtk.vtkNamedColors()

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(reference_actor.GetMapper().GetInput())

    del_filter = vtk.vtkDelaunay3D()
    del_filter.SetInputConnection(clean_filter.GetOutputPort())
    del_mapper = vtk.vtkDataSetMapper()
    del_mapper.SetInputConnection(del_filter.GetOutputPort())
    
    del_actor = vtk.vtkActor()
    del_actor.SetMapper(del_mapper)
    del_actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
    del_actor.GetProperty().SetOpacity(opacity)
    del_actor.Modified()

    return del_actor
