
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm, trange
from scipy.spatial import Voronoi, voronoi_plot_2d

import vtk
from vtk.util import numpy_support
from matplotlib import colors as mcolors
# from scipy.stats import norm as snorm
try:
    import qt_utils
except ImportError:
    print('Could not import qt properly')
import vtk_utils

try:
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
except:
    print('Could not Latex-ify labels ')

# sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super

print('Current backend', matplotlib.pyplot.get_backend())

conflicting_backends = ['QtAgg', 'Qt5Agg']

if matplotlib.pyplot.get_backend() in conflicting_backends:
    try:
        matplotlib.use('GTK3Agg')
        print('backend set to GTK3Agg to avoid conflict with VTK plot')
    except:
        print('Tried to switch to GTK3, failed')
        print('Currently using:', matplotlib.get_backend())




def plot_edge_matrix(edge_matrix):

    plt.figure(figsize=(10,10))
    plt.scatter(edge_matrix[:,0], edge_matrix[:,1])
    plt.xlabel('Parent Node ID')
    plt.ylabel('Child Node ID')
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()

def plot_level_frequency(levels):

    plt.figure()
    # plt.hist(levels)
    levels_x, counts = np.unique(levels, return_counts=True)
    plt.bar(levels_x, height=counts)
    plt.xlabel('Node Level')
    plt.ylabel('Frequency')
    plt.show()

def plot_time_complexity(dataset_sizes, execution_times):


    # execution_times_tree = execution_times_tree

    plt.figure()
    plt.grid()
    plt.scatter(dataset_sizes, execution_times, label='Exhaustive Search')

    # time_matrix = list()
    # for i in range(2, 31):
    #     filename = 'times_taken_tree_numba-%i.npy' % i
    #     times = np.load(filename)
    #     time_matrix.append(times)
    # time_matrix = np.asarray(time_matrix)
    # size_matrix = np.tile(dataset_sizes, [time_matrix.shape[0],1])

    # data_matrix = np.dstack((size_matrix, time_matrix))


    # print(time_matrix.shape, size_matrix.shape, data_matrix.shape)

    # color_range = np.linspace(0, 1, time_matrix.shape[0])
    
    # line_colors = [cm.jet(c) for c in color_range]

    
    # plt.scatter(dataset_sizes, times, label='Node size %i' % i)



    # line_segments = LineCollection(segments=data_matrix, colors=line_colors)

    # plt.gca().add_collection(line_segments)

    # for t, time in enumerate(time_matrix):
    #     plt.scatter(dataset_sizes, time, label='Node size = %i' %(t+2), color=line_colors[t])

    # plt.scatter(dataset_sizes, execution_times_tree, label='MTree-Accelerated Search')
    plt.xlabel('Dataset Size')
    plt.ylabel('Execution time (s)')

    # x0 = dataset_sizes[0]
    # t0e = execution_times_exhaustive[0]
    # xn = dataset_sizes / x0
    # xs = np.power(xn, 3) * 0.63

    # plt.plot(dataset_sizes, xs, label='$O x^3$ Time Complexity Fit', linestyle='dashed')



    # x0 = dataset_sizes[0]
    # t0t = execution_times_tree[0]
    # xn = dataset_sizes / x0
    # xt = np.power(xn, 2) * t0t * 0.16

    # plt.plot(dataset_sizes, xt, label='$O x^2$ Time Complexity Fit', linestyle='dashed')


    plt.legend()

    plt.show()


def plot_2D_map(vertex_dict, vertex_level=0, reference_data=None, query_data=None, query_radius=None, edges=None, colors=None, savename=None, return_figure=False, figure=None, use_legend=True, convex_hull=False):


    if figure is not None:
        fig = figure
    else:
        fig = plt.figure(figsize=(10,10))
    ax = plt.gca()



    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    if reference_data is not None:
        if reference_data.shape[0] > 2:
            vor = Voronoi(reference_data[:,:2])
            fig = voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k', line_width=1, line_alpha=0.6, point_size=0.5)

    
    if query_data is not None:
        if np.ndim(query_data) == 2:
            plt.scatter(query_data[:,0], query_data[:,1], c='r', zorder=0, label='Query Point', marker='x')

        elif np.ndim(query_data) == 1:
            if query_radius is not None:
                circle1 = plt.Circle((query_data[0], query_data[1]), query_radius, color='lime', fill=True, alpha=0.2, label='Query Point')
                plt.gca().add_patch(circle1)
            plt.scatter(query_data[0], query_data[1], c='r', zorder=0, label='Query Point', marker='x')

 

    if colors is not None:
        circle_color = colors[vertex_level % len(colors)]
    else:
        circle_color = 'k'
        colors = ['k']

    scatter_points = list()

    for vertex_ID in tqdm(vertex_dict.keys()):

        vertex_data = vertex_dict[vertex_ID]

        # print(vertex_data)

        if vertex_data[1] is None:
            print('centroid is None, presumably this is the old MTree format?')
            centroid = [0, 0]
            radius = 200
        else:
            # return fig

            centroid = vertex_data[1]
            radius = vertex_data[2]

        scatter_points.append(centroid[:2])

        circle1 = plt.Circle((centroid[0], centroid[1]), radius, color=circle_color, fill=False, alpha=0.8, linestyle='dashed')
        plt.gca().add_patch(circle1)

    if edges is not None:
        for edge in edges:
            # print(edge)
            if edge[0] is None:
                edge[0] = [0,0]
            if edge[1] is None:
                continue
 
            movement_vector = edge[1] - edge[0]

            plt.arrow(edge[0][0], edge[0][1], movement_vector[0], movement_vector[1], zorder=10, head_width=0.5, length_includes_head=True, color=colors[vertex_level % len(colors)])

            # plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=colors[vertex_level % len(colors)])

    scatter_points = np.asarray(scatter_points)
    plt.scatter(scatter_points[:,0], scatter_points[:,1], color=colors[vertex_level % len(colors)], label='Level %i' % vertex_level)

    if vertex_level is not None:
        plt.title('Node depth %i' % vertex_level)

    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.legend()


    if savename is not None:
        plt.savefig(savename, dpi=600)
        plt.close()
    elif return_figure:
        fig = plt.gcf()
        return fig
    else:
        plt.show()


def plot_2D_map_flattened(vertex_dict, reference_data=None, children=None, save_name=None, colors=None):

    levelled_vertices = dict()

    if children is not None:
        levelled_edges = dict()


    for dict_key in tqdm(vertex_dict.keys()):

        vertex_data = vertex_dict[dict_key]

        vertex_level = vertex_data[3]

        if children is not None:

            if vertex_level not in levelled_edges.keys():
                sublist = list()
                levelled_edges.update({vertex_level:sublist})

            edge_data = children.get(dict_key)

            # print(edge_data.shape)

            
            
            if edge_data is not None:

                origin = vertex_data[1]

                for edge in edge_data:
                    # print(edge)
                    terminus = vertex_dict.get(edge[0])[1]
                    # print(origin, terminus)
                    levelled_edges[vertex_level].append([origin, terminus])


        if vertex_level not in levelled_vertices.keys():


            print('Adding %i level' % vertex_level)

            subdict = dict()
            subdict.update({dict_key:vertex_data})
            levelled_vertices.update({vertex_level:subdict})
        
        else:

            levelled_vertices[vertex_level].update({dict_key:vertex_data})


    if reference_data is not None:
        if reference_data.shape[0] > 2:
            vor = Voronoi(reference_data[:,:2])
            flat_figure = voronoi_plot_2d(vor, show_vertices=False, line_colors='k', line_width=1, line_alpha=0.6, point_size=0.5)

    else:

        flat_figure = plt.figure()

    for vertex_level in levelled_vertices.keys():

        subdict = levelled_vertices[vertex_level]

        if children is not None:
            edge_data = levelled_edges[vertex_level]

        flat_figure = plot_2D_map(subdict, vertex_level=vertex_level, reference_data=None, edges=edge_data, figure=flat_figure, return_figure=True, colors=colors)

    plt.show()


def plot_2D_map_levelled(vertex_dict, reference_data=None, save_name=None, colors=None):

    levelled_vertices = dict()


    for dict_key in tqdm(vertex_dict.keys()):

        vertex_data = vertex_dict[dict_key]

        vertex_level = vertex_data[3]

        if vertex_level not in levelled_vertices.keys():


            print('Adding %i level' % vertex_level)

            subdict = dict()
            subdict.update({dict_key:vertex_data})
            levelled_vertices.update({vertex_level:subdict})
        
        else:

            levelled_vertices[vertex_level].update({dict_key:vertex_data})


    for vertex_level in levelled_vertices.keys():

        subdict = levelled_vertices[vertex_level]

        plot_2D_map(subdict, vertex_level=vertex_level, reference_data=reference_data)


def plot_dataset_variance(dataset):

    np.seterr(under='ignore')

    print(dataset.shape)

        # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0,0,1,1])

    # Create the boxplot

    violin_axes = list()

    for dim in tqdm(range(50)):
    # for dim in tqdm(range(dataset.shape[1])):
        dim_data = dataset[:,dim].astype(float)
        violin_axes.append(dim_data)

    bp = ax.violinplot(violin_axes)

    plt.grid()

    # print(bp)

    plt.show()



def plot_vtk(locations=None, scalars=None, actor_dict=None, scalar_name='Values', title=None, glyph_scale=1, glyph_type='cube', colormap='viridis', show_scalar_bar=True, show_axes=True, use_qt=False):

    if actor_dict is None:
        actor_dict = dict()

    if locations is not None:
        if locations.shape[1] == 2:
            third = np.zeros([locations.shape[0],1])
            locations = np.hstack([locations, third])

        if scalars is None:
            scalars = np.ones(locations.shape[0])


        if type(scalars) is not dict:
            scalar_dict =  {scalar_name:scalars}
        else:
            scalar_dict = scalars

        if show_scalar_bar:
            if show_axes:
                actor_dict.update({'<[polydata': vtk_utils.add_polydata(locations=locations, scalar_dict=scalar_dict, glyph_scale=glyph_scale, glyph_type=glyph_type, colormap=colormap)})
            else:
                actor_dict.update({'[polydata': vtk_utils.add_polydata(locations=locations, scalar_dict=scalar_dict, glyph_scale=glyph_scale, glyph_type=glyph_type, colormap=colormap)})
    
        elif show_axes:
            actor_dict.update({'<polydata': vtk_utils.add_polydata(locations=locations, scalar_dict=scalar_dict, glyph_scale=glyph_scale, glyph_type=glyph_type, colormap=colormap)})


        if use_qt:
            qt_utils.render_data(actor_dict=actor_dict)
        else:
            create_renderer(actors=actor_dict)

    else:

        if use_qt:
            qt_utils.render_data(actor_dict=actor_dict)
        else:
            create_renderer(actors=actor_dict)




def create_renderer(actors=None,
                    title='field',
                    background_color='White',
                    add_skybox=False,
                    window_size=(600, 600),
                    display=True):

    renderer = vtk.vtkRenderer()

    # try:

    #     cube_path = 'data/cubemap/lab'

    #     cubemap = ReadCubeMap(cube_path, '/', '.png', 2)

    #     if add_skybox:
    #         skybox = ReadCubeMap(cube_path, '/', '.png', 2)
    #         # skybox = ReadCubeMap(cube_path, '/skybox', '.jpg', 2)
    #         skybox.InterpolateOn()
    #         skybox.RepeatOff()
    #         skybox.EdgeClampOn()

    #         skyboxActor = vtk.vtkSkybox()
    #         skyboxActor.SetTexture(skybox)
    #         renderer.AddActor(skyboxActor)

    #     renderer.UseImageBasedLightingOn()
    #     renderer.SetEnvironmentTexture(cubemap)

    # except:
    #     print('raytracing failed to initialise, are you on VTK 9.0?')

    render_camera = renderer.GetActiveCamera()

    renderWindow = vtk.vtkRenderWindow()

    renderWindow.SetSize(window_size)
    renderWindow.AddRenderer(renderer)
    renderWindow.PolygonSmoothingOn()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.SetInteractorStyle(vtk_utils.MyInteractorStyle(renderWindowInteractor, render_camera, renderWindow))
    renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(background_color))

    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(10)

    # print(dir(renderer))
    # renderer.SetStereoTypeToSplitViewportHorizontal()
    scalarbar_dict = dict()

    if actors is not None:
        if isinstance(actors, list):
            print('List of actors supplied, adding list')
            for actor in actors:
                renderer.AddActor(actor)
        elif isinstance(actors, dict):
            print('List of actors supplied, adding list')
            for key_name in actors:

                if key_name[0] == '<':
                    axes_actor = vtk_utils.add_axes(actors[key_name], renderer, axes_type='cartesian', axes_placement='outer')
                    axes_key = '%s-axes' % (key_name)
                    renderer.AddActor(axes_actor)
                if key_name[0] == '(':
                    axes_actor = vtk_utils.add_axes(actors[key_name], renderer, axes_type='polar', axes_placement='outer')
                    axes_key = '%s-axes' % (key_name)
                    renderer.AddActor(axes_actor)
                if key_name[0] == '[':
                    try:
                        bar_actor = vtk_utils.add_colorbar(actors[key_name], title=key_name, return_widget=False, interactor=renderWindowInteractor)
                        bar_name = 'colorbar_' + key_name
                        scalarbar_dict.update({bar_name: bar_actor})
                    except AttributeError:
                        pass

                if key_name[1] == '[':
                    print('adding scalar bar')
                    bar_actor = vtk_utils.add_colorbar(actors[key_name], title=key_name, return_widget=True, interactor=renderWindowInteractor)
                    bar_name = 'colorbar_' + key_name
                    scalarbar_dict.update({bar_name: bar_actor})
                    # renderer.AddActor(bar_actor)

                if isinstance(actors[key_name], list):
                    for sub_actor in actors[key_name]:
                        renderer.AddActor(sub_actor)
                elif type(actors[key_name]) is list:
                    for sub_actor in actors[key_name]:
                        renderer.AddActor(sub_actor)
                elif isinstance(actors[key_name], dict):
                    for sub_actor in actors[key_name].values():
                        renderer.AddActor(sub_actor)
                elif isinstance(actors[key_name], tuple):
                    for sub_actor in actors[key_name]:
                        renderer.AddActor(sub_actor)
                else:
                    renderer.AddActor(actors[key_name])

    else:
        renderer.AddActor(actors)

    renderer.ResetCamera()

    # exporter = vtk.vtkJSONRenderWindowExporter()
    # exporter.GetArchiver().SetArchiveName('/home/fraser/Videos/LiverView/vtk_only_2.js')
    # exporter.SetRenderWindow(renderWindow)
    # exporter.Write()
    # exporter.Update()

    # add_camera_widget(renderer, renderWindowInteractor)

    if display:
        renderWindow.Render()
        renderWindowInteractor.Start()
        return renderer, renderWindow, renderWindowInteractor
    else:
        return renderer, renderWindow, renderWindowInteractor


def test_3D_plot(ref_points=None, query_point=None, optimum=None, r_points=None):

    actor_dict = dict()

    sphere_res = 24
    sphere_scale = 0.01

    if ref_points is not None:

        ref_vtk = vtk_utils.add_polydata(ref_points, glyph_scale=sphere_scale, glyph_type='sphere', resolution=sphere_res, colors='maroon')

        actor_dict.update({'<Reference_points': ref_vtk})


    if query_point is not None:
        query_vtk = vtk_utils.add_polydata(query_point, glyph_scale=sphere_scale, glyph_type='sphere', resolution=sphere_res, colors='purple')
        actor_dict.update({'query_point': query_vtk})

        startpoints = np.tile(query_point,(3,1))

        print(startpoints)

        distances_vtk = vtk_utils.add_lines(np.hstack((ref_points, startpoints)), render_type='tube', tube_radius=0.002)
        actor_dict.update({'distances_point': distances_vtk})


    if optimum is not None:
        optimum_vtk = vtk_utils.add_polydata(optimum, glyph_scale=sphere_scale, glyph_type='sphere', resolution=sphere_res, colors='purple')
        actor_dict.update({'optimum_point': optimum_vtk})
        if query_point is not None:
            vector_vtk = vtk_utils.add_line(startPoint=query_point, endPoint=optimum, linewidth=2)
            actor_dict.update({'vector': vector_vtk})

        tri_1 = vtk_utils.add_polygon(vertices=np.asarray([ref_points[0], ref_points[1], optimum]))
        actor_dict.update({'tri_1': tri_1})
        tri_2 = vtk_utils.add_polygon(vertices=np.asarray([ref_points[0], ref_points[2], optimum]), mesh_color='Green')
        actor_dict.update({'tri_2': tri_2})
        tri_3 = vtk_utils.add_polygon(vertices=np.asarray([ref_points[1], ref_points[2], optimum]), mesh_color='Blue')
        actor_dict.update({'tri_3': tri_3})
    else:
        if ref_points is not None:
            del_actor = vtk_utils.generate_delaunay_space(ref_vtk, opacity=0.5)
            actor_dict.update({'del space': del_actor})

    if r_points is not None:
        r_vtk = vtk_utils.add_polydata(r_points, glyph_scale=sphere_scale, glyph_type='sphere', resolution=sphere_res, colors='purple')
        actor_dict.update({'r_points': r_vtk})

    # tri_points = np.zeros([3,3])
    # tri_points[0,:] = [1,0,0]
    # tri_points[1,:] = [0,1,0]
    # tri_points[2,:] = [0,0,1]

    # tri = add_polygon(vertices=tri_points, mesh_color='blue', opacity=0.2)

    # actor_dict.update({'111': tri})

    # outline = vtk_utils.add_outline(corner_vtk)

    # actor_dict.update({'outine': outline})

    plot_vtk(actor_dict=actor_dict, use_qt=True)



def plot_3D_scatter(points, glyph_scale=0.5, scalars=None, colors=None, labels=None):

    point_actor = vtk_utils.add_polydata(points, scalar_dict=scalars, colors=colors, glyph_type='sphere', resolution=12, glyph_scale=glyph_scale)
    # axes_actor = add_axes(point_actor)

    actor_dict = dict()
    actor_dict.update({'<locs': point_actor})
    # actor_dict.update({'axes': axes_actor})



    if labels is not None:

        label_actor = vtk_utils.add_labels(points, labels)
        actor_dict.update({'labels': label_actor})

    plot_vtk(actor_dict=actor_dict)

