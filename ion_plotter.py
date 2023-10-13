import plot_utils
import vtk_utils

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ion_path = '/home/grtdaq/Downloads/config783/Calcium_pos.csv'
# ion_path = '/home/grtdaq/Downloads/config783/Krypton_pos.csv'
ion_path = '/home/grtdaq/Downloads/config783/Water-h2_pos.csv'



def get_ion_positions(ion_path):

    ion_data = pd.read_csv(ion_path, delimiter=',').values

    # print(ion_data.shape)

    ion_positions = ion_data[:,:3] 
    ion_velocities = ion_data[:,3:]

    # print(ion_positions.shape)

    # print(ion_positions)

    # print(np.max(ion_positions[:,0]) - np.min(ion_positions[:,0]))
    # print(np.max(ion_positions[:,1]) - np.min(ion_positions[:,1]))
    # print(np.max(ion_positions[:,2]) - np.min(ion_positions[:,2]))

    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.scatter(ion_positions[:,0], ion_positions[:,1])
    # plt.subplot(1,3,2)
    # plt.scatter(ion_positions[:,0], ion_positions[:,2])
    # plt.subplot(1,3,3)
    # plt.scatter(ion_positions[:,1], ion_positions[:,2])
    # plt.show()

    ion_positions = ion_positions * [1000,1000,1000]

    return ion_positions, ion_velocities

actor_dict = dict()

# calcium_positions, calcium_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Calcium_pos.csv')
# actor_dict.update({'<calcium': vtk_utils.add_polydata(calcium_positions, scalar_dict=calcium_velocities[:,1], glyph_scale=2*0.0118, glyph_type='sphere', colormap='viridis', opacity=1)})

# krypton_positions, krypton_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Krypton_pos.csv')
# actor_dict.update({'krypton': vtk_utils.add_polydata(krypton_positions, scalar_dict=krypton_velocities[:,0], glyph_scale=0.03, glyph_type='sphere', colormap='hot')})

# water_positions, water_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Water-h2_pos.csv')
# actor_dict.update({'water': vtk_utils.add_polydata(water_positions, scalar_dict=water_velocities[:,0], glyph_scale=0.03, glyph_type='sphere', colormap='Greys')})

calcium_positions, calcium_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Calcium_pos.csv')
actor_dict.update({'calcium': vtk_utils.add_polydata(calcium_positions, scalar_dict=calcium_velocities[:,0], glyph_scale=2*0.0118, glyph_type='sphere', colors='dark_blue', opacity=0.2)})

krypton_positions, krypton_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Krypton_pos.csv')
actor_dict.update({'krypton': vtk_utils.add_polydata(krypton_positions, scalar_dict=krypton_velocities[:,0], glyph_scale=2*0.0169, glyph_type='sphere', colors='dark_green')})

water_positions, water_velocities = get_ion_positions('/home/grtdaq/Downloads/config783/Water-h2_pos.csv')
actor_dict.update({'water': vtk_utils.add_polydata(water_positions, scalar_dict=water_velocities[:,0], glyph_scale=2*0.01, glyph_type='sphere', colors='dark_red')})



plot_utils.plot_vtk(actor_dict=actor_dict, glyph_scale=0.03, glyph_type='sphere', use_qt=True, colormap='viridis')
