# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt

def step_terrain(terrain,obstacle_height=10.,obstacle_width=2.):
    # heightfield=0*np.ones((terrain.width,terrain.length),dtype='int16')
    # heightfield[int(terrain.width/4):int(3*terrain.width/4),int(terrain.length/4):int(3*terrain.length/4)]=-1

    heightfield=np.zeros((terrain.width,terrain.length),dtype='int16')
    heightfield[int(terrain.width/2):,:]=obstacle_height
    # num_steps = 3
    # for i in range(num_steps-1):
    # num_steps=3
    # indx_wid=np.linspace(0,terrain.width/2,num_steps+1,dtype=int)
    # step_height=1
    # midp=int(terrain.width/2)
    # for i in range(num_steps-1):
    #     heightfield[indx_wid[i+1]:indx_wid[i+2],indx_wid[i+1]:indx_wid[i+2]]=step_height
    #     step_height+=1        
    # heightfield[int(terrain.width/4):int(3*terrain.width/4),int(terrain.width/4):int(3*terrain.width/4)]=-1
    # heightfield[:,int(terrain.width/4):int(3*terrain.width/4)]=-1

    heightfield=np.zeros((terrain.width,terrain.length),dtype='int16')
    heightfield[int(terrain.width/2):,:]=10
    # for i in range(0,400,50):
    #     heightfield[i:i+35,:]=-100

    return heightfield

def Test_terrain(terrain, dune_height=12, dune_length_ratio=1,num_dunes=2, max_height=0.5, min_size=0.5, max_size=3, num_rects=1000, platform_size=1.):
    heightfield = np.zeros((terrain.width, terrain.length), dtype='int16')
    def dune(d_height, length):
        heightfield = np.zeros((int(terrain.width/4), int(length)), dtype='int16')

        # Parameters for dune
        dune_length = int(dune_length_ratio * length)
        dune_width = int(terrain.width/4)

        # Adjust the frequency of the dunes
        dune_frequency_length = num_dunes * 2 * np.pi / dune_width
        dune_frequency_width = num_dunes * 2 * np.pi / dune_width

        # Create the sand dune with a smooth curve
        
        for x in range(dune_width):
            for y in range(dune_length):
                # Use a sinusoidal function to create a smooth dune shape in both directions
                height = d_height * np.sin(dune_frequency_length * y) * np.sin(dune_frequency_width * x)
                heightfield[x, y] = int(np.round(height))
        return heightfield
    def random_objects(terrain, max_height=0.5, min_size=0.5, max_size=3, num_rects=100, platform_size=1.):
        max_height = int(max_height / terrain.vertical_scale)
        min_size = int(min_size / terrain.horizontal_scale)
        max_size = int(max_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        terrain.height_field_raw = np.zeros((100,100), dtype='int16')
        (i, j) = terrain.height_field_raw.shape
        height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)

        for _ in range(num_rects):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain.height_field_raw
    # modified_terrain = 
    heightfield[:int(terrain.width/4),:] = dune(3*dune_height*0.03/terrain.vertical_scale, int(terrain.length))
    heightfield[int(terrain.width/4):int(2*terrain.width/4),:] = 0.3/terrain.vertical_scale
    heightfield[int(2*terrain.width/4):int(3*terrain.width/4),:] = random_objects(terrain)
    # heightfield[int(2*terrain.width/4):int(3*terrain.width/4),:] = dune(3*dune_height*0.03/terrain.vertical_scale, int(terrain.length))
    heightfield[int(3*terrain.width/4):int(4*terrain.width/4),:] = 2*0.3/terrain.vertical_scale
    return heightfield
    



def sand_dune_terrain(terrain, dune_height=12, dune_length_ratio=1, obstacle_height=20., obstacle_width_ratio=0.1, num_dunes=2):
    
    heightfield = np.zeros((terrain.width, terrain.length), dtype='int16')
    def dune(d_height, length):
        heightfield = np.zeros((terrain.width, int(length)), dtype='int16')

        # Parameters for dune
        dune_length = int(dune_length_ratio * length)
        dune_width = int(terrain.width/2)

        # Adjust the frequency of the dunes
        dune_frequency_length = num_dunes * 2 * np.pi / dune_width
        dune_frequency_width = num_dunes * 2 * np.pi / dune_width

        # Create the sand dune with a smooth curve
        
        for x in range(dune_width):
            for y in range(dune_length):
                # Use a sinusoidal function to create a smooth dune shape in both directions
                height = d_height * np.sin(dune_frequency_length * y) * np.sin(dune_frequency_width * x)
                heightfield[x, y] = int(np.round(height))
        return heightfield


    # Creating sand dunes of varying height
                

    heightfield[:,int(2*terrain.length/5):int(3*terrain.length/5)] = dune(dune_height*0.03/terrain.vertical_scale, int(terrain.length/5))
    heightfield[:,int(3*terrain.length/5):int(4*terrain.length/5)] = dune(dune_height*0.03/terrain.vertical_scale, int(terrain.length/5))
    heightfield[:,int(4*terrain.length/5):int(5*terrain.length/5)] = dune(2*dune_height*0.03/terrain.vertical_scale, int(terrain.length/5))
    # heightfield[:,int(4*terrain.length/5):int(terrain.length)] = dune(dune_height, terrain.length/5)
    # Create the ledge
    # heightfield[int(terrain.width/2):,:int(terrain.length/5)]=5
    heightfield[int(terrain.width/2):,:int(terrain.length/5)]=0.5*0.3/terrain.vertical_scale
    heightfield[int(terrain.width/2):,int(terrain.length/5):int(2*terrain.length/5)]=0.3/terrain.vertical_scale
    heightfield[int(terrain.width/2):,int(2*terrain.length/5):int(3*terrain.length/5)]=0.3/terrain.vertical_scale
    heightfield[int(terrain.width/2):,int(3*terrain.length/5):int(4*terrain.length/5)]=2*0.3/terrain.vertical_scale
    heightfield[int(terrain.width/2):,int(4*terrain.length/5):int(5*terrain.length/5)]=0.3/terrain.vertical_scale
    return heightfield

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain


def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, max_height=0.5, min_size=0.5, max_size=3, num_rects=1000, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain.height_field_raw


def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    Generate a wavy terrain

    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    Returns:
        terrain (SubTerrain): update terrain
    """
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain


def stairs_terrain(terrain, step_width=1, step_height=0.5):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = ((terrain.width - 20) // step_width)//2
    height = step_height
    terrain.height_field_raw[:,:] = 1
    terrain.height_field_raw[:int(terrain.width/2)-12,:] = height*num_steps
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] -= height
        height += step_height
    return terrain.height_field_raw


def pyramid_stairs_terrain(terrain, step_width=0.5, step_height=0.5, platform_size=2.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain.height_field_raw


def stepping_stones_terrain(terrain, stone_size=10, stone_distance=5, max_height=10, platform_size=1., depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
