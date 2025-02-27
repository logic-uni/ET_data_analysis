"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/10/2024
data from: Xinchao Chen
"""

import math

def calculate_azimuth(x1, y1, x2, y2):
    """
    Calculate the azimuth angle between two points in the XY-plane.
    """
    dx = x2 - x1
    dy = y2 - y1
    
    azimuth = math.atan2(dy, dx)
    azimuth = math.degrees(azimuth)
    if azimuth < 0:
        azimuth += 360  # Normalize to 0-360 degrees

    return azimuth

def calculate_elevation(x1, y1, z1, x2, y2, z2):
    """
    Calculate the elevation angle between two points in 3D space.
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    horizontal_distance = math.sqrt(dx**2 + dy**2)
    elevation_angle = math.atan2(dz, horizontal_distance)
    elevation_angle = math.degrees(elevation_angle)

    return elevation_angle


def calculate_yaw(x1, y1, x2, y2):
    """
    Calculate the yaw angle (azimuth) between two points in the XY-plane.
    """
    dx = x2 - x1
    dy = y2 - y1
    
    yaw = math.atan2(dy, dx)
    yaw = math.degrees(yaw)
    if yaw < 0:
        yaw += 360  # Normalize to 0-360 degrees

    return yaw

def calculate_pitch(x1, y1, z1, x2, y2, z2):
    """
    Calculate the pitch angle (elevation) between two points in 3D space.
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    pitch = math.asin(dz / distance)
    pitch = math.degrees(pitch)

    return pitch

def calculate_roll():
    """
    Roll is typically related to the orientation of an object and not 
    directly to the line between two points. Here we assume no roll.
    """
    return 0  # Assuming no roll for the purpose of two points

# Given coordinates

x1, y1, z1 = 1148,99,667 #AP,DV,ML
x2, y2, z2 = 995,724,672

def trans(x,y,z): #X, Y and Z axes up to the number of pixels in each dimension (ML, DV, AP respectily)
    #x = x - 540
    #y = y - 44
    #z = z - 570
    x = (x - 5700) / 1000
    y = y / 1000
    z = (5400 - z) / 1000
    #AP in CCF = –1 × AP coordinates × 1000 + 5400; DV in CCF = DV coordinates × 1000; ML in CCF = ML coordinates × 1000 + 5700.
    #X = (x * math.cos(0.0873) - y * math.sin(0.0873)) * 10
    #Y = (x * math.sin(0.0873) + y * math.cos(0.0873)) * 10
    #Z = (z * 0.9434) * 10
    print(f"ML: {x}")
    print(f"DV: {y}")
    print(f"AP: {z}")

trans(z1,y1,x1)

#x:ml,y:ap,z:dv
# Calculate azimuth and elevation
azimuth = calculate_azimuth(z1, x1, z2, x2)
elevation = calculate_elevation(z1, x1, y1, z2, x2, y2)

print(f"Azimuth: {azimuth} degrees")
print(f"Elevation: {elevation} degrees")

# Calculate azimuth and elevation
yaw = calculate_yaw(z1, x1, z2, x2)
pitch = calculate_pitch(z1, x1, y1, z2, x2, y2)

print(f"yaw: {yaw} degrees")
print(f"pitch: {pitch} degrees")