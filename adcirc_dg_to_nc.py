import numpy as np
import pandas as pd
from netCDF4 import Dataset
import time

# ==========================================
# 1. Configuration
# ==========================================
fort14_file = 'fort.14'
input_nc = 'adcirc_dg_out.nc'
output_nc = 'DG.63.nc'

# Specify the start time (Format: 'YYYY-MM-DD HH:MM:SS')
START_TIME = '2017-08-25 12:00:00'

# ==========================================
# 2. Parse fort.14 Mesh
# ==========================================
print(f"Reading {fort14_file}...")
start_time = time.time()

with open(fort14_file, 'r') as f:
    grid_name = f.readline().strip()
    ne, np_nodes = map(int, f.readline().split())

# Read Nodes: [node_id, x, y, depth]
nodes = pd.read_csv(fort14_file, skiprows=2, nrows=np_nodes, 
                    sep=r'\s+', header=None).values

# Read Elements: [element_id, 3, n1, n2, n3]
elements = pd.read_csv(fort14_file, skiprows=2+np_nodes, nrows=ne, 
                        sep=r'\s+', header=None).values

print(f"Loaded {np_nodes} nodes and {ne} elements in {time.time() - start_time:.2f} seconds.")

# Extract 1-based element connectivity directly for full mesh
connectivity = elements[:, 2:5].astype(int)

# ==========================================
# 3. Create Output NetCDF and Write Full Mesh Data
# ==========================================
print(f"Writing to {output_nc}...")

ds_in = Dataset(input_nc, 'r')
ds_out = Dataset(output_nc, 'w', format='NETCDF4')

# Global Attributes
ds_out.rundes = 'MeshVer1_Roads'
ds_out.runid = 'Full_Mesh_Run'

# Dimensions
ds_out.createDimension('time', None)
ds_out.createDimension('node', np_nodes)
ds_out.createDimension('nele', ne)
ds_out.createDimension('nfaces', ne)
ds_out.createDimension('nvertex', 3)
ds_out.createDimension('dof', 1)

# Variables: X
var_x = ds_out.createVariable('x', 'f8', ('node',))
var_x.long_name = 'longitude'
var_x.standard_name = 'longitude'
var_x.units = 'degrees_east'
var_x.positive = 'east'
var_x[:] = nodes[:, 1]

# Variables: Y
var_y = ds_out.createVariable('y', 'f8', ('node',))
var_y.long_name = 'latitude'
var_y.standard_name = 'latitude'
var_y.units = 'degrees_north'
var_y.positive = 'north'
var_y[:] = nodes[:, 2]

# Variables: Depth
var_depth = ds_out.createVariable('depth', 'f8', ('node',))
var_depth.long_name = 'distance below geoid'
var_depth.standard_name = 'depth below geoid'
var_depth.coordinates = 'time y x'
var_depth.location = 'node'
var_depth.mesh = 'adcirc_mesh'
var_depth.units = 'm'
var_depth[:] = nodes[:, 3]

# Variables: Element Connectivity (Python order: nfaces, nvertex)
var_elem = ds_out.createVariable('element', 'i8', ('nfaces', 'nvertex'))
var_elem.long_name = 'element'
var_elem.cf_role = 'face_node_connectivity'
var_elem.start_index = 1
var_elem.units = 'nondimensional'
var_elem[:] = connectivity

# Variables: Mesh Topology
var_mesh = ds_out.createVariable('adcirc_mesh', 'i8', ())
var_mesh.cf_role = 'mesh_topology'
var_mesh.long_name = 'mesh_topology'
var_mesh.topology_dimension = 2
var_mesh.node_coordinates = 'x y'
var_mesh.face_node_connectivity = 'element'

# Variables: Time
in_time = ds_in.variables['time']
var_time = ds_out.createVariable('time', 'f8', ('time',))
var_time.long_name = 'model time'
var_time.standard_name = 'time'
var_time.units = f'seconds since {START_TIME}'
var_time.calendar = 'standard'
var_time[:] = in_time[:]

# Variables: Zeta (Python order: time, nfaces, dof)
var_zeta = ds_out.createVariable('zeta', 'f8', ('time', 'nfaces', 'dof'), fill_value=-99999.0)
var_zeta.long_name = 'water surface elevation above geoid'
var_zeta.standard_name = 'sea_surface_height_above_geoid'
var_zeta.coordinates = 'time y x'
var_zeta.location = 'face'
var_zeta.mesh = 'adcirc_mesh'
var_zeta.units = 'm'

num_times = len(in_time[:])
print(f"Extracting Zeta for {num_times} timesteps (Iterative to save memory)...")

# Write zeta timestep by timestep for all elements
for t in range(num_times):
    var_zeta[t, :, 0] = ds_in.variables['ze'][t, :]
    
    if (t + 1) % 50 == 0:
        print(f"  Processed {t + 1}/{num_times} timesteps...")

# Close files
ds_in.close()
ds_out.close()

print(f"Success! Full mesh netCDF created: {output_nc}")
