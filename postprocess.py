# generate two output datasets 
# 1: whole earth SLP for MCMS input
# 2: storm patches for direct to evals 
# codicast outputs are south-to-north. Maintain that for (2) but flip for (1) 

# expected output MCMS lat/lon ranges:
#   * lat         (lat) float64 87.19 81.56 75.94 70.31 ... -75.94 -81.56 -87.19
#   * lon         (lon) float64 2.812 8.438 14.06 19.69 ... 345.9 351.6 357.2

import numpy as np 
import xarray as xr 
import pandas as pd 
import os
from pyproj import Proj 
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

inpath = '/mnt/data/sonia/codicast-out/date/multivar/raw-out3/test'
input_topo = '/home/cyclone/topo.nc'
outpath_mcms = '/mnt/data/sonia/codicast-out/date/multivar/mcms-in/test3'
outpath_patches = '/mnt/data/sonia/codicast-out/date/multivar/patches/test3'

timesteps = 8

############################ load storm df #############################
start_year = 2015
trackspath1='/mnt/data/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/mnt/data/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 # overlap for the track data
tracks1 = pd.read_csv(trackspath1, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# storms that start before the join year (even if they continue into the join year):
sids1 = tracks1[(tracks1['sid']==tracks1['tid']) & (tracks1['year']<joinyear)]['sid'].unique()
tracks1 = tracks1[tracks1['sid'].isin(sids1)]
tracks2 = pd.read_csv(trackspath2, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# filter out storms that "start" at the beginning of the join year since they probably started before and are 
# included in tracks1
sids2 = tracks2[(tracks2['sid']==tracks2['tid']) & \
        ((tracks2['year']>=joinyear) | (tracks2['month']>1) | (tracks2['day']>1) | (tracks2['hour']>0))]['sid'].unique()
tracks2 = tracks2[tracks2['sid'].isin(sids2)]
tracks = pd.concat([tracks1, tracks2], ignore_index=True)
tracks = tracks[tracks['year']>=start_year]
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])

# conversions from the MCMS lat/lon system, as described in Jimmy's email:
tracks['lat'] = 90-tracks['unk1'].values/100
tracks['lon'] = tracks['unk2'].values/100

tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]

sids = sorted(os.listdir('/home/cyclone/train/multivar/0.25/date/natlantic/test'))
if len(sids) > len(os.listdir(inpath)):
    print('warning, more sids than input files')
    sids = sids[:len(os.listdir(inpath))]

############################ make MCMS and patch outputs ############################
lats = np.linspace(-90 + (5.625/2), 90 - (5.625/2), 32)
lons = np.linspace(0 + (5.625/2), 360 - (5.625/2), 64)
resolution = 5.625
l = 800 # (half length: l/2 km from center in each direction)
s = 32 # box will be dimensions s by s (eg 32x32)
x_lin = np.linspace(-l, l, s) 
y_lin = np.linspace(-l, l, s)
x_grid, y_grid = np.meshgrid(x_lin, y_lin) # equal-spaced points from -l to l in both x and y dimensions

for sid, sid_in_id in tqdm(zip(sids, sorted(os.listdir(inpath))), total=len(sids)):
        datas = [np.load(os.path.join(inpath, sid_in_id, f'{t}.npy')) for t in range(timesteps)]
        data = np.stack(datas, axis=0)

        start_row = tracks[tracks['tid']==sid].to_dict(orient='records')[0]
        start_date = pd.to_datetime(
        f"{start_row['year']}-{start_row['month']}-{start_row['day']} {start_row['hour']:02d}:00:00")
        time_dim = np.array([start_date + pd.Timedelta(hours=6*t) for t in range(timesteps)])

        ds = xr.Dataset(
        data_vars={
                "slp": (["time", "lat", "lon"], data[:,:,:,0]),
                "u": (["time", "lat", "lon"], data[:,:,:,1]),
                "v": (["time", "lat", "lon"], data[:,:,:,2]),
                "t": (["time", "lat", "lon"], data[:,:,:,3]),
                "q": (["time", "lat", "lon"], data[:,:,:,4]),
        },
        coords={
                "time": time_dim,
                "lat": (["lat"], lats, {"units": "degrees_north", "standard_name": "latitude"}),
                "lon": (["lon"], lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
                "description": f'{sid}\n{start_row["lat"]}\n{start_row["lon"]}',
        }
        )
        # flip on lat for MCMS 
        ds_flip = ds.reindex(lat=ds.lat[::-1]) # lat now 90 to -90

        os.makedirs(os.path.join(outpath_mcms, sid), exist_ok=True)

        if time_dim[0].year == time_dim[-1].year:
                ds_flip['slp'].to_netcdf(os.path.join(outpath_mcms, sid, f'slp.{start_row["year"]}.nc'))
        else: # split into two years to make mcms happy
                ds_year1 = ds_flip.sel(time=slice(f'{start_row["year"]}-01-01', f'{start_row["year"]}-12-31'))
                ds_year1['slp'].to_netcdf(os.path.join(outpath_mcms, sid, f'slp.{start_row["year"]}.nc'))
                ds_year2 = ds_flip.sel(time=slice(f'{start_row["year"]+1}-01-01', f'{start_row["year"]+1}-12-31'))
                ds_year2['slp'].to_netcdf(os.path.join(outpath_mcms, sid, f'slp.{start_row["year"]+1}.nc'))

        ##### patch extraction #####
        siddf = tracks[tracks['sid']==sid]
        os.makedirs(os.path.join(outpath_patches, sid), exist_ok=True)
        
        # Initialize tracking center with ground truth for T=0
        lat_center, lon_center = siddf.iloc[0]['lat'], siddf.iloc[0]['lon'] % 360
        search_radius = 10.0 # ~1100 km radius for tracking
        
        synth_track = [(lat_center, lon_center)]
        
        for t in range(timesteps):
                if t>0:
                        # LAZY TRACKER: Find local minimum SLP in the predicted drift area
                        slp_search = ds['slp'].isel(time=t).sel(
                                lat=slice(lat_center - search_radius, lat_center + search_radius),
                                lon=slice(lon_center - search_radius, lon_center + search_radius)
                        )
                        # Find the index of the minimum SLP
                        if slp_search.size > 0: # Ensure the box didn't fall off the map
                                # Get 2D index of minimum value
                                min_idx = np.unravel_index(slp_search.argmin().values, slp_search.shape)
                                lat_center = slp_search.lat[min_idx[0]].values.item()
                                lon_center = slp_search.lon[min_idx[1]].values.item()
                        synth_track.append((lat_center, lon_center))
                
                
                # NOTE LATITUDE WE MADE NEGATIVE SINCE CODICAST OUTS ARE NORTH/SOUTH FLIPPED
                # 'aeqd': https://proj.org/en/stable/operations/projections/aeqd.html
                proj_km = Proj(proj='aeqd', lat_0=lat_center, lon_0=lon_center, units='km')
                # Project to find lat/lon corners of the equal-area box
                lon_grid, lat_grid = proj_km(x_grid, y_grid, inverse=True) #translate km to deg
                lon_grid=(lon_grid+360)%360 # because these datasets have lon as 0 to 360 (lat is still -90 to 90)
                lon_min = lon_grid.min() - resolution # +- reso because otherwise xarray will not include the edge points
                lon_max = lon_grid.max() + resolution
                lat_min = lat_grid.min() - resolution
                lat_max = lat_grid.max() + resolution
                
                dsarea = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                
                arealats = dsarea.lat.values 
                arealons = dsarea.lon.values
                
                data = dsarea.to_array().values[:, t, :, :] # V H W
                data = np.transpose(data, (1, 2, 0)) # H W V
                # print(data.shape)
                interp = RegularGridInterpolator(
                        (arealats, arealons),
                        data,
                        bounds_error=False,
                        fill_value=None
                )
                
                # Interpolate at new (lat, lon) pairs
                interp_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
                interp_values = interp(interp_points).reshape(s, s, data.shape[-1]) # H W V
                
                # print(interp_values.shape)
                np.save(os.path.join(outpath_patches, sid, f'{t}.npy'), interp_values)
                
        with open(os.path.join(outpath_patches, sid, 'track.csv'), 'w') as f:
                f.write('t,lat,lon\n')
                for t, (lat, lon) in enumerate(synth_track):
                        f.write(f'{t},{lat},{lon}\n')
        
        
        
# print(ds)

## make the topo 
ds = xr.open_dataset(input_topo) 
# print(ds)
lats = np.linspace(90 - (5.625/2), -90 + (5.625/2), 32) # already in MCMS order
ds = ds.interp(lat=lats, lon=lons, method='linear', )

# print(ds)
ds.to_netcdf(os.path.join(outpath_mcms, 'topo.nc'))




