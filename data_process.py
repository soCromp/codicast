# %%
import climate_learn as cl
import numpy as np
import xarray as xr


import xarray as xr
import pandas as pd
import os
from tqdm import tqdm


# for the era5 data (my stuff)
root_directory = '/mnt/data/sonia/cyclone/0.25'
target_directory = '/mnt/data/sonia/codicast-data/windmag-500hpa'
temp_directory = os.path.join(target_directory, 'temporary')
# dirnames=['slp', 'wind_500hpa', 'wind_500hpa', 'temperature', 'humidity']
# shortnames = ['slp', 'u', 'v', 't', 'q']

os.makedirs(temp_directory, exist_ok=True)

######### MULTIVAR ############
# # 180 / 5.625 = 32 exactly
# # 360 / 5.625 = 64 exactly
# # Generating coordinates centered within the 5.625 degree bins
# new_lat = np.linspace(-90 + (5.625/2), 90 - (5.625/2), 32)
# new_lon = np.linspace(0 + (5.625/2), 360 - (5.625/2), 64)

# for var, short in zip(dirnames, shortnames):
#     os.makedirs(os.path.join(target_directory, 'temporary', var), exist_ok=True)
#     for year in tqdm(range(1940, 2025)):
#         if 'wind' in var:
#             fname = f'wind.{year}.nc'
#         else:
#             fname = f'{var}.{year}.nc'
            
#         ds = xr.open_dataset(os.path.join(root_directory, var, fname))
#         ds = ds.interp(lat=new_lat, lon=new_lon, method="linear")
#         start = f'{year}-01-01 00:00:00'
#         correct_time = pd.date_range(start=pd.to_datetime(start), periods=len(ds.time), freq='6h')
#         ds = ds.assign_coords(time=correct_time)
#         ds.to_netcdf(os.path.join(target_directory, 'temporary', var, f'{var}_{year}_5.625deg.nc'))


######## WINDMAG ############
dirnames = ['wind_500hpa']
shortnames = ['__xarray_dataarray_variable__']

# os.makedirs(os.path.join(target_directory, 'temporary', dirnames[0]), exist_ok=True)
# for year in tqdm(range(1940, 2025)):
#     ds = xr.open_dataset(os.path.join('/mnt/data/sonia/codicast-data/multivar/temporary/wind_500hpa',
#                                       f'wind_500hpa_{str(year)}_5.625deg.nc'))
#     windmag = np.sqrt(ds['u']**2 + ds['v']**2)
#     windmag.to_netcdf(os.path.join(target_directory, 'temporary', dirnames[0], f'wind_500hpa_{str(year)}_5.625deg.nc'))

def select_merge_data(name_list, short_list, year_start, year_end, data_folder, src_folder, resolution, lat, long):
    directory_paths = name_list
    concat_years = []
    counts = 0
    years = []
    
    for year in range(year_start, year_end+1):
        years.append(str(year))

    for year in years:
        print('>>>', year, '<<<')
        datas = []
        for directory_path, short_name in zip(directory_paths, short_list):
            # # Open the NetCDF file using xarray
            # ds = xr.open_dataset(data_folder + '/' + directory_path + '/' + directory_path + '_' + year + '_' + str(resolution) + 'deg.nc')
    
            ds = xr.open_dataset(os.path.join(src_folder, directory_path, f'{directory_path}_{year}_{str(resolution)}deg.nc'))
            if directory_path == 'temperature':
                ds = ds.sel(pressure_level=925)
            if year == 2015:
                ds = ds.sel(time=slice('2015-01-01 00:00:00', '2015-08-31 19:00:00'))
            data = ds[short_name].values
            data = data.reshape((-1, 1, lat, long))
            datas.append(data)
        
            # # =========== pressure-level =============  
            # if directory_path == 'geopotential_500':
            #     ds = xr.open_dataset(src_folder + '/' + directory_path + '/' + directory_path + 'hPa_' + year + '_' + str(resolution) + 'deg.nc')
            #     ds = ds.isel(time=slice(None, None, 6))
            #     geopotential = ds['z'].values
            #     geopotential = geopotential.reshape((-1, 1, lat, long))
            #     print('geopotential_500:', geopotential.shape)
                
            # if directory_path == 'temperature_850':
            #     ds = xr.open_dataset(src_folder + '/' + directory_path + '/' + directory_path + 'hPa_' + year + '_' + str(resolution) + 'deg.nc')
            #     ds = ds.isel(time=slice(None, None, 6))
            #     temperature = ds['t'].values
            #     temperature = temperature.reshape((-1, 1, lat, long))
            #     print('temperature_850:', temperature.shape)
        
            # # ======================= surface variable ======================
            # if directory_path == '2m_temperature':
            #     ds = xr.open_dataset(src_folder + '/' + directory_path + '/' + directory_path + '_' + year + '_' + str(resolution) + 'deg.nc')
            #     # ds = ds.isel(time=slice(None, None, 6))
            #     t2m_temperature = ds['2m_temperature'].values
            #     t2m_temperature = t2m_temperature.reshape((-1, 1, lat, long))
            #     print('2m_temperature:', t2m_temperature.shape)
        
            # if directory_path == '10m_u_component_of_wind':
            #     ds = xr.open_dataset(src_folder + '/' + directory_path + '/' + directory_path + '_' + year + '_' + str(resolution) + 'deg.nc')
            #     # ds = ds.isel(time=slice(None, None, 6))
            #     u10m = ds['10m_u_component_of_wind'].values
            #     u10m = u10m.reshape((-1, 1, lat, long))
            #     print('10m_u_component_of_wind:', u10m.shape)
        
            # if directory_path == '10m_v_component_of_wind': 
            #     ds = xr.open_dataset(src_folder + '/' + directory_path + '/' + directory_path + '_' + year + '_' + str(resolution) + 'deg.nc')
            #     print(ds)
            #     # ds = ds.isel(time=slice(None, None, 6))
            #     v10m = ds['10m_v_component_of_wind'].values
            #     v10m = v10m.reshape((-1, 1,lat, long))
            #     print('10m_v_component_of_wind:', v10m.shape)
        
        # concatenate one year
        for ds in datas:
            print(ds.shape, np.isnan(ds).sum())
        concat_one_year = np.concatenate(datas, axis=1)        
        print("concat_one_year.shape:", concat_one_year.shape)
    
        concat_years.append(concat_one_year)
    
        counts += concat_one_year.shape[0]

    concat_years = np.concatenate(concat_years, axis=0)
    
    print("concat_years.shape:", concat_years.shape)
    
    print("total time points:", counts)

    print(">>> saving data <<<") 
    np.save(data_folder + '/concat_' + str(year_start) + '_' + str(year_end) + '_' + str(resolution) + '_' + str(concat_years.shape[1]) + 'var.npy', concat_years)
    

    print(">>> saved data <<<")


resolution = 5.625 
lat, long = 32, 64

# test set
year_start, year_end = 2016, 2024
select_merge_data(dirnames, shortnames, year_start, year_end, target_directory, temp_directory, resolution, lat, long)

# validation set
year_start, year_end = 2016, 2016
select_merge_data(dirnames, shortnames, year_start, year_end, target_directory, temp_directory, resolution, lat, long)

# ### Training set
year_start, year_end = 1940, 2015
select_merge_data(dirnames, shortnames, year_start, year_end, target_directory, temp_directory, resolution, lat, long)
