import pyreadr
import pdb
import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt
import pandas as pd
import xarray as xr
from datetime import datetime
import dtw
import scipy as scp
from alive_progress import alive_bar

#http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html
#https://stats.stackexchange.com/questions/131281/dynamic-time-warping-clustering
#LB Keough time warping
#https://pypi.org/project/dtw-python/

#swap_dict = {
    #"dim_0" : "lat",
    #"dim_1" : "lon",
    #"dim_2" : "year"
#}
#load_rdata_to_xarray = lambda filename : list(pyreadr.read_r(filename).values())[0]
#set_dim_names = lambda xarray_obj : xarray_obj.swap_dims(swap_dict)

#peak_swe_date = set_dim_names(load_rdata_to_xarray('peak_swe_date_HJA.RData'))
#peak_swe = set_dim_names(load_rdata_to_xarray('peak_swe_HJA.RData'))
##swe_mm = load_rdata_to_xarray('swe_HJA_matrix_mm.RData')
#f_melt = set_dim_names(load_rdata_to_xarray('f_melt_HJA.RData'))


#start, end are rgb tuples between 
def color_interpolation(start, end, steps):
    #assert(len(start) == len(end))
    #everything is less than = 1assert()
    color_slope = tuple([(start[i], (start[i] - end[i])/steps) for i in range(len(start))])
    return lambda step : tuple(slope[0] - (slope[1] * step) for slope in color_slope)

##visualize the melt dates
def graph_watershed_data(grid_data, colors=None, show=False, save=True, title=None):
        #linearly interpolate for 100 colors 
        color_quantity = 30
        colors_fx = color_interpolation((1, 1, 1), (0, 0, 1), color_quantity)
        colors_list = [colors_fx(x) for x in range(color_quantity)]
        colors_bounds = [0 + x * (grid_data.max()/color_quantity) for x in range(color_quantity+1)] 
        cmap = mpl.colors.ListedColormap(colors_list)
        norm = mpl.colors.BoundaryNorm(colors_bounds, cmap.N+1)

        img = plt.imshow(grid_data, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)# cmap)
        if title:
            plt.title(title)
        #plt.colorbar(img)#, cmap=cmap, norm=norm, boundaries=colors_bounds)
        if save:
            plt.savefig(f"watershed_colored_{title}.png")
        if show:
            plt.show()
        plt.clf()

def graph_day_coord_swe(swe_array, year='year', lat='lat', lon='lon'):
    plt.plot(range(len(swe_array)), swe_array)
    plt.title(f'{year} at {lat}-{lon}')
    plt.xlabel('Day')
    plt.ylabel('SWE')
    plt.savefig(f"daily_melt_graphs/SWE_{year}_{lat}_{lon}.png")
    plt.clf()

def load_swe_csv():
    df_raw = pd.read_csv("swe_matrix.csv")
    #change date to epoch or day

    #
    pdb.set_trace()
    water_year_month_cutoff = 10
    water_year_start_date = lambda year : datetime.strptime(f"{year}-10-01")
    #df_raw['date'] = df_raw.apply(lambda row: datetime.strptime(row['date'], "%Y-%m-%d"))
    #df_raw['year'] = df_raw.apply(lambda row: row['date'][:4] + (0 if int(row['date'][5:7]) < water_year_month_cutoff else 1), axis=1) #select year from date

    #number of days since previous oct-1

    df_raw = pd.pivot_table(df_raw, values='swe_mm', index=['latitude', 'longitude', 'waterday', 'wateryear'])
    #df_raw = df_raw[['swe_mm', 'latitude', 'longitude', 'waterday', 'wateryear']]
    pdb.set_trace()
    xrt = xr.DataArray(df_raw).unstack("dim_0")
    matrix = xrt.values[0].T
    np.save("swe_matrix.npy", matrix)

#dim ordering is (year, day, lat, lon)
#nan values need to be eliminated
#interpolate missing days in year-lat-lon series
#years with all nan data? mostly nan data?
#specific coords without data for some/all years

    
    
cutoff_swe = .1 
def clean_swe_matrix():
    matrix = np.load("swe_matrix.npy")
    matrix_max = np.max(matrix)
    matrix_min = np.min(matrix)
    pdb.set_trace()
    valid_grid_day_cnt = 0
    skip_cnt = 0
    nan_coord_year_cutoff = 20
    data_mask = np.zeros((matrix.shape[2], matrix.shape[3]))
    for year in range(matrix.shape[0]): #iterate over years
        for lat in range(matrix.shape[2]):
            for lon in range(matrix.shape[3]):
#https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
                day_list = matrix[year,:,lat,lon]
                if np.isnan(day_list[0]):
                    day_list[0] = 0
                if np.isnan(day_list[-1]):
                    day_list[-1] = 0
                if lat == 6 and lon == 6:
                    pdb.set_trace()
                idxs = np.arange(day_list.shape[0])
                day_list[day_list < 0] = np.nan
                good_vals = np.isfinite(day_list)
                interp_vals = scp.interpolate.interp1d(idxs[good_vals], day_list[good_vals], bounds_error=False)
                day_list = np.where(np.isfinite(day_list), day_list, interp_vals(idxs))
                if np.isnan(np.sum(day_list)) or np.sum(day_list) < cutoff_swe:
                    data_mask[lat, lon] = 1
                    print(f'skipped {year} at {lat}-{lon}')
                    skip_cnt += 1
                    continue#skip the grid-cell-day if there is not enough data
                valid_grid_day_cnt += 1
                matrix[year, :, lat, lon] = (day_list - np.min(day_list)) / (np.max(day_list) - np.min(day_list))

                print(f'!!!Graphed {year} at {lat}-{lon}')
                #graph_day_coord_swe(day_list, year, lat, lon)
                #pdb.set_trace()
    np.save("swe_matrix_clean.npy", matrix)
    np.save("swe_data_mask.npy", data_mask)
    print(valid_grid_day_cnt)
    print(skip_cnt)

#load_swe_csv()
#clean_swe_matrix()

def DWT_distance_swe(series_1, series_2):
    if np.isnan(np.sum(series_1)) or np.sum(series_1) < cutoff_swe:
        return np.nan#skip the grid-cell-day if there is not data to do distance calculation
    if np.isnan(np.sum(series_2)) or np.sum(series_2) < cutoff_swe:
        return np.nan#skip the grid-cell-day if there is not data to do distance calculation
    result = dtw.dtw(series_1, series_2, keep_internals=False)
    return result.distance

def compute_swe_distance(distance_fx):
    matrix = np.load("swe_matrix_clean.npy")
    mask = np.load("swe_data_mask.npy")
    print(matrix.shape)
    latlon_1d = lambda lat, lon : lat * matrix.shape[2] + lon
    latlon_2d = lambda latlon1d : (int(latlon1d / matrix.shape[2]), latlon1d % matrix.shape[2])
    pdb.set_trace()
    #distance_latlon = np.ones((max_1d_latlon_idx, max_1d_latlon_idx)) * 1000
    distance_latlon = np.ones((matrix.shape[0], matrix.shape[2], matrix.shape[3], matrix.shape[2], matrix.shape[3])) * -1
    for year in range(matrix.shape[0]):
        lat_idx = np.arange(matrix.shape[2])
        lon_idx = np.arange(matrix.shape[3])

        grid_cell_year = lambda lat, lon : matrix[year, :, lat, lon]
        with alive_bar(len(lat_idx) * len(lon_idx)) as bar:
            for lat_a in lat_idx:
                for lon_a in lon_idx:
                    bar()
                    if mask[lat_a, lon_a]:
                        continue
                    for lat_b in lat_idx:
                        for lon_b in lon_idx:
                            if mask[lat_b, lon_b]:
                                continue
                            distance_latlon[year, lat_a, lon_a, lat_b, lon_b] = distance_fx(grid_cell_year(lat_a, lon_a), grid_cell_year(lat_b, lon_b))
    pdb.set_trace()
    np.save("distance_latlon.npy", distance_latlon)
    pdb.set_trace()


def do_clustering():
    matrix_shape = (17, 366, 19, 10)
    matrix = np.zeros(matrix_shape)
    latlon_1d = lambda lat, lon : lat * matrix.shape[2] + lon
    latlon_2d = lambda latlon1d : (int(latlon1d / (matrix.shape[2])), latlon1d % (matrix.shape[3]))
    distances = np.load("distance_latlon.npy")
    mask = np.load("swe_data_mask.npy")
    for year in range(distances.shape[0]):
        pdb.set_trace()
        for lat in range(distances.shape[1]):
            for lon in range(distances.shape[2]):
                if not mask[lat, lon]:
                    graph_watershed_data(distances[year, lat, lon], save=True, show=False, title="{}, {}-{}".format(year, lat, lon))

    
            

    pdb.set_trace()

    
    for year in range(distances.shape[0]):
        sums = np.zeros((19, 10))
        cnt = 0
        for lat in range(distances.shape[1]):
            for lon in range(distances.shape[2]):
                if not mask[lat, lon]:
                    sums += distances[year, lat, lon]
                    cnt += 1
        average = sums / cnt



#compute_swe_distance(DWT_distance_swe)
#clean_swe_matrix()
do_clustering()

        


#spatial vs time-series interpolation
