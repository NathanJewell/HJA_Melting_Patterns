import pyreadr
import pdb
import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt
import pandas as pd
import xarray as xr
from datetime import datetime

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


##visualize the melt dates
#candidate = peak_swe
#for year_enum in candidate.sizes["year"]:
    #year_grid_data = candidate[:, :, year_enum]

    ##linearly interpolate for 100 colors 
    #color_quantity = 100
    #colors_list = [[0, 0, x] for x in range(color_quantity)]
    #colors_bounds = [0 + x * year_grid_data.max/color_quantity for x in range(color_quantity+1)] #cmap = mpl.colors.ListedColormap(colors_list)
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    #img = pyplot.imshow()

#pdb.set_trace() 

def load_swe_csv():
    df_raw = pd.read_csv("swe_matrix.csv")
    #change date to epoch or day

    #
    water_year_month_cutoff = 10
    water_year_start_date = lambda year : datetime.strptime(f"{year}-10-01")
    #df_raw['date'] = df_raw.apply(lambda row: datetime.strptime(row['date'], "%Y-%m-%d"))
    #df_raw['year'] = df_raw.apply(lambda row: row['date'][:4] + (0 if int(row['date'][5:7]) < water_year_month_cutoff else 1), axis=1) #select year from date

    #number of days since previous oct-1

    df_raw = pd.pivot_table(df_raw, values='swe_mm', index=['latitude', 'longitude', 'waterday', 'wateryear'])
    xrt = xr.DataArray(df_raw).unstack("dim_0")
    matrix = xrt.values[0].T
    np.save("swe_matrix.npy", matrix)

#dim ordering is (year, day, lat, lon)
def clean_swe_matrix():
    matrix = np.load("swe_matrix.npy")
    clean_matrix = np.nan_to_num(matrix)
    for year in range(matrix.shape[0]): #iterate over years
        for lat in range(matrix.shape[2]):
            for lon in range(matrix.shape[3]):
                if (np.any(matrix[year,:,lat,lon]) != np.nan):
    

    np.save("swe_matrix_clean.npy", clean_matrix)

def graph_day_coord_swe(swe_array, year='year', lat='lat', lon='lon'):
    plt.plot(range(len(swe_array)), swe_array)
    plt.title(f'{year} at {lat}-{lon}')
    plt.xlabel('Day')
    plt.ylabel('SWE')
    plt.savefig(f"daily_melt_graphs/SWE_{year}_{lat}_{lon}.png")
    plt.clf()

cutoff_swe = .1 
def compute_cluster_similarity(similarity_fx):
    matrix = np.load("swe_matrix_clean.npy")
    matrix_max = np.max(matrix)
    matrix_min = np.min(matrix)
    pdb.set_trace()
    valid_grid_day_cnt = 0
    skip_cnt = 0
    for year in range(matrix.shape[0]): #iterate over years
        for lat in range(matrix.shape[2]):
            for lon in range(matrix.shape[3]):
                day_list = matrix[year,:,lat,lon]
                nan_idx = np.isnan(day_list)
                day_list[day_list < 0] = np.nan
                pdb.set_trace()
                day_list[nan_idx] = np.interp(nan_idx, ~nan_idx, day_list[~nan_idx])
                if np.isnan(np.sum(day_list)) or np.sum(day_list) < cutoff_swe:
                    print(f'skipped {year} at {lat}-{lon}')
                    skip_cnt += 1
                    continue#skip the grid-cell-day if there is not enough data
                valid_grid_day_cnt += 1
                day_list = (day_list - matrix_min)/matrix_max
                graph_day_coord_swe(day_list, year, lat, lon)
                #pdb.set_trace()
    print(valid_grid_day_cnt)
    print(skip_cnt)

clean_swe_matrix()
compute_cluster_similarity(lambda x : x)


#spatial vs time-series interpolation
