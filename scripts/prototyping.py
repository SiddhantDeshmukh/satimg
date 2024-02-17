import ee
import geemap.colormaps as cm
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from PIL import Image
from typing import Dict, Tuple
import requests

# Set seed for testing
np.random.seed(420)
PALETTE = cm.palettes["coolwarm"]["default"]


def random_bars_img_grid(img_width: int, img_height: int) -> np.ndarray:
    # Random bars with set bars in the middle
    img_grid = np.empty(shape=(img_height, img_width))
    # Create vertical bars, every other column is random
    for j in range(img_width):
        if j % 2 == 0:
            img_grid[:, j] = 0.
        else:
            img_grid[:, j] = np.random.uniform(
                low=img_width / 2, high=img_width*3/2)

    return img_grid


def consecutive_bars_img_grid(img_width: int, img_height: int) -> np.ndarray:
    # Consecutive bars
    img_grid = np.empty(shape=(img_height, img_width))
    for j in range(img_width):
        img_grid[:, j] = j

    return img_grid


def shear_img_grid(img_grid: np.ndarray, step=1) -> np.ndarray:
    shift = 1
    for i in range(0, img_grid.shape[0], step):
        if i > 0:
            img_grid[i] = np.roll(img_grid[i], shift=shift, axis=0)
            shift += 1

    return img_grid


def roll_img_grid(img_grid: np.ndarray, shift=1, axis=0) -> np.ndarray:
    # Roll the image grid as requested
    return np.roll(img_grid, shift=shift, axis=axis)


def roll_random_img():
    img_width, img_height = 32, 24  # in px, height value later will be number of inputs
    # Create a test img for manipulation
    nrows, ncols = 5, 3
    center_idx = (nrows // 2, ncols // 2)
    fig, ax = plt.subplots(nrows, ncols)
    cmap = "inferno"
    # Original img_grid
    # img_grid_og = random_bars_img_grid(img_width, img_height)
    img_grid_og = consecutive_bars_img_grid(img_width, img_height)
    img_grid_og = shear_img_grid(img_grid_og, step=5)
    for i in range(nrows):
        for j in range(ncols):
            if i == center_idx[0] and j == center_idx[1]:
                # Middle image is original
                ax[i, j].imshow(img_grid_og, cmap=cmap, extent="lower")
            else:
                # Roll image then plot
                img_grid = img_grid_og.copy()
                roll_x = center_idx[1] - j
                roll_y = center_idx[0] - i

                if roll_x != 0:
                    img_grid = roll_img_grid(img_grid, shift=roll_x, axis=0)
                if roll_y != 0:
                    img_grid = roll_img_grid(img_grid, shift=roll_y, axis=1)

                ax[i, j].imshow(img_grid, cmap=cmap, extent="lower")

    plt.show()


def get_mean_in_timeframe(lst_data: ee.ImageCollection, initial_date: str,
                          final_date: str) -> ee.Image:
    # Often the LST data needs to be stacked across a larger timeframe bc
    # the "daily" coverage is just what the satellite mapped that day
    _lst_data = lst_data.filterDate(initial_date, final_date)
    return _lst_data.mean()


def lst_img_transform(lst_img: ee.Image, key="LST_Day_1km") -> ee.Image:
    # LST Data needs to be scaled by 0.02 and converted into C
    lst_img = lst_img.select(key).multiply(0.02).add(-273.15)
    return lst_img


def process_lst_data(initial_date="2000-01-01", final_date="2020-01-01",
                     bands=["LST_Day_1km", "LST_Night_1km"]):
    # Get all data
    lst = ee.ImageCollection("MODIS/006/MOD11A1")
    # Apply transformations (scale by 0.02, convert to C)
    lst = lst.select(bands).filterDate(initial_date, final_date)
    return lst


def minmax(arr: np.ndarray, axis=None) -> Tuple:
    return np.min(arr, axis=axis), np.max(arr, axis=axis)


def get_lst_data(download=False):
    # Download data arrays for climate stripe testing
    start_year, end_year = 2010, 2021
    initial_date = f"{start_year}-01-01"
    final_date = f"{end_year}-01-01"
    u_lon = 0.1275
    u_lat = 51.507222
    u_poi = ee.Geometry.Point(u_lon, u_lat)
    # Define a region of interest with a buffer zone of 1000 km
    roi = u_poi.buffer(1e6)
    # Load LST
    lst = ee.ImageCollection("MODIS/006/MOD11A1")
    # Filter by date
    lst = lst.select("LST_Day_1km", "LST_Night_1km").filterDate(initial_date,
                                                                final_date)
    data_arrs = {}
    # Generate requested data, download and store array
    for year in range(start_year, end_year):
        lst = process_lst_data(initial_date=f"{year}-01-01",
                               final_date=f"{year}-12-31",
                               bands=["LST_Day_1km", "LST_Night_1km"])

        for band_id in ["Day", "Night"]:
            band = f"LST_{band_id}_1km"
            img = lst_img_transform(lst.mean(), key=band)
            # Transform to correct temperature scale
            # img = lst_img_transform(lst_img, key=band)
            # Get download URL
            print(f"Setting up URL for {band} {year}")
            url = img.getDownloadURL({
                "bands": [band],
                "scale": 2000,
                "region": roi,
                "format": "NPY"
            })
            # Download NPY structured data
            print(f"Downloading array for {band} {year}")
            response = requests.get(url)
            print(response.status_code)
            data = np.load(io.BytesIO(response.content))[band]
            data_arrs[f"{band}_{year}"] = data
            if download:
                np.save(f"../res/stripe_arrs/climate_stripes_{band}_{year}",
                        data)
    return data_arrs


def load_lst_test_data() -> Dict:
    res_dir = "../res/stripe_arrs"
    files = [f"{res_dir}/{f}" for f in os.listdir(res_dir)]
    data_arrs = {
        "_".join(f.split("/")[-1].split(".")[0].split("_")[2:]): np.load(f)
        for f in files
    }
    return data_arrs


def add_colorbar(ax: plt.Axes, image):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(image, ax=ax, cax=cax, orientation="horizontal")
    return cbar


def plot_climate_stripes(data_arrs: Dict):
    # Plots a dictionary of processed data arrays as climate stripes
    years = list(set([k.split("_")[-1].split(".")[0]
                 for k in data_arrs.keys()]))
    years.sort()
    start_year, end_year = int(years[0]), int(years[-1])

    num_stripes = end_year - start_year  # 1 stripe for each year
    img_width = data_arrs[list(data_arrs.keys())[0]].shape[1]
    stripe_width = img_width // num_stripes
    # Create stripes in data
    day_data = np.empty_like(data_arrs[list(data_arrs.keys())[0]])
    night_data = np.empty_like(day_data)
    for i in range(num_stripes):
        year = start_year + i
        idxs = slice(i*stripe_width, (i+1)*stripe_width)
        day_data[:, idxs] = data_arrs[f"LST_Day_1km_{year}"][:, idxs]
        night_data[:, idxs] = data_arrs[f"LST_Night_1km_{year}"][:, idxs]

    # Display each stripe
    fig, ax = plt.subplots(1, 2)
    im_day = ax[0].imshow(day_data, cmap="coolwarm", origin="upper")
    im_night = ax[1].imshow(night_data, cmap="coolwarm", origin="upper")

    cbar_day = add_colorbar(ax[0], im_day)
    cbar_night = add_colorbar(ax[1], im_night)

    # TODO:
    # - x axis label as year
    # - combined colorbar for each side
    for ax_ in ax:
        for i in range(num_stripes):
            ax_.axvline(i*stripe_width, c='k', ls='-')
        # ax_.set_xticklabels(np.linspace(start_year, end_year, num=img_width))
    ax[0].set_title("MODIS LST Day")
    ax[1].set_title("MODIS LST Night")

    return fig, ax


def main():
    # Initialise Google Earth Engine
    ee.Authenticate()
    ee.Initialize(project="ee-warming-stripes")

    data = get_lst_data(download=False)
    plot_climate_stripes(data)
    # test_data = load_lst_test_data()
    # plot_climate_stripes(test_data)
    plt.show()
    exit()

    # # Initial date of interest (inclusive)
    # i_date = '2017-01-01'
    # # Final date of interest (exclusive)
    # f_date = '2020-01-01'
    # # Selection of appropriate bands and dates for LST
    # lst_data = get_lst_data(i_date, f_date)
    # num_stripes = 10

    # lst_img = get_mean_in_timeframe(lst_data, "2018-01-01", "2018-12-31")
    # fig, ax = lst_imshow(lst_img)
    # plt.show()

    # # Define the urban location of interest as a point near Lyon, France.
    # u_lon = 4.8148
    # u_lat = 45.7758
    # u_poi = ee.Geometry.Point(u_lon, u_lat)

    # # Define the rural location of interest as a point away from the city.
    # r_lon = 5.175964
    # r_lat = 45.574064
    # r_poi = ee.Geometry.Point(r_lon, r_lat)

    # # Define a region of interest with a buffer zone of 1000 km
    # roi = r_poi.buffer(1e6)

    # # Reduce the LST collection by mean.
    # # TODO: Select 'num_stripes' images in evenly-spaced parts and slice them
    # # lst_img = lst.mean()

    # # # Adjust for scale factor.
    # # lst_img = lst_img.select('LST_Day_1km').multiply(0.02)

    # # # Convert Kelvin to Celsius.
    # # lst_img = lst_img.select('LST_Day_1km').add(-273.15)

    # url = lst_img.getThumbURL({
    #     'min': 10, 'max': 30, 'dimensions': 512, 'region': roi,
    #     'palette': ['blue', 'yellow', 'orange', 'red']})
    # print(url)


if __name__ == "__main__":
    main()
