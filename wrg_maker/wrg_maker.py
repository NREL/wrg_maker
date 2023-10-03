import logging
import os
import time

import numpy as np
import pandas as pd
import utm
from rex.multi_year_resource import MultiYearWindResource
from rex.renewable_resource import WindResource
from rex.resource_extraction import MultiYearResourceX, ResourceX
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import check_res_file
from scipy.interpolate import griddata
from scipy.stats import weibull_min

logger = logging.getLogger(__name__)

WIND_SPEED = "windspeed"
WIND_DIR = "winddirection"


class WrgMaker:
    def __init__(self, h5_file, resolution, wrg_file=None, hub_height=100, bin_size=30, box_coords=None):
        self._h5_file = h5_file
        self._wrg_file = wrg_file or h5_file.replace(".h5", ".wrg")
        assert self._wrg_file.endswith(".wrg"), "Wrg file must end with '.wrg'"
        self._hub_height = hub_height
        self._bin_size = bin_size
        self._handler = self._get_resource_handler(self._h5_file)
        self._pixels = self._get_pixels(box_coords=box_coords)
        # Convert from Km to m
        self._resolution = resolution * 1000

        out_dir = os.path.dirname(self._wrg_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    @staticmethod
    def _get_resource_handler(h5_file):
        """
        Get resource handler for h5_file

        Parameters
        ----------
        h5_file : str
            Path to h5 file

        Returns
        -------
        ResourceHandler
            Resource handler for h5_file
        """
        multi_h5, _ = check_res_file(h5_file)
        if multi_h5:
            return MultiYearWindResource
        else:
            return WindResource

    def _get_pixels(self, box_coords=None):
        """
        Get pixel ids for all pixels in h5_file

        Parameters
        ----------
        box_coords : tuple, optional
            (lat_lon_1, lat_lon_2) coordinates of the bounding box

        Returns
        -------
        int[]
            List of Pixel ids to compute WRG for
        """
        if box_coords:
            logger.debug(f"Computing WRG for pixels within bounding box: {box_coords}")
            lat_lon_1, lat_lon_2 = box_coords
            multi_h5, _ = check_res_file(self._h5_file)
            if multi_h5:
                rex_cls = MultiYearResourceX
            else:
                rex_cls = ResourceX

            with rex_cls(self._h5_file, res_cls=self._handler) as h5:
                pixels = h5.box_gids(lat_lon_1, lat_lon_2)
        else:
            logger.debug("Computing WRG for all pixels")
            with self._handler(self._h5_file) as h5:
                pixels = list(range(h5.shape[1]))

        return pixels

    @staticmethod
    def _compute_sector_stats(sector, total_count):
        """
        Compute sector level wrg stats

        Parameters
        ----------
        sector : pandas.DataFrame
            Wind speed and direction for one sector of angles
        total_count : int
            Total number of data points

        Returns
        -------
        pandas.Series
        """
        try:
            k, _, a = weibull_min.fit(sector[WIND_SPEED], floc=0)
        except Exception:
            k, a = 0, 0
            logger.exception("Failed to fit weibull distribution!")

        freq = len(sector) / total_count

        return pd.Series({"freq": freq * 1000, "A": a * 10, "k": k * 100})

    @staticmethod
    def _compute_power_density(wind_speed, rotor_radius=50):
        """
        Compute the power density for the given wind resource

        Parameters
        ----------
        wind_speed : float
            Wind speed [m/s].
        rotor_radius : float, optional
            Rotor radius [m].

        Returns
        -------
        float
        """
        rho = 1.225  # air density at sea level in kg/m3
        area = np.pi * rotor_radius**2  # swept area of the wind turbine in m2
        C_p = 0.33  # power coefficient of the wind turbine

        return 0.5 * rho * area * wind_speed.mean() ** 3 * C_p / 1000

    @classmethod
    def _compute_wrg_values(cls, h5_file, handler, pixel_id, hub_height=100, bin_size=30):
        """
        Compute the wind resource gid files values

        Parameters
        ----------
        pixel_id : int
            Location ID
        hub_height : int, optional
            Hub height [m]
        bin_size : int, optional
            Size of each sector bin in degrees, by default 30

        Returns
        -------
        list:
            WRG parameters for the given pixel_id
        """
        with handler(h5_file) as h5:
            wspd = h5[f"{WIND_SPEED}_{int(hub_height)}m", :, pixel_id]
            # replace zeros with small non-zero values
            non_zero_values = np.random.uniform(1e-6, 1e-3, wspd.size).reshape(wspd.shape)
            wspd = np.where(wspd > 0, wspd, non_zero_values)

            wdir = h5[f"{WIND_DIR}_{int(hub_height)}m", :, pixel_id]

            meta = h5.meta
            if "elevation" in meta:
                elevation = meta.loc[pixel_id, "elevation"]
            else:
                elevation = 0

        # We want our first wind direction sector centered at zero. Therefore, subtract 360 degrees
        # from angles > center
        center = 360 - bin_size / 2
        wdir[wdir > center] -= 360

        # get sector angles
        sector_angles = np.arange(-bin_size / 2, center + 1, bin_size)

        # Compute A and K for all sectors
        k, _, A = weibull_min.fit(wspd, floc=0)
        # Compute power density
        power_density = cls._compute_power_density(wspd, rotor_radius=hub_height / 2)

        pixel_wrg = [int(hub_height), elevation, A, k, power_density, len(sector_angles) - 1]

        # Create dataframe to handle computation
        df = pd.DataFrame(data={WIND_SPEED: wspd, WIND_DIR: wdir})
        df["sector"] = pd.cut(wdir, sector_angles)

        # Compute sector level stats and add to list
        pixel_wrg.extend(df.groupby("sector").apply(cls._compute_sector_stats, len(df)).stack().values)

        return pixel_wrg

    def compute_wrg_stats(self, max_workers=None):
        """
        Compute WRG stats for all pixels in h5_file

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of workers to use, by default None

        Returns
        -------
        ndarray
            Array of WRG stats. Each row represents stats for a given pixel
        """
        ts = time.time()
        max_workers = max_workers or os.cpu_count()
        logger.info(f"Computing WRG stats for {len(self._pixels)} pixels using {max_workers} workers...")
        wrg_stats = []
        if max_workers > 1:
            EXECUTOR = SpawnProcessPool
            with EXECUTOR(max_workers=max_workers, loggers=__name__) as executor:
                futures = []
                for pixel in self._pixels:
                    futures.append(
                        executor.submit(
                            self._compute_wrg_values,
                            self._h5_file,
                            self._handler,
                            pixel,
                            hub_height=self._hub_height,
                            bin_size=self._bin_size,
                        )
                    )

                for i, future in enumerate(futures):
                    wrg_stats.append(future.result())
                    if not (i % 100):
                        logger.debug(f"- Computed WRG stats for {i + 1} out of {len(self._pixels)} pixels")
        else:
            for i, pixel in enumerate(self._pixels):
                wrg_stats.append(
                    self._compute_wrg_values(
                        self._h5_file, self._handler, pixel, hub_height=self._hub_height, bin_size=self._bin_size
                    )
                )
                if not (i % 100):
                    logger.debug(f"- Computed WRG stats for {i + 1} out of {len(self._pixels)} pixels")

        tt = (time.time() - ts) / 60
        logger.info(f"- WRG stats compute in {tt:.4f} minutes!")

        return np.array(wrg_stats, dtype=np.float32)

    @staticmethod
    def _buffer_coords(coords_min, coords_max, buffer=0.2):
        """
        Buffer coordinates to remove the given amount from each edge.
        Retained data will be 1 - buffer*2

        Parameters
        ----------
        coords_min : float
            Minimum x coordinate
        coords_max : float
            Maximum x coordinate
        buffer : float, optional
            Buffer size in %, by default 0.2

        Returns
        -------
        tuple
            Buffered min and max coordinates
        """
        coords_range = coords_max - coords_min

        buffered_min = coords_min + coords_range * buffer
        buffered_max = coords_max - coords_range * buffer

        return buffered_min, buffered_max

    @classmethod
    def _get_buffered_grid(cls, x, y, resolution, buffer=0.2):
        """
        Get meshgrid of x and y coordinates after buffering by given ammount

        Parameters
        ----------
        x : ndarray
            Vector of x coordinates
        y : ndarray
            Vector of y coordinates
        resolution : int
            Pixel resolution in meters
        buffer : float, optional
            Buffer size in %, by default 0.2

        Returns
        -------
        tuple
            x and y meshgrids respresenting uniform buffered coordinates
        """
        x_min = np.ceil(np.min(x) / resolution) * resolution
        x_max = np.floor(np.max(x) / resolution) * resolution
        x_min, x_max = cls._buffer_coords(x_min, x_max, buffer=buffer)
        xi = np.arange(x_min, x_max, resolution)

        y_min = np.ceil(np.min(y) / resolution) * resolution
        y_max = np.floor(np.max(y) / resolution) * resolution
        y_min, y_max = cls._buffer_coords(y_min, y_max, buffer=buffer)
        yi = np.arange(y_min, y_max, resolution)

        return np.meshgrid(xi, yi)

    def _fill_col_data(self, col_data, column_num, columns):
        """
        Replace fill value in col data with dummy data

        Parameters
        ----------
        col_data : ndarray
            Column data to fill if needed
        column_num : int
            Column number
        columns : int
            Total number of columns

        Returns
        -------
        ndarray
            Column data with fill values replaced with dummy data
        """
        bad_value = -9e16
        dummy_a = 100
        dummy_k = 300
        n_sectors = (columns - 6) // 3
        fill_data = [self._hub_height, 0, dummy_a, dummy_k, 5000, n_sectors, 1000, dummy_a, dummy_k] + [
            0,
            dummy_a,
            dummy_k,
        ] * (n_sectors - 1)
        msg = f"fill_data length ({len(fill_data)}) does not match number of columns ({columns})!"
        assert len(fill_data) == columns, msg

        return np.where(col_data > bad_value, col_data, fill_data[column_num])

    def regrid_wrg(self, wrg_stats, buffer=0.2):
        """
        Regrid WRG data to fit a uniform buffered grid

        Parameters
        ----------
        wrg_stats : ndarray
            Ndarray of raw WRG stats. Each row represents stats for a given pixel.
        buffer : float, optional
            Buffer size in %, by default 0.2

        Returns
        -------
        pandas.DataFrame
            DataFrame of regridded WRG data
        """
        with self._handler(self._h5_file) as h5:
            # Convert pixel coordinate to UTM
            lat, lon = h5.lat_lon[self._pixels].T
            mean_lat, mean_lon = h5.lat_lon.mean(axis=0)
            _, _, zone_number, zone_letter = utm.from_latlon(mean_lat, mean_lon)
            x, y, _, _ = utm.from_latlon(lat, lon, zone_number, zone_letter)

        xi, yi = self._get_buffered_grid(x, y, self._resolution, buffer=buffer)

        regrid_wrg = pd.DataFrame(index=range(len(xi.ravel())))
        regrid_wrg["X"] = xi.ravel(order="F").astype(np.float32)
        regrid_wrg["Y"] = yi.ravel(order="F").astype(np.float32)

        interp_data = griddata((x, y), wrg_stats, (xi, yi), method="linear").astype(np.float32)
        for i in range(interp_data.shape[-1]):
            col_data = interp_data[:, :, i].ravel(order="F")
            col_data = self._fill_col_data(col_data, i, interp_data.shape[-1])
            if i in [2, 3]:
                col_data = np.round(col_data, 2)
            elif i == 4:
                col_data = np.round(col_data, 0)
            else:
                col_data = col_data.astype(int)

            regrid_wrg[i] = col_data

        return regrid_wrg

    def write_wrg_file(self, wrg_df):
        """
        Write WRG data to wrg_file

        Parameters
        ----------
        wrg_df : pandas.DataFrame
            DataFrame of WRG data with each row representing a pixel
        """
        ts = time.time()
        logger.info(f"Writing WRG data to {self._wrg_file}...")
        num_X = len(wrg_df["X"].unique())
        num_Y = len(wrg_df["Y"].unique())

        min_X = wrg_df["X"].min()
        min_Y = wrg_df["Y"].min()

        header = [num_X, num_Y, round(min_X), round(min_Y), round(self._resolution)]
        header = " ".join([str(h) for h in header])

        wrg_df.to_csv(self._wrg_file, sep="\t", header=None)

        with open(self._wrg_file, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + "\n" + content)

        tt = (time.time() - ts) / 60
        logger.info(f"- WRG data written in {tt:.4f} minutes!")

    def execute(self, max_workers=None, buffer=0.2):
        """
        Create WRG file from h5_file

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of workers to use, by default None
        buffer : float, optional
            Buffer size in %, by default 0.2
        """
        try:
            logger.info(f"Creating {self._wrg_file} from {self._h5_file} at {self._hub_height}m...")
            wrg_stats = self.compute_wrg_stats(max_workers=max_workers)
            wrg_regrid = self.regrid_wrg(wrg_stats, buffer=buffer)
            self.write_wrg_file(wrg_regrid)
        except Exception:
            logger.exception(f"Failed to create WRG file {self._wrg_file} from {self._h5_file}")

    @classmethod
    def run(
        cls,
        h5_file,
        resolution,
        wrg_file=None,
        hub_height=100,
        bin_size=30,
        box_coords=None,
        max_workers=None,
        buffer=0.2,
    ):
        """
        Create .wrg file from Wind Resource h5 file at given hub-height

        Parameters
        ----------
        h5_file : str
            Path to h5 file
        resolution : int
            Pixel resolution in km
        wrg_file : str, optional
            Path to output file, if None use h5_file.replace(".h5", ".wrg")
            by default None
        hub_height : int, optional
            Hub height in meters, by default 100
        bin_size : int, optional
            Sector bin size in degrees, by default 30
        box_coords : tuple, optional
            (lat_lon_1, lat_lon_2) coordinates of the bounding box, by default None
        max_workers : int, optional
            Maximum number of workers to use, by default None
        buffer : float, optional
            Buffer size in %, by default 0.2
        """
        cls(
            h5_file, resolution, wrg_file=wrg_file, hub_height=hub_height, bin_size=bin_size, box_coords=box_coords
        ).execute(max_workers=max_workers, buffer=buffer)
