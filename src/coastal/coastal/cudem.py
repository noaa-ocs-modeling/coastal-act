# NCEI - Continuously Updated DEMs
# from argparse import Namespace
# from enum import Enum
import os
import pathlib
from typing import Union

import geopandas as gpd  # type: ignore[import]


TILE_INDEX_FILENAME = 'tileindex_NCEI_ninth_Topobathy_2014'

TILE_INDEX_URL = "https://coast.noaa.gov/htdata/raster2/elevation/" \
                 "NCEI_ninth_Topobathy_2014_8483/" \
                 f"{TILE_INDEX_FILENAME}.zip"


class CudemSync:

    def __init__(self):
        self.tile_index = gpd.read_file(TILE_INDEX_URL)

    def sync(self, path: Union[str, os.PathLike]):
        path = pathlib.Path(path)
        inventory = path.glob('*')
        # for tile in self.tile_index.itertuples():
        #     print(tile)


# class CudemActionDispatch(Enum):
#     SYNC = CudemSync


# class CudemActions(Enum):
#     SYNC = 'sync'


# class CudemInterface:

#     def __init__(self, args: Namespace):
#         dispatch = CudemActionDispatch[CudemActions(args.action).name].value()
#         if isinstance(dispatch, CudemSync):
#             dispatch.sync(args.cache_dir)
