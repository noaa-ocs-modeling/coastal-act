import argparse
from enum import Enum

from .interp import InterpInterface


class Dispatch(Enum):
    INTERP = InterpInterface


class Env(Enum):
    INTERP = 'interp'


def add_interp(subparsers):
    interp = subparsers.add_parser('interp')
    interp.add_argument('input_mesh_path', help="Path to input mesh.")
    interp.add_argument('output_mesh_path', help="Path to output mesh.")
    interp.add_argument(
        'DEM', nargs="+",
        help="List of at least one input DEM(s). "
        "These will be interpolated in the same order as "
        "they are given. Recommended is to pass them in order of lowest "
        "priority to highest priority."
        )
    interp.add_argument(
        "--overwrite", action="store_true",
        help="Used in case the output_mesh_path exists and the user wants to "
        "allow overwrite.")
    interp.add_argument(
        "--nprocs", type=int, help="Total number of processors to use. This "
        "algorithm can make use of virtual cores, so this value is not "
        "restricted to the number of physical cores or the amount of DEM's "
        "to be interpolated.")
    interp.add_argument(
        "--chunk-size", type=int, help="Useful when passing large rasters that"
        " do not fit in memory, it will subdivide the rasters into boxes "
        " of maximum pixel size chunk-size x chunk-size. If your job runs out "
        "of memory, try using --chunk-size=3000")
    interp.add_argument(
        "--crs",
        help="Input mesh CRS. Output will have the same CRS as the input.")
    interp.add_argument("--verbose", action="store_true")
    interp.add_argument("--use-anti-aliasing", action="store_true")
    interp.add_argument(
        "--anti-aliasing-method",
        default="reuse",
        choices=['reuse', 'fv']
        )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    add_interp(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()
    Dispatch[Env(args.mode).name].value(args)


if __name__ == '__main__':
    main()
