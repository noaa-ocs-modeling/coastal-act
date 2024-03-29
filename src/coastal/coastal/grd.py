from abc import ABC
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
import numbers
import os
import pathlib
from typing import Union, Sequence, Hashable, List, Dict, TextIO
import warnings


import geopandas as gpd  # type: ignore[import]
from matplotlib.cm import ScalarMappable  # type: ignore[import]
from matplotlib.collections import PolyCollection  # type: ignore[import]
from matplotlib.colors import LinearSegmentedColormap, Normalize  # type: ignore[import]  # noqa: E501
from matplotlib.path import Path  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib import rcParams
from matplotlib.tri import Triangulation  # type: ignore[import]
from matplotlib.transforms import Bbox  # type: ignore[import]
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable  # type: ignore[import]  # noqa: E501
import numpy as np  # type: ignore[import]
from pyproj import Transformer, CRS  # type: ignore[import]
from pyproj.exceptions import CRSError  # type: ignore[import]
from shapely.geometry import (  # type: ignore[import]  # noqa: E501
    MultiPolygon,
    Polygon,
    LineString,
    LinearRing,
)


def buffer_to_dict(buf: TextIO):
    description = buf.readline()
    NE, NP = map(int, buf.readline().split())
    nodes = {}
    for _ in range(NP):
        line = buf.readline().strip('\n').split()
        # Gr3/fort.14 format cannot distinguish between a 2D mesh with one
        # vector value (e.g. velocity, which uses 2 columns) or a 3D mesh with
        # one scalar value. This is a design problem of the mesh format, which
        # renders it ambiguous, and the main reason why the use of fort.14/grd
        # formats is discouraged, in favor of UGRID.
        # Here, we assume the input mesh is strictly a 2D mesh, and the data
        # that follows is an array of values.
        if len(line[3:]) == 1:
            nodes[line[0]] = [
                (float(line[1]), float(line[2])), float(line[3])]
        else:
            nodes[line[0]] = [
                (float(line[1]), float(line[2])),
                [float(line[i]) for i in range(3, len(line[3:]))]
            ]
    elements = {}
    for _ in range(NE):
        line = buf.readline().split()
        elements[line[0]] = line[2:]
    # Assume EOF if NOPE is empty.
    try:
        NOPE = int(buf.readline().split()[0])
    except IndexError:
        return {'description': description,
                'nodes': nodes,
                'elements': elements}
    # let NOPE=-1 mean an ellipsoidal-mesh
    # reassigning NOPE to 0 until further implementation is applied.
    boundaries: Dict = defaultdict(dict)
    _bnd_id = 0
    buf.readline()
    while _bnd_id < NOPE:
        NETA = int(buf.readline().split()[0])
        _cnt = 0
        boundaries[None][_bnd_id] = dict()
        boundaries[None][_bnd_id]['indexes'] = list()
        while _cnt < NETA:
            boundaries[None][_bnd_id]['indexes'].append(
                buf.readline().split()[0].strip())
            _cnt += 1
        _bnd_id += 1
    NBOU = int(buf.readline().split()[0])
    _nbnd_cnt = 0
    buf.readline()
    while _nbnd_cnt < NBOU:
        npts, ibtype = map(int, buf.readline().split()[:2])
        _pnt_cnt = 0
        if ibtype not in boundaries:
            _bnd_id = 0
        else:
            _bnd_id = len(boundaries[ibtype])
        boundaries[ibtype][_bnd_id] = dict()
        boundaries[ibtype][_bnd_id]['indexes'] = list()
        while _pnt_cnt < npts:
            line = buf.readline().split()
            if len(line) == 1:
                boundaries[ibtype][_bnd_id]['indexes'].append(line[0])
            else:
                index_construct = []
                for val in line:
                    if '.' in val:
                        continue
                    index_construct.append(val)
                boundaries[ibtype][_bnd_id]['indexes'].append(index_construct)
            _pnt_cnt += 1
        _nbnd_cnt += 1
    return {'description': description,
            'nodes': nodes,
            'elements': elements,
            'boundaries': boundaries}


def to_string(description, nodes, elements, boundaries=None, crs=None):
    """
    must contain keys:
        description
        vertices
        elements
        vertex_id
        element_id
        values
        boundaries (optional)
            indexes
    """
    NE, NP = len(elements), len(nodes)
    out = [f"{description}", f"{NE} {NP}"]
    # TODO: Probably faster if using np.array2string
    for id, (coords, values) in nodes.items():
        if isinstance(values, numbers.Number):
            values = [values]
        line = [f"{id}"]
        line.extend([f"{x:<.16E}" for x in coords])
        line.extend([f"{x:<.16E}" for x in values])
        out.append(" ".join(line))

    for id, element in elements.items():
        line = [f"{id}"]
        line.append(f"{len(element)}")
        line.extend([f"{e}" for e in element])
        out.append(" ".join(line))

    # ocean boundaries
    if boundaries is not None:
        out.append(f"{len(boundaries[None]):d} "
                   "! total number of ocean boundaries")
        # count total number of ocean boundaries
        _sum = 0
        for bnd in boundaries[None].values():
            _sum += len(bnd['indexes'])
        out.append(f"{int(_sum):d} ! total number of ocean boundary nodes")
        # write ocean boundary indexes
        for i, boundary in boundaries[None].items():
            out.append(f"{len(boundary['indexes']):d}"
                       f" ! number of nodes for ocean_boundary_{i}")
            for idx in boundary['indexes']:
                out.append(f"{idx}")
    else:
        out.append("0 ! total number of ocean boundaries")
        out.append("0 ! total number of ocean boundary nodes")
    # remaining boundaries
    _cnt = 0
    for key in boundaries:
        if key is not None:
            for bnd in boundaries[key]:
                _cnt += 1
    out.append(f"{_cnt:d}  ! total number of non-ocean boundaries")
    # count remaining boundary nodes
    _cnt = 0
    for ibtype in boundaries:
        if ibtype is not None:
            for bnd in boundaries[ibtype].values():
                _cnt += np.asarray(bnd['indexes']).size
    out.append(f"{_cnt:d} ! Total number of non-ocean boundary nodes")
    # all additional boundaries
    for ibtype, boundaries in boundaries.items():
        if ibtype is None:
            continue
        for id, boundary in boundaries.items():
            line = [
                f"{len(boundary['indexes']):d}",
                f"{ibtype}",
                f"! boundary {ibtype}:{id}"]
            out.append(' '.join(line))
            for idx in boundary['indexes']:
                out.append(f"{idx}")
    return "\n".join(out)


def read(resource: Union[str, os.PathLike], boundaries: bool = True, crs=True):
    """Converts a file-like object representing a grd-formatted unstructured
    mesh into a python dictionary:

    Args:
        resource: Path to file on disk or file-like object such as
            :class:`io.StringIO`
    """
    resource = pathlib.Path(resource)
    with open(resource, 'r') as stream:
        grd = buffer_to_dict(stream)
    if boundaries is False:
        grd.pop('boundaries', None)
    if crs is True:
        crs = None
    if crs is None:
        for try_crs in grd['description'].split():
            try:
                crs = CRS.from_user_input(try_crs)
                break
            except CRSError:
                pass
    if crs is None:
        warnings.warn(f'File {str(resource)} does not contain CRS '
                      'information.')
    if crs is not False:
        grd.update({'crs': crs})
    return grd


def write(grd, path, overwrite=False):
    path = pathlib.Path(path)
    if path.is_file() and not overwrite:
        raise Exception('File exists, pass overwrite=True to allow overwrite.')
    with open(path, 'w') as f:
        f.write(to_string(**grd))


def get_axes(axes, figsize=None, subplot=111):
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    if axes is None:
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(subplot)
    return axes


def figure(f):
    def decorator(*argv, **kwargs):
        axes = get_axes(
            kwargs.get('axes', None),
            kwargs.get('figsize', None)
            )
        kwargs.update({'axes': axes})
        axes = f(*argv, **kwargs)
        if kwargs.get('show', False):
            axes.axis('scaled')
            plt.show()
        return axes
    return decorator


class Description:

    def __set__(self, obj, description):
        if description is None:
            description = ''
        if not isinstance(description, str):
            raise TypeError('Argument description must be a string, not '
                            f'type {type(description)}.')
        obj.__dict__['description'] = description.replace('\n', ' ')

    def __get__(self, obj, val):
        return obj.__dict__['description']


class Nodes:

    def __set__(self, obj: "Grd", nodes: Dict[Hashable, List[List]]):
        """Setter for the nodes attribute.

        Argument nodes must be of the form:
            {id: [(x0, y0), z0]}
            or
            {id: [(x0, y0), [z0, ..., zn]}

        Gr3 format is assumed to be exclusively a 2D format that can hold
        triangles or quads.

        """
        for coords, _ in nodes.values():
            if len(coords) != 2:
                raise ValueError(
                    'Coordinate vertices for a grd type must be 2D, but got '
                    f'coordinates {coords}.')
        obj.__dict__['nodes'] = nodes
        self.grd = obj
        # self.ball = NodeBall(self.grd)

        obj.__dict__['id_to_index'] = {
            self.id()[i]: i for i in range(len(self.id()))}

        obj.__dict__['index_to_id'] = {
            i: self.id()[i] for i in range(len(self.id()))}

    def __call__(self):
        return self.grd.__dict__['nodes']

    @lru_cache(maxsize=1)
    def id(self):
        return list(self().keys())

    @lru_cache(maxsize=1)
    def index(self):
        return np.arange(len(self()))

    @lru_cache(maxsize=1)
    def coord(self):
        return np.array([coords for coords, _ in self().values()])

    @lru_cache(maxsize=1)
    def values(self):
        return np.array([val for _, val in self().values()])

    def get_index_by_id(self, id: Hashable):
        return self.grd.__dict__['id_to_index'][id]

    def get_id_by_index(self, index: int):
        return self.grd.__dict__['index_to_id'][index]

    def get_indexes_around_index(self, index):
        indexes_around_index = self.__dict__.get('indexes_around_index')
        if indexes_around_index is None:
            def append(geom):
                for simplex in geom:
                    for i, j in permutations(simplex, 2):
                        indexes_around_index[i].add(j)
            indexes_around_index = defaultdict(set)
            append(self.grd.elements.triangles())
            append(self.grd.elements.quads())
            self.__dict__['indexes_around_index'] = indexes_around_index
        return list(indexes_around_index[index])


class Elements:

    def __set__(self, obj: "Grd", elements: Dict[Hashable, Sequence]):
        if not isinstance(elements, dict):
            raise TypeError('Argument elements must be a dict.')
        vertex_id_set = set(obj.nodes.id())
        for i, element in enumerate(elements):
            if not isinstance(element, Sequence):
                raise TypeError(f'Element with index {i} of the elements '
                                f'argument must be of type {Sequence}, not '
                                f'type {type(element)}.')
            if not set(element).issubset(vertex_id_set):
                ValueError(f'Element with index {i} is not a subset of the '
                           "coordinate id's.")
        obj.__dict__['elements'] = elements
        self.grd = obj

    def __call__(self):
        return self.grd.__dict__["elements"]

    @lru_cache(maxsize=1)
    def id(self):
        return list(self().keys())

    @lru_cache(maxsize=1)
    def index(self):
        return np.arange(len(self()))

    @lru_cache(maxsize=1)
    def array(self):
        rank = int(max(map(len, self().values())))
        array = np.full((len(self()), rank), -1)
        for i, element in enumerate(self().values()):
            row = np.array(list(map(self.grd.nodes.get_index_by_id, element)))
            array[i, :len(row)] = row
        return np.ma.masked_equal(array, -1)

    @lru_cache(maxsize=1)
    def triangles(self):
        return np.array(
            [list(map(self.grd.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 3])

    @lru_cache(maxsize=1)
    def quads(self):
        return np.array(
            [list(map(self.grd.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 4])

    @lru_cache(maxsize=1)
    def triangulation(self):
        triangles = self.triangles().tolist()
        for quad in self.quads():
            triangles.append([quad[0], quad[1], quad[3]])
            triangles.append([quad[1], quad[2], quad[3]])
        return Triangulation(
            self.grd.nodes.coord()[:, 0],
            self.grd.nodes.coord()[:, 1],
            triangles)

    def geodataframe(self):
        data = []
        for id, element in self().items():
            data.append({
                'geometry': Polygon(
                    self.grd.nodes.coord()[list(
                        map(self.grd.nodes.get_index_by_id, element))]),
                'id': id})
        return gpd.GeoDataFrame(data, crs=self.grd.crs)


class Crs:

    def __set__(self, obj, val):
        if val is not None:
            obj.__dict__['crs'] = CRS.from_user_input(val)

    def __get__(self, obj, val):
        return obj.__dict__.get('crs')


class Edges:

    def __init__(self, grd: "Grd"):
        self.grd = grd

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        data = []
        for ring in self.grd.hull.rings().itertuples():
            coords = ring.geometry.coords
            for i in range(1, len(coords)):
                data.append({
                    "geometry": LineString([coords[i-1], coords[i]]),
                    "bnd_id": ring.bnd_id,
                    "type": ring.type})
        return gpd.GeoDataFrame(data, crs=self.grd.crs)

    @lru_cache(maxsize=1)
    def exterior(self):
        return self().loc[self()['type'] == 'exterior']

    @lru_cache(maxsize=1)
    def interior(self):
        return self().loc[self()['type'] == 'interior']


class Rings:

    def __init__(self, grd: "Grd"):
        self.grd = grd

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        tri = self.grd.elements.triangulation()
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        boundary_edges = []
        for i, j in idxs:
            boundary_edges.append(
                (tri.triangles[i, j], tri.triangles[i, (j+1) % 3]))
        sorted_rings = sort_rings(edges_to_rings(boundary_edges),
                                  self.grd.nodes.coord())
        data = []
        for bnd_id, rings in sorted_rings.items():
            coords = self.grd.nodes.coord()[rings['exterior'][:, 0], :]
            geometry = LinearRing(coords)
            data.append({
                    "geometry": geometry,
                    "bnd_id": bnd_id,
                    "type": 'exterior'
                })
            for interior in rings['interiors']:
                coords = self.grd.nodes.coord()[interior[:, 0], :]
                geometry = LinearRing(coords)
                data.append({
                    "geometry": geometry,
                    "bnd_id": bnd_id,
                    "type": 'interior'
                })
        return gpd.GeoDataFrame(data, crs=self.grd.crs)

    def exterior(self):
        return self().loc[self()['type'] == 'exterior']

    def interior(self):
        return self().loc[self()['type'] == 'interior']


class Hull:

    def __init__(self, grd: "Grd"):
        self.grd = grd
        self.edges = Edges(grd)
        self.rings = Rings(grd)

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        data = []
        for bnd_id in np.unique(self.rings()['bnd_id'].tolist()):
            exterior = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'exterior')]
            interiors = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'interior')]
            data.append({
                    "geometry": Polygon(
                        exterior.iloc[0].geometry.coords,
                        [row.geometry.coords for _, row
                            in interiors.iterrows()]),
                    "bnd_id": bnd_id
                })
        return gpd.GeoDataFrame(data, crs=self.grd.crs)

    @lru_cache(maxsize=1)
    def exterior(self):
        data = []
        for exterior in self.rings().loc[
                self.rings()['type'] == 'exterior'].itertuples():
            data.append({"geometry": Polygon(exterior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.grd.crs)

    @lru_cache(maxsize=1)
    def interior(self):
        data = []
        for interior in self.rings().loc[
                self.rings()['type'] == 'interior'].itertuples():
            data.append({"geometry": Polygon(interior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.grd.crs)

    @lru_cache(maxsize=1)
    def implode(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"geometry": MultiPolygon([polygon.geometry for polygon
                                       in self().itertuples()])},
            crs=self.grd.crs)

    def multipolygon(self) -> MultiPolygon:
        return self.implode().iloc[0].geometry


class Grd(ABC):

    _description = Description()
    _nodes = Nodes()
    _elements = Elements()
    _crs = Crs()

    def __init__(self, nodes, elements=None, description=None, crs=None):
        self._nodes = nodes
        self._elements = elements
        self._description = description
        self._crs = crs
        self._hull = Hull(self)

    def __str__(self):
        return to_string(**self.to_dict())

    def to_dict(self):
        return {
            "description": self.description,
            "nodes": self.nodes(),
            "elements": self.elements(),
            "crs": self.crs}

    def write(self, path, overwrite=False):
        write(self.to_dict(), path, overwrite)

    def get_x(self, crs: Union[CRS, str] = None):
        return self.get_xy(crs)[:, 0]

    def get_y(self, crs: Union[CRS, str] = None):
        return self.get_xy(crs)[:, 1]

    def get_xy(self, crs: Union[CRS, str] = None):
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if not crs.equals(self.crs):
                transformer = Transformer.from_crs(
                    self.crs, crs, always_xy=True)
                x, y = transformer.transform(self.x, self.y)
                return np.vstack([x, y]).T
        return np.vstack([self.x, self.y]).T

    def get_bbox(self, crs: Union[CRS, str] = None) -> Bbox:
        vertices = self.get_xy(crs)
        x0 = np.min(vertices[:, 0])
        x1 = np.max(vertices[:, 0])
        y0 = np.min(vertices[:, 1])
        y1 = np.max(vertices[:, 1])
        return Bbox([[x0, y0], [x1, y1]])

    def transform_to(self, dst_crs):
        """Transforms coordinate system of mesh in-place.
        """
        dst_crs = CRS.from_user_input(dst_crs)
        if not self.crs.equals(dst_crs):
            xy = self.get_xy(dst_crs)
            self.nodes.coord.cache_clear()
            self._nodes = {self.nodes.id()[i]:
                           (coord.tolist(), self.nodes.values()[i])
                           for i, coord in enumerate(xy)}
            self._crs = dst_crs

    def vertices_around_vertex(self, index):
        return self.nodes.vertices_around_vertex(index)

    @classmethod
    def open(cls, file: Union[str, os.PathLike],
             crs: Union[str, CRS] = None):
        return cls(**read(pathlib.Path(file), boundaries=False))

    @figure
    def tricontourf(self, axes=None, show=True, figsize=None, **kwargs):
        if len(self.triangles) > 0:
            axes.tricontourf(self.triangulation, self.values, **kwargs)
        return axes

    @figure
    def tripcolor(self, axes=None, show=True, figsize=None, **kwargs):
        if len(self.triangles) > 0:
            axes.tripcolor(self.triangulation, self.values, **kwargs)
        return axes

    @figure
    def triplot(
        self,
        axes=None,
        show=False,
        figsize=None,
        linewidth=0.07,
        color='black',
        **kwargs
    ):
        if len(self.triangles) > 0:
            kwargs.update({'linewidth': linewidth})
            kwargs.update({'color': color})
            axes.triplot(self.triangulation, **kwargs)
        return axes

    @figure
    def quadplot(
        self,
        axes=None,
        show=False,
        figsize=None,
        facecolor='none',
        edgecolor='k',
        linewidth=0.07,
        **kwargs
    ):
        if len(self.quads) > 0:
            pc = PolyCollection(
                self.coords[self.quads],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.07,
            )
            axes.add_collection(pc)
        return axes

    @figure
    def quadface(
        self,
        axes=None,
        show=False,
        figsize=None,
        **kwargs
    ):
        if len(self.quads) > 0:
            pc = PolyCollection(
                self.coords[self.quads],
                **kwargs
            )
            quad_value = np.mean(self.values[self.quads], axis=1)
            pc.set_array(quad_value)
            axes.add_collection(pc)
        return axes

    @figure
    def plot_wireframe(self, axes=None, show=False, **kwargs):
        axes = self.triplot(axes=axes, **kwargs)
        axes = self.quadplot(axes=axes, **kwargs)
        return axes

    @property
    def nodes(self):
        return self._nodes

    @property
    def coords(self):
        return self._nodes.coord()

    @property
    def vertices(self):
        return self._nodes.coord()

    @property
    def vertex_id(self):
        return self._nodes.id()

    @property
    def elements(self):
        return self._elements

    @property
    def element_id(self):
        return self._elements.id()

    @property
    def values(self):
        return self._nodes.values()

    @property
    def description(self):
        return self._description

    @property
    def crs(self):
        return self._crs

    @property
    def x(self):
        return self._nodes.coord()[:, 0]

    @property
    def y(self):
        return self._nodes.coord()[:, 1]

    @property
    def hull(self):
        return self._hull

    @property
    def triangles(self):
        return self._elements.triangles()

    @property
    def quads(self):
        return self._elements.quads()

    @property
    def triangulation(self):
        return self._elements.triangulation()

    @property
    def bbox(self):
        return self.get_bbox()


def edges_to_rings(edges):
    if len(edges) == 0:
        return edges
    # start ordering the edges into linestrings
    edge_collection = list()
    ordered_edges = [edges.pop(-1)]
    e0, e1 = [list(t) for t in zip(*edges)]
    while len(edges) > 0:
        if ordered_edges[-1][1] in e0:
            idx = e0.index(ordered_edges[-1][1])
            ordered_edges.append(edges.pop(idx))
        elif ordered_edges[0][0] in e1:
            idx = e1.index(ordered_edges[0][0])
            ordered_edges.insert(0, edges.pop(idx))
        elif ordered_edges[-1][1] in e1:
            idx = e1.index(ordered_edges[-1][1])
            ordered_edges.append(
                list(reversed(edges.pop(idx))))
        elif ordered_edges[0][0] in e0:
            idx = e0.index(ordered_edges[0][0])
            ordered_edges.insert(
                0, list(reversed(edges.pop(idx))))
        else:
            edge_collection.append(tuple(ordered_edges))
            idx = -1
            ordered_edges = [edges.pop(idx)]
        e0.pop(idx)
        e1.pop(idx)
    # finalize
    if len(edge_collection) == 0 and len(edges) == 0:
        edge_collection.append(tuple(ordered_edges))
    else:
        edge_collection.append(tuple(ordered_edges))
    return edge_collection


def sort_rings(index_rings, vertices):
    """Sorts a list of index-rings.

    Takes a list of unsorted index rings and sorts them into an "exterior" and
    "interior" components. Any doubly-nested rings are considered exterior
    rings.

    TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
    probably be optimized using shapely.
    """

    # sort index_rings into corresponding "polygons"
    areas = list()
    for index_ring in index_rings:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_rings.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_rings = dict()
    _index_rings[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
    }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_rings) > 0:
        # find all internal rings
        potential_interiors = list()
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = list()
        for i, p_interior in reversed(
                list(enumerate(potential_interiors))):
            _p_interior = index_rings[p_interior]
            check = [index_rings[k]
                     for j, k in
                     reversed(list(enumerate(potential_interiors)))
                     if i != j]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                _path = Path(vertices[e0 + [e0[0]], :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            _index_rings[_id]['interiors'].append(
                np.asarray(index_rings.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_rings) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_rings.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_rings[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
            }
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return _index_rings


def signed_polygon_area(vertices):
    # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(vertices)  # of vertices
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
        return area / 2.0


class BoundariesDescriptor:

    def __set__(self, obj, boundaries: Union[dict, None]):
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        if boundaries is not None:
            for ibtype, bnds in boundaries.items():
                if ibtype is None:
                    for id, data in bnds.items():
                        indexes = list(map(obj.nodes.get_index_by_id,
                                       data['indexes']))
                        ocean_boundaries.append({
                            'id': id,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(obj.vertices[indexes])
                            })

                elif str(ibtype).endswith('1'):
                    for id, data in bnds.items():
                        indexes = list(map(obj.nodes.get_index_by_id,
                                       data['indexes']))
                        interior_boundaries.append({
                            'id': id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(obj.vertices[indexes])
                            })
                else:
                    for id, data in bnds.items():
                        _indexes = np.array(data['indexes'])
                        if _indexes.ndim > 1:
                            # ndim > 1 implies we're dealing with an ADCIRC
                            # mesh that includes boundary pairs, such as weir
                            new_indexes = []
                            for i, line in enumerate(_indexes.T):
                                if i % 2 != 0:
                                    new_indexes.extend(np.flip(line))
                                else:
                                    new_indexes.extend(line)
                            _indexes = np.array(new_indexes).flatten()
                        else:
                            _indexes = _indexes.flatten()
                        indexes = list(map(obj.nodes.get_index_by_id,
                                       _indexes))

                        land_boundaries.append({
                            'id': id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(obj.vertices[indexes])
                            })

        self.ocean = gpd.GeoDataFrame(ocean_boundaries)
        self.land = gpd.GeoDataFrame(land_boundaries)
        self.interior = gpd.GeoDataFrame(interior_boundaries)
        self.boundaries = boundaries

    def __call__(self):
        return self.boundaries


class FixPointNormalize(Normalize):
    """
    This class is used for plotting. The reason it is declared here is that
    it is used by more than one submodule. In the future, this class will be
    native part of matplotlib. This definiton will be removed once the native
    matplotlib definition becomes available.
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.
    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue/turquise range.
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val=0.5,
                 clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent
        # the sealevel.
        self.col_val = col_val
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        if np.ma.is_masked(value) is False:
            value = np.ma.masked_invalid(value)
        return np.ma.masked_where(value.mask, np.interp(value, x, y))


def get_topobathy_kwargs(values, vmin, vmax, colors=256):
    vmin = np.min(values) if vmin is None else vmin
    vmax = np.max(values) if vmax is None else vmax
    if vmax <= 0.:
        cmap = plt.cm.seismic
        col_val = 0.
        levels = np.linspace(vmin, vmax, colors)
    else:
        wet_count = int(np.floor(
            colors*(float((values < 0.).sum()) / float(values.size))))
        col_val = float(wet_count)/colors
        dry_count = colors - wet_count
        colors_undersea = plt.cm.bwr(np.linspace(1., 0., wet_count))
        colors_land = plt.cm.terrain(np.linspace(0.25, 1., dry_count))
        colors = np.vstack((colors_undersea, colors_land))
        cmap = LinearSegmentedColormap.from_list('cut_terrain', colors)
        wlevels = np.linspace(vmin, 0.0, wet_count, endpoint=False)
        dlevels = np.linspace(0.0, vmax, dry_count)
        levels = np.hstack((wlevels, dlevels))
    if vmax > 0:
        norm = FixPointNormalize(
            sealevel=0.0,
            vmax=vmax,
            vmin=vmin,
            col_val=col_val
            )
    else:
        norm = None
    return {'cmap': cmap,
            'norm': norm,
            'levels': levels,
            'col_val': col_val,
            # 'extend': 'both'
            }


class Fort14(Grd):
    """
    Class that represents the unstructured planar mesh used by SCHISM.
    """
    _boundaries = BoundariesDescriptor()

    def __init__(self, *args, boundaries=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._boundaries = boundaries

    @staticmethod
    def open(path, crs=None):
        _grd = read(path, crs=crs)
        _grd['nodes'] = {id: (coords, -val) for id, (coords, val)
                         in _grd['nodes'].items()}
        return Fort14(**_grd)

    def to_dict(self):
        return {
            "description": self.description,
            "nodes": self.nodes(),
            "elements": self.elements(),
            "crs": self.crs,
            "boundaries": self.boundaries()}

    @figure
    def make_plot(
        self,
        axes=None,
        vmin=None,
        vmax=None,
        show=False,
        title=None,
        figsize=rcParams["figure.figsize"],
        extent=None,
        cbar_label=None,
        **kwargs
    ):
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        kwargs.update(**get_topobathy_kwargs(self.values, vmin, vmax))
        kwargs.pop('col_val')
        levels = kwargs.pop('levels')
        if vmin != vmax:
            self.tricontourf(
                axes=axes,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
        else:
            self.tripcolor(axes=axes, **kwargs)
        self.quadface(axes=axes, **kwargs)
        axes.axis('scaled')
        if extent is not None:
            axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=kwargs['cmap'])
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(
            mappable,
            cax=cax,
            orientation='horizontal'
        )
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        return axes

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def ocean_boundaries(self):
        return self.boundaries.ocean

    @property
    def land_boundaries(self):
        return self.boundaries.land

    @property
    def interior_boundaries(self):
        return self.boundaries.interior
