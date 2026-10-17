import numpy as np
from wave_sim2d.config import Grid
from wave_sim2d.geometry import polygon_mask,lens_vertices


def test_bounding_box_raster_preserves_cell_boundary_and_curved_geometry():
    g=Grid(96,80,.125,.0625,0)
    for vertices in (((0,0),(4,0),(4,3),(0,3)),((1.0625,1.0625),(6.0625,1.1875),(6,1.5),(1,1.3)),((1,1),(8,1),(8,2),(2,2),(2,7),(1,7)),lens_vertices(6,5,4,3,.4)):
        x,y=np.meshgrid((np.arange(g.nx)+.5)*g.dx,(np.arange(g.ny)+.5)*g.dx)
        expected=np.zeros(g.shape,dtype=bool)
        for (ax,ay),(bx,by) in zip(vertices,vertices[1:]+vertices[:1]):
            if ay!=by:expected^=((ay>y)!=(by>y))&(x<(bx-ax)*(y-ay)/(by-ay)+ax)
        np.testing.assert_array_equal(polygon_mask(g,vertices),expected)
