r"""
Axial and radial displaced metacone component
---------------------------------------------

.. topic:: Create a 3d mesh for a metacone component out of hexahedrons.

   * create a :class:`~felupe.MeshContainer` for meshes associated to two materials
"""

import felupe as fem
import numpy as np
import pypardiso

layers = [2, 11, 2, 11, 2]
lines = [
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2, axis=1),
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(4.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(6.5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(7, axis=1),
]
faces = fem.MeshContainer(
    [
        first.fill_between(second, n=n)
        for first, second, n in zip(lines[:-1], lines[1:], layers)
    ]
)
point = lambda m: m.points[np.unique(m.cells)].mean(axis=0) - np.array([2, 0])
mask = lambda m: np.unique(m.cells)
kwargs = dict(axis=1, exponent=2.5, normalize=True)
faces.points[:] = (
    faces[1]
    .add_runouts([3], centerpoint=point(faces[1]), mask=mask(faces[1]), **kwargs)
    .points
)
faces.points[:] = (
    faces[3]
    .add_runouts([7], centerpoint=point(faces[3]), mask=mask(faces[3]), **kwargs)
    .points
)
faces.points[:21] = faces.points[21 : 2 * 21]
faces.points[-21:] = faces.points[-2 * 21 : -21]
faces.points[:] = faces[0].rotate(15, axis=2).points
faces[0].y[:21] = 2
faces[-1].y[-21:] = 8.5
container = fem.MeshContainer([faces.stack([1, 3]), faces.stack([0, 2, 4])])
container.imshow(colors=[None, "white"], show_edges=True, opacity=1)

container = fem.MeshContainer(
    [
        m.revolve(n=37, phi=180).rotate(-90, axis=1).rotate(-90, axis=2)
        for m in container
    ],
    merge=True,
    decimals=4,
)
container.imshow(colors=[None, "white"], show_edges=True, opacity=1)

mesh = container.stack()
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

regions = [fem.RegionHexahedron(m) for m in container]
fields = [fem.FieldsMixed(r, n=1) for r in regions]

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, axis=2, sym=(0, 1, 0))
solids = [
    fem.SolidBodyNearlyIncompressible(fem.NeoHooke(mu=1), fields[0], bulk=200),
    fem.SolidBody(fem.LinearElasticLargeStrain(E=2.1e5, nu=0.3), fields[1]),
]

move = fem.math.linsteps([0, -1.5], num=3)
ramp = {boundaries["move"]: move}
step = fem.Step(
    items=[
        *solids,
    ],
    ramp=ramp,
    boundaries=boundaries,
)

job = fem.Job(steps=[step])
job.evaluate(x0=field, tol=1e-1, solver=pypardiso.spsolve, parallel=True)

boundaries, loadcase = fem.dof.shear(
    field,
    axes=(0, 2),
    moves=(0, 0, -1.5),
    sym=True,
)

move = fem.math.linsteps([0, 0.5], num=5)
ramp = {boundaries["move"]: move}
step = fem.Step(
    items=[
        *solids,
    ],
    ramp=ramp,
    boundaries=boundaries,
)

job = fem.Job(steps=[step])
job.evaluate(x0=field, tol=1e-2, solver=pypardiso.spsolve, parallel=True)

plotter = fields[1].plot(show_undeformed=False, color="white", show_edges=False)
fields[0].plot(
    "Principal Values of Logarithmic Strain",
    show_undeformed=False,
    project=fem.topoints,
    plotter=plotter,
).show()