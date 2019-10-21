import matplotlib.pyplot as plt
import dolfin
import pulse
from geometry import Geometry, example_meshes


geo = Geometry.from_file(example_meshes['simple_ellipsoid'])
# geo = pulse.Geometry.from_file(pulse.mesh_paths['simple_ellipsoid'])

activation = dolfin.Function(dolfin.FunctionSpace(geo.mesh, "R", 0))
activation.assign(dolfin.Constant(0.2))
matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(activation=activation,
                                parameters=matparams,
                                f0=geo.f0,
                                s0=geo.s0,
                                n0=geo.n0)

# LV Pressure
lvp = dolfin.Constant(1.0)
lv_marker = geo.markers['ENDO'][0]
lv_pressure = pulse.NeumannBC(traction=lvp,
                              marker=lv_marker, name='lv')
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [pulse.RobinBC(value=dolfin.Constant(base_spring),
                          marker=geo.markers["BASE"][0])]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(V.sub(0),
                            dolfin.Constant(0.0),
                            geo.ffun, geo.markers["BASE"][0])
    return bc


dirichlet_bc = [fix_basal_plane]
# You can also use a built in function for this
# from functools import partial
# dirichlet_bc = partial(pulse.mechanicsproblem.dirichlet_fix_base_directional,
#                        ffun=geometry.ffun,
#                        marker=geometry.markers["BASE"][0])

# Collect boundary conditions
bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                               neumann=neumann_bc,
                               robin=robin_bc)

# Create the problem
problem = pulse.MechanicsProblem(geo, material, bcs)

# Solve the problem
problem.solve()

# Get the solution
u, p = problem.state.split(deepcopy=True)

# Move mesh accoring to displacement
u_int = dolfin.interpolate(u,
                           dolfin.VectorFunctionSpace(geo.mesh, "CG", 1))
mesh = dolfin.Mesh(geo.mesh)
dolfin.ALE.move(mesh, u_int)

# Plot the result on to of the original
dolfin.plot(geo.mesh, alpha=0.1, edgecolor='k', color='w')
dolfin.plot(mesh, color="r")

ax = plt.gca()
ax.view_init(elev=-67, azim=-179)
ax.set_axis_off()
plt.show()
