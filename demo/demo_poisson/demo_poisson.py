import matplotlib.pyplot as plt
from dolfin import *
from geometry import MarkerFunctions2D, Geometry2D


geo = Geometry2D.from_file('poisson.h5')

# Define input data
a0 = Constant(1.0)
a1 = Constant(0.01)
g_L = Expression("- 10*exp(- pow(x[1] - 0.5, 2))", degree=2)
g_R = Constant(1.0)
f = Constant(1.0)

# Define function space and basis functions
V = FunctionSpace(geo.mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define Dirichlet boundary conditions at top and bottom boundaries
bcs = [DirichletBC(V, 5.0, geo.ffun, geo.markers['top_marker'][0]),
       DirichletBC(V, 0.0, geo.ffun, geo.markers['bottom_marker'][0])]

# Define new measures associated with the interior domains and
# exterior boundaries
dx = geo.dx
ds = geo.ds

# Define variational form
F = (inner(a0*grad(u), grad(v))*dx(geo.markers['default_domain'][0]) +\
        inner(a1*grad(u), grad(v))*dx(geo.markers['obstacle_domain'][0])\
        - g_L*v*ds(geo.markers['left_marker'][0]) - g_R*v*ds(geo.markers['right_marker'][0])\
        - f*v*dx(geo.markers['default_domain'][0]) - f*v*dx(geo.markers['obstacle_domain'][0]))

# Separate left and right hand sides of equation
a, L = lhs(F), rhs(F)

# Solve problem
u = Function(V)
solve(a == L, u, bcs)

# Evaluate integral of normal gradient over top boundary
n = FacetNormal(geo.mesh)
m1 = dot(grad(u), n)*ds(geo.markers['top_marker'][0])
v1 = assemble(m1)
print("\int grad(u) * n ds(2) = ", v1)

# Evaluate integral of u over the obstacle
m2 = u*dx(geo.markers['obstacle_domain'][0])
v2 = assemble(m2)
print("\int u dx(1) = ", v2)

# Plot solution
plt.figure()
plot(u, title="Solution u")

# Plot solution and gradient
plt.figure()
plot(grad(u), title="Projected grad(u)")

# Show plots
plt.show()
