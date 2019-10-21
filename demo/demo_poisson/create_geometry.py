from dolfin import *
from geometry import MarkerFunctions2D, Geometry2D


# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0)))

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()
obstacle = Obstacle()

# Define mesh
mesh = UnitSquareMesh(64, 64)

# Markers (first index is the marker, second is the topological dimension)
markers = dict(default_domain=(0, 2),
                obstacle_domain=(1, 2),
                left_marker=(1, 1),
                top_marker=(2, 1),
                right_marker=(3, 1),
                bottom_marker=(4, 1))

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(markers['default_domain'][0])
obstacle.mark(domains, markers['obstacle_domain'][0])

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, markers['left_marker'][0])
top.mark(boundaries, markers['top_marker'][0])
right.mark(boundaries, markers['right_marker'][0])
bottom.mark(boundaries, markers['bottom_marker'][0])

marker_functions = MarkerFunctions2D(ffun=boundaries, cfun=domains)
geometry = Geometry2D(mesh, markers=markers,
                    markerfunctions=marker_functions)

geometry.save('poisson')
