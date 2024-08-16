module BubblesIntegration

import Meshes.Point3
import Base.length

export Point3

include("SurfaceIntergration.jl")
surface_integral = SurfaceIntegration.surface_integral
export surface_integral

end
