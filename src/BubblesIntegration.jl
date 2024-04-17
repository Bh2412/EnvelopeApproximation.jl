module BubblesIntegration

import Meshes.Point3
import Base.length

export Point3

include("SurfaceIntergration.jl")

export SurfaceIntegration
surface_integral = SurfaceIntegration.surface_integral
export surface_integral

include("VolumeIntegration.jl")

export VolumeIntegration
volume_integral = VolumeIntegration.volume_integral
export volume_integral

end
