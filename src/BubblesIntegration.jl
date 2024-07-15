module BubblesIntegration

import Meshes.Point3
import Base.length

export Point3

include("SurfaceIntergration.jl")
surface_integral = SurfaceIntegration.surface_integral
BubblePoint = SurfaceIntegration.BubblePoint
export surface_integral
export BubblePoint

include("VolumeIntegration.jl")

export VolumeIntegration
volume_integral = VolumeIntegration.volume_integral
export volume_integral

end
