module EnvelopeApproximation

include("BubbleBasics.jl")

include("Spaces.jl")

include("BoundaryConditions.jl")

include("BubblesEvolution.jl")

include("CFT/CFTInteface.jl")

include("CFT/ChebyshevCFT.jl")

include("CFT/QuadGKCFT.jl")

include("EnvelopeAnalysis.jl")

include("AngularIntegration/AngularIntegrationInterface.jl")

include("AngularIntegration/SphericalHarmonics.jl")

include("GravitationalWaves.jl")

include("Visualization.jl")

end
