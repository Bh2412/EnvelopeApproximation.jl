module EnvelopeApproximation

include("BubbleBasics.jl")

include("BoundaryConditions.jl")

include("Spaces.jl")

include("BubblesEvolution.jl")

include("CFT/CFTInteface.jl")

include("CFT/ChebyshevCFT.jl")

include("CFT/QuadGKCFT.jl")

include("EnvelopeAnalysis.jl")

include("AngularIntegration/AngularIntegrationInterface.jl")

include("AngularIntegration/SphericalHarmonics.jl")

include("StressEnergyTensorComponents.jl")

include("StressEnergyTensorContractions.jl")

include("TemporalWeights.jl")

include("RayCastingStressEnergyTensor.jl")

include("GravitationalWaves.jl")

include("TwoPointStressEnergyTensor.jl")

end
