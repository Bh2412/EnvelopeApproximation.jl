module EnvelopeApproximation

include("NucleationProcess/BubbleBasics.jl")

include("API/BoundaryConditions.jl")

include("API/Spaces.jl")

include("NucleationProcess/BubblesEvolution.jl")

include("CFT/CFTInterface.jl")

include("CFT/ChebyshevCFT.jl")

include("CFT/QuadGKCFT.jl")

include("EnvelopeAnalysis/EnvelopeAnalysis.jl")

include("StressEnergyTensorComponents.jl")

include("StressEnergyTensorContractions.jl")

include("API/TemporalWeights.jl")

include("RayCastingKernels/Kernels.jl")

include("RayCastingEnvelopeIntegration/EnvelopeIntegration.jl")

include("RayCastingKernels/TrueVacuumVolume.jl")

include("RayCastingKernels/StressTensor.jl")

include("RayCastingStressTensor.jl")

include("TwoPointStressEnergyTensor.jl")

end
