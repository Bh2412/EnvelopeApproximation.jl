begin
    # 1. Imports
    using EnvelopeApproximation
    using EnvelopeApproximation.GravitationalWaves
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    using EnvelopeApproximation.Spaces
    using EnvelopeApproximation.BoundaryConditions
    using StaticArrays
    using Measurements
    using Random
    using StableRNGs
end

begin
    # 2. Configuration Parameters
    r_bubble = 5.0          # Bubble radius (time of interest)
    d_sep    = 6.0          # Separation distance
    L_box    = 40.0         # Box size
    N_samples = 200_000     # MC samples
    
    # Wavenumbers k to evaluate
    ks = collect(LinRange(0.1, 2.0, 10))
end

begin
    # 3. System Setup (Two Bubbles)
    # Define sites centered at ±d/2 along z
    b1_site = Point3(0.0, 0.0, -d_sep/2)
    b2_site = Point3(0.0, 0.0,  d_sep/2)
    
    # Create mock nucleation data (nucleated at t=0)
    nucleations = [
        (time=0.0, site=b1_site),
        (time=0.0, site=b2_site)
    ]
    
    # Create Snapshot and Space
    snapshot = BubblesSnapShot(nucleations, r_bubble * 2.0)
    space    = BoxSpace(L_box, Point3(0.0, 0.0, 0.0))
end

begin
    # 4. Computation using GravitationalWaves Interface
    println("Running Monte Carlo computation via GravitationalWaves interface...")
    
    # Define the Strategy
    mc_strategy = MonteCarloSampling(N=N_samples, rng=StableRNG(1))
    
    # Call the high-level Π function
    # Note: We pass 'ks' to the argument 'ωs' as they are equivalent in this context (c=1)
    pi_results = Π(
        r_bubble,      # t1
        r_bubble,      # t2
        ks,            # ωs (k vector)
        snapshot,
        space,
        Periodic(),    # Boundary Condition
        mc_strategy;   # The Strategy
        ΔV=1.0
    )
end
