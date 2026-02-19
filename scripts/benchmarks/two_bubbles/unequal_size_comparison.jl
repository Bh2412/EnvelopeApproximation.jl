# ==============================================================================
# UNEQUAL BUBBLES COMPARISON SCRIPT
# ==============================================================================

begin
    # 1. Imports and Configuration
    using Printf
    using Random
    using Measurements
    using StaticArrays
    using LinearAlgebra
    using QuadGK
    import CairoMakie as CM

    # Numeric Library
    using EnvelopeApproximation
    using EnvelopeApproximation.GravitationalWaves
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    using EnvelopeApproximation.Spaces
    using EnvelopeApproximation.BoundaryConditions

    include("Analytic.jl") # Contains the analytic T_ij and Π calculations
    include("Numeric.jl")

    # System Parameters
    # We want two bubbles with radii R1 and R2 at the observation time t_obs
    R1 = 5.0
    R2 = 3.0
    d_sep = 6.0
    
    # We set observation time large enough so nucleation times are positive (optional, but cleaner)
    t_obs = 10.0 
    
    # Nucleation times derived from R = t - t_nuc (assuming v_w = 1)
    tn1 = t_obs - R1
    tn2 = t_obs - R2
    
    # Simulation Parameters
    L_box = 50.0
    vol = L_box^3
    N_samples = 20_000_000
    ks = collect(10 .^ LinRange(log10(0.1), log10(5.0), 100))

    println("--- Configuration ---")
    println("  R1 = $R1, R2 = $R2 (at t = $t_obs)")
    println("  Separation d = $d_sep")
    println("  Nucleation Times: t_n1 = $tn1, t_n2 = $tn2")
    println("  Box Volume = $vol")
end

begin
    # 2. NUMERIC CALCULATION (Monte Carlo)
    println("\n[Numeric] Running Monte Carlo Simulation...")

    # Define Geometry
    b1_site = Point3(0.0, 0.0, -d_sep/2)
    b2_site = Point3(0.0, 0.0,  d_sep/2)

    nucleations = [
        (time=tn1, site=b1_site),
        (time=tn2, site=b2_site)
    ]

    # Create Snapshot
    # Note: The snapshot is just a container; the state is evaluated at t_obs in the Π call
    snapshot = BubblesSnapShot(nucleations, t_obs * 2.0)
    space = BoxSpace(L_box, Point3(0.0, 0.0, 0.0)) # Cubic box

    # Define Strategy
    strategy = MonteCarloSampling(N=N_samples, rng=StableRNG(42))

    # Compute Π
    # We evaluate at equal times t1=t2=t_obs
    pi_numeric = Π(t_obs, t_obs, ks, snapshot, space, Periodic(), strategy; ΔV=1.0)
    
    println("  Done.")
end

begin
    # 3. ANALYTIC CALCULATION (Exact Integration)
    println("\n[Analytic] Computing Exact Reference...")

    pi_analytic = Float64[]
    for k in ks
        val = compute_analytic_isotropic_Π(k, R1, R2, d_sep, vol)
        push!(pi_analytic, val)
    end
    println("  Done.")
end

begin
    # 4. PLOTTING
    println("\n[Plotting] Generating Comparison Plot...")
    
    # Extract numeric values
    num_vals = Measurements.value.(pi_numeric)
    num_errs = Measurements.uncertainty.(pi_numeric)
    
    # Filter for log plot (remove non-positive if any)
    mask = (num_vals .> 0)
    ks_p = ks[mask]
    vals_p = num_vals[mask]
    errs_p = num_errs[mask]
    ana_p = pi_analytic[mask]

    fig = CM.Figure(size = (900, 700), fontsize = 20)
    
    # -- Top Panel: Spectrum --
    ax1 = CM.Axis(fig[1, 1],
        title = "Unequal Bubbles (R1=$R1, R2=$R2, d=$d_sep)",
        ylabel = "Π(k)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true,
        yminorticksvisible = true
    )

    CM.lines!(ax1, ks_p, ana_p, color = :red, linewidth = 2, label = "Analytic")
    CM.scatter!(ax1, ks_p, vals_p, color = :blue, label = "Monte Carlo")
    CM.errorbars!(ax1, ks_p, vals_p, errs_p, color = :blue)
    
    CM.axislegend(ax1, position = :lb)

    # -- Bottom Panel: Residuals (Sigma) --
    ax2 = CM.Axis(fig[2, 1],
        xlabel = "k",
        ylabel = "Deviation (σ)",
        xscale = log10,
        height = 150
    )
    
    # Calculate Sigma: (Num - Ana) / Err
    sigmas = (vals_p .- ana_p) ./ errs_p
    
    CM.scatter!(ax2, ks_p, sigmas, color = :black)
    CM.hlines!(ax2, [0.0], color = :red, linestyle = :dash)
    CM.hlines!(ax2, [-3.0, 3.0], color = :gray, linestyle = :dot) # 3-sigma bounds
    
    # Link x-axes
    CM.linkxaxes!(ax1, ax2)
    
    # save("unequal_bubbles_comparison.png", fig)
    println("  Plot saved to 'unequal_bubbles_comparison.png'")
    fig
end