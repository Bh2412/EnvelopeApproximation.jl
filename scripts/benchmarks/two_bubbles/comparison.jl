# ==============================================================================
# 3. COMPARISON SCRIPT
# ==============================================================================

using Printf
using Random
using Measurements
using StaticArrays
using EnvelopeApproximation
using EnvelopeApproximation.GravitationalWaves
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions

include("Analytic.jl")
include("Numeric.jl")

# ------------------------------------------------------------------------------
# A. Configuration
# ------------------------------------------------------------------------------
# We define the parameters here to ensure both methods use the exact same setup.
begin
    r_bubble  = 5.0           # Bubble radius (time of interest)
    d_sep     = 6.0           # Separation distance
    L_box     = 40.0          # Box size (Large enough to approximate vacuum)
    N_samples = 10_000_000       # Number of Monte Carlo samples
    ks        = collect(LinRange(0.1, 10.0, 100)) # Wavenumbers to evaluate

    # Derived Parameters
    vol = L_box^3
    b1_site = Point3(0.0, 0.0, -d_sep/2)
    b2_site = Point3(0.0, 0.0,  d_sep/2)

    println("System Configuration:")
    println("  R = $r_bubble, d = $d_sep")
    println("  Box Length = $L_box (Volume = $vol)")
    println("  N_samples = $N_samples")
    println("-------------------------------------------------------------------------")
end

# ------------------------------------------------------------------------------
# B. Compute Numeric (Monte Carlo)
# ------------------------------------------------------------------------------
begin
    println("\n1. Computing Numeric Π (Monte Carlo)...")

    # Setup Snapshot (Bubbles nucleated at t=0)
    nucleations = [(time=0.0, site=b1_site), (time=0.0, site=b2_site)]
    snapshot    = BubblesSnapShot(nucleations, r_bubble * 2.0)

    # Setup Space & Strategy
    # Note: Using the BoxSpace constructor consistent with your Numeric.jl
    space       = BoxSpace(L_box, Point3(0.0, 0.0, 0.0)) 
    strategy    = MonteCarloSampling(N=N_samples, rng=StableRNG(1))

    # Run Simulation
    # We pass 'ks' as ωs (assuming c=1, k=ω)
    pi_numeric = Π(r_bubble, r_bubble, ks, snapshot, space, Periodic(), strategy; ΔV=1.0)


    # ------------------------------------------------------------------------------
    # C. Compute Reference Analytic
    # ------------------------------------------------------------------------------
    println("2. Computing Analytic Reference (Isotropic Integration)...")

    pi_analytic = Float64[]

    for k in ks
        # compute_analytic_isotropic_Pi is defined in Analytic.jl
        # It integrates the exact direction-dependent solution over all angles.
        val = compute_analytic_isotropic_Pi(k, r_bubble, d_sep, vol)
        push!(pi_analytic, val)
    end
end

# ------------------------------------------------------------------------------
# D. Compare & Output
# ------------------------------------------------------------------------------
println("\n3. Comparison Results (Isotropic Π):")
println("   k    |   Numeric (MC)       |   Analytic (Exact)   | Deviation (σ)")
println("-------------------------------------------------------------------------")

for i in eachindex(ks)
    k = ks[i]
    
    # Extract MC Measurement
    meas    = pi_numeric[i]
    num_val = Measurements.value(meas)
    num_err = Measurements.uncertainty(meas)
    
    # Analytic Value
    ref_val = pi_analytic[i]
    
    # Compute Sigma Difference
    # If error is zero (unlikely in MC), avoid division by zero
    sigma = num_err > 0 ? abs(num_val - ref_val) / num_err : Inf
    
    @printf("  %4.2f  |  %10.4e ± %.0e  |  %10.4e          |  %5.2f\n", 
            k, num_val, num_err, ref_val, sigma)
end

println("-------------------------------------------------------------------------")
if all(m -> (abs(Measurements.value(m[1]) - m[2]) / Measurements.uncertainty(m[1])) < 3.0, zip(pi_numeric, pi_analytic))
    println("SUCCESS: All MC points are within 3σ of the analytic reference.")
else
    println("WARNING: Some points deviate by more than 3σ.")
end

# ------------------------------------------------------------------------------
# D. Create Mask for Plotting & Print Warnings
# ------------------------------------------------------------------------------
println("\n3. Validating MC Results (Masking negatives / low signal)...")

# We want to filter out points where:
# 1. The mean value <= 0 (Non-physical)
# 2. The mean - std <= 0 (Indistinguishable from zero/noise)

valid_indices = Int[]

for i in eachindex(pi_numeric)
    val = Measurements.value(pi_numeric[i])
    err = Measurements.uncertainty(pi_numeric[i])
    k_val = ks[i]
    
    if val <= 0
        println("  [WARNING] k=$k_val excluded: Negative mean ($val)")
    elseif (val - err) <= 0
        println("  [WARNING] k=$k_val excluded: Mean - Std <= 0 ($val - $err = $(val-err))")
    else
        push!(valid_indices, i)
    end
end

println("  -> retained $(length(valid_indices)) / $(length(ks)) points.")

# Apply Mask
ks_valid       = ks[valid_indices]
mc_vals_valid  = Measurements.value.(pi_numeric[valid_indices])
mc_errs_valid  = Measurements.uncertainty.(pi_numeric[valid_indices])

# ------------------------------------------------------------------------------
# E. Plotting
# ------------------------------------------------------------------------------
begin
println("\n4. Generating Plot...")

fig = CM.Figure(size = (800, 600), fontsize = 18)
ax = CM.Axis(fig[1, 1],
    title = "Isotropic Π(k): Monte Carlo vs Analytic",
    xlabel = "Wavenumber k",
    ylabel = "Power Spectral Density Π(k)",
    xscale = log10,
    yscale = log10,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xgridstyle = :dash,
    ygridstyle = :dash
)

# Plot Analytic (All points, as these are exact)
CM.lines!(ax, ks, pi_analytic, 
    color = :red, 
    linewidth = 2, 
    label = "Analytic (Exact)"
)

# Plot Numeric (Only valid points)
if !isempty(ks_valid)
    CM.scatter!(ax, ks_valid, mc_vals_valid, 
        color = :blue, 
        markersize = 12, 
        label = "Monte Carlo (Filtered)"
    )
    CM.errorbars!(ax, ks_valid, mc_vals_valid, mc_errs_valid, 
        color = :blue, 
        linewidth = 1.5,
        whiskerwidth = 10
    )
else
    println("  [Alert] No valid MC points to plot.")
end

CM.axislegend(ax, position = :rt)
output_filename = "comparison_plot_masked.png"
CM.save(output_filename, fig)
println("Plot saved to '$output_filename'")
fig
end

begin
    println("\n=== Low-k Zoom Analysis (k < 1/R) ===")

    # 1. Define Low-k Range
    # ---------------------
    # Assuming β⁻¹ ~ R (characteristic scale). We want to look at k < 1/5 = 0.2
    # We use a log-spaced range to better visualize the power law behavior.
    ks_zoom = collect(10 .^ LinRange(log10(0.02), log10(1.), 15))
    
    println("  Zoom Range: k ∈ [$(minimum(ks_zoom)), $(maximum(ks_zoom))]")

    # 2. Compute Numeric (Monte Carlo)
    # --------------------------------
    println("  Computing Monte Carlo for low k...")
    # Re-use the setup variables (snapshot, space, strategy) from Numeric.jl
    # Note: We increase N_samples slightly if needed, but 200k is usually sufficient.
    pi_numeric_zoom = Π(
        r_bubble, r_bubble, ks_zoom, 
        snapshot, space, Periodic(), strategy; 
        ΔV=1.0
    )

    # 3. Compute Analytic Reference
    # -----------------------------
    println("  Computing Analytic Reference for low k...")
    pi_analytic_zoom = Float64[]
    for k in ks_zoom
        val = compute_analytic_isotropic_Pi(k, r_bubble, d_sep, vol)
        push!(pi_analytic_zoom, val)
    end

    # 4. Generate Zoomed Plot
    # -----------------------
    println("  Generating Low-k Plot...")
    
    # Extract values
    zoom_vals = Measurements.value.(pi_numeric_zoom)
    zoom_errs = Measurements.uncertainty.(pi_numeric_zoom)

    # Create Figure
    fig_zoom = CM.Figure(size = (800, 600), fontsize = 18)
    ax_zoom = CM.Axis(fig_zoom[1, 1],
        title = "Low-k Spectrum",
        xlabel = "Wavenumber k",
        ylabel = "Power Spectral Density Π(k)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xgridstyle = :dash,
        ygridstyle = :dash
    )

    # Plot Analytic Line
    CM.lines!(ax_zoom, ks_zoom, pi_analytic_zoom, 
        color = :red, linewidth = 2, label = "Analytic"
    )

    # Plot MC Scatter
    CM.scatter!(ax_zoom, ks_zoom, zoom_vals, 
        color = :blue, markersize = 12, label = "Monte Carlo"
    )
    CM.errorbars!(ax_zoom, ks_zoom, zoom_vals, zoom_errs, 
        color = :blue, linewidth = 1.5, whiskerwidth = 10
    )

    # Reference Slope k^3 (Arbitrary normalization for visual guide)
    # We anchor it to the middle of the analytic curve
    mid_idx = length(ks_zoom) ÷ 2
    ref_k = ks_zoom[mid_idx]
    ref_val = pi_analytic_zoom[mid_idx]

    CM.axislegend(ax_zoom, position = :lb) # Top-left is usually empty for growing functions

    # save("low_k_comparison.png", fig_zoom)
    println("  Zoomed plot saved to 'low_k_comparison.png'")
    fig_zoom
end