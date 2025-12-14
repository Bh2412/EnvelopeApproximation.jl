# compare_simulation_theory.jl

using JLD2
using CairoMakie
using Statistics
using LaTeXStrings

# Include the analytic formula code
# Make sure this file is in the same directory or provide the correct path
include("JapaneseFormula.jl") 

"""
    compare_simulation_theory(simulation_file::String)

Loads simulation data from `simulation_file`, calculates the analytic 
prediction using the same parameters, and plots them side-by-side.
"""
function compare_simulation_theory(simulation_file::String)
    println("--- Comparison Script ---")
    println("Loading simulation data from: $simulation_file")
    
    if !isfile(simulation_file)
        error("File not found: $simulation_file")
    end

    # 1. Load Simulation Data
    # -----------------------
    data = load(simulation_file)
    
    # Extract data (saved in KT_bigpi.jl)
    ks = data["ks"]                       # Vector of k values
    results_Π = data["results_Π"]         # Matrix (Realizations x k)
    t_target = get(data, "t_target", 5.0) # Simulation time (default 5.0 if missing)
    
    # Calculate Simulation Statistics
    real_Π = real.(results_Π)
    sim_mean = vec(mean(real_Π, dims=1))
    sim_std  = vec(std(real_Π, dims=1))
    
    println("Simulation Data Loaded:")
    println("  Time (t): $t_target")
    println("  k range:  [$(minimum(ks)), $(maximum(ks))]")
    println("  Samples:  $(size(results_Π, 1)) realizations")

    # 2. Compute Analytic Data
    # ------------------------
    println("\nComputing analytic prediction (Japanese Formula)...")
    
    # Parameters used in KT_bigpi.jl
    # Note: These must match the simulation settings. 
    # KT_bigpi.jl uses G* = 1.38e-3, beta=1.0.
    Γ_star = 1.38e-3 
    β_val  = 1.0
    
    analytic_total  = zeros(length(ks))
    analytic_single = zeros(length(ks))
    analytic_double = zeros(length(ks))

    # Loop over k values used in simulation
    for (i, k) in enumerate(ks)
        # Compute Single Bubble contribution
        s = compute_Pi_single(t_target, t_target, k; 
                              Gamma_star=Γ_star, beta=β_val, 
                              atol=1e-12, rtol=1e-4)
        
        # Compute Double Bubble contribution
        d = compute_Pi_double(t_target, t_target, k; 
                              Gamma_star=Γ_star, beta=β_val, 
                              atol=1e-12, rtol=1e-4)
        
        analytic_single[i] = abs(s)
        analytic_double[i] = abs(d)
        analytic_total[i]  = abs(s + d)
    end
    println("Analytic computation complete.")

    # 3. Visualization
    # ----------------
    println("\nGenerating comparison plot...")
    
    fig = Figure(size = (1000, 700))
    
    # Main Axis
    ax = Axis(fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = L"Wavenumber $k/\beta$",
        ylabel = L"|\Pi(k, t, t)|",
        title = "Simulation vs. Analytic Theory (t=$t_target)",
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true
    )

    # --- Plot Analytic Theory (Lines) ---
    # Total
    lines!(ax, ks, analytic_total, 
        color = :black, linewidth = 2, label = "Analytic Total")

    # --- Plot Simulation Data (Scatter + Error Bands) ---
    # Standard Deviation Band
    band!(ax, ks, sim_mean .- sim_std, sim_mean .+ sim_std, 
        color = (:cornflowerblue, 0.3), label = "Sim ±1σ")
    
    # Mean Points
    scatter!(ax, ks, sim_mean, 
        color = :cornflowerblue, markersize = 6, strokewidth = 1, strokecolor=:black,
        label = "Simulation Mean")

    # Log-Log Slope Guides (Optional)
    # k^3 slope for low k
    # k_low = ks[ks .< 1.0]
    # lines!(ax, k_low, (analytic_total[1] / k_low[1]^3) .* k_low.^3, 
    #     color = :grey, linestyle = :dot, label = L"k^3")

    # Legend
    axislegend(ax, position = :lb)

    # Display
    display(fig)
    return fig
end

# --- Usage Example ---
# Update this path to your specific output file from KT_bigpi.jl
file_path = "/home/ben/dev/EnvelopeApproximation/KT_bigpi_2025_12_14.jld2" 
compare_simulation_theory(file_path)