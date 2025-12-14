# plot_ratio.jl

using JLD2
using CairoMakie
using Statistics
using LaTeXStrings

# Make sure this file is available in your path
include("JapaneseFormula.jl") 

"""
    plot_simulation_analytic_ratio(simulation_file::String)

Loads simulation data, computes the theoretical prediction, and plots
the ratio (Simulation / Theory) to verify accuracy.
"""
function plot_simulation_analytic_ratio(simulation_file::String)
    println("--- Ratio Plotting Script ---")
    
    # 1. Load Simulation Data
    # -----------------------
    if !isfile(simulation_file)
        error("File not found: $simulation_file")
    end
    
    println("Loading $simulation_file...")
    data = load(simulation_file)
    
    # Extract data saved by KT_bigpi.jl
    ks_sim = data["ks"]                       # k values
    results_Π = data["results_Π"]             # Complex matrix (N_realizations x N_k)
    t_target = get(data, "t_target", 5.0)     # Time t1=t2
    
    # Compute Simulation Mean (Real part)
    # The correlator is physically real; imaginary parts are numerical noise
    sim_mean = vec(mean(real.(results_Π), dims=1))
    
    # Calculate Standard Error of the Mean (SEM) for error bars on the ratio
    # SEM = std / sqrt(N)
    n_realizations = size(results_Π, 1)
    sim_sem = vec(std(real.(results_Π), dims=1)) ./ sqrt(n_realizations)

    # 2. Compute Analytic Data
    # ------------------------
    println("Computing analytic benchmark for t=$t_target...")
    
    # Parameters must match the simulation (KT_bigpi.jl source 269)
    Γ_star = 1.38e-3 
    β_val  = 1.0
    
    analytic_vals = zeros(Float64, length(ks_sim))

    for (i, k) in enumerate(ks_sim)
        # Compute Total Pi (Single + Double)
        analytic_vals[i] = compute_Pi(t_target, t_target, k; 
                                      v=1.0, 
                                      beta=β_val, 
                                      kappa=1.0, 
                                      rho0=1.0, 
                                      Gamma_star=Γ_star, 
                                      atol=1e-12, rtol=1e-4)
    end

    # 3. Compute Ratio
    # ----------------
    # Ratio = Simulation / Analytic
    ratio = sim_mean ./ analytic_vals
    
    # Propagate error to ratio: σ_R = R * (σ_sim / μ_sim) 
    # (Assuming analytic has negligible error)
    ratio_err = ratio .* (sim_sem ./ sim_mean)

    # 4. Visualization
    # ----------------
    println("Generating ratio plot...")
    
    fig = Figure(size = (1000, 600))
    
    ax = Axis(fig[1, 1],
        xscale = log10,
        xlabel = L"Wavenumber $k/\beta$",
        ylabel = L"Ratio $\Pi_{\text{sim}} / \Pi_{\text{anal}}$",
        title = "Validation Ratio: Simulation vs. Japanese Formula (t=$t_target)",
        xminorticksvisible = true, 
        yminorticksvisible = true,
        xminorgridvisible = true,
        # Set y-limits to zoom in on the relevant region (e.g., 0.5 to 1.5)
        # Adjust these based on your specific convergence
    )

    # Reference line at Ratio = 1.0 (Perfect Agreement)
    hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 2, label = "Exact Agreement")

    # Ratio with Error Bars
    errorbars!(ax, ks_sim, ratio, ratio_err, color = :cornflowerblue)
    scatter!(ax, ks_sim, ratio, color = :blue, markersize = 8, label = "Simulation Mean")

    axislegend(ax, position = :rb)

    display(fig)
    return fig
end

# --- Run ---
# Replace with your actual filename
file_path = "/home/ben/dev/EnvelopeApproximation/KT_bigpi_2025_12_14.jld2"
if isfile(file_path)
    plot_simulation_analytic_ratio(file_path)
else
    println("Please update 'file_path' to point to your .jld2 simulation output.")
end