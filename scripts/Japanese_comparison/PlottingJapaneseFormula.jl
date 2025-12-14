include("./JapaneseFormula.jl")

using CairoMakie
using LaTeXStrings

begin
    # --- Setup Parameters ---
    t1_val = 5.0
    t2_val = 5.0
    G_star = 1.38e-3
    
    # Generate log-spaced k values from 0.01 to 100
    k_min, k_max = 0.1, 10.0
    n_points = 50
    k_values = 10 .^ range(log10(k_min), log10(k_max), length=n_points)

    # --- Pre-allocate Storage ---
    pi_single = zeros(length(k_values))
    pi_double = zeros(length(k_values))
    pi_total  = zeros(length(k_values))

    println("Computing contributions to Pi for t=$t1_val, Gamma_star=$G_star...")

    # --- Main Computation Loop ---
    for (i, k) in enumerate(k_values)
        # Calculate raw contributions
        s = compute_Pi_single(t1_val, t2_val, k; Gamma_star=G_star, atol=1e-13, rtol=1e-3)
        d = compute_Pi_double(t1_val, t2_val, k; Gamma_star=G_star, atol=1e-13, rtol=1e-3)
        
        # Store absolute values for log-log plotting
        # (The correlator Pi can oscillate/be negative, but magnitude is needed for log plots)
        pi_single[i] = abs(s)
        pi_double[i] = abs(d)
        pi_total[i]  = abs(s + d)
    end

    println("Computation complete.")
end

begin
    # Create the figure and axis with log-log scales
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "k",
        ylabel = L"|\Pi(t, t, k)|",
        title = "Unequal-time Correlator (t1=t2=$t1_val, G*=$G_star)",
        xminorticksvisible = true, 
        yminorticksvisible = true,
        xminorgridvisible = true, 
        yminorgridvisible = true
    )

    # Plot lines
    # Total Pi (Black, solid)
    lines!(ax, k_values, pi_total, 
        label = L"Total |\Pi|", 
        color = :black, 
        linewidth = 2
    )

    # Single Bubble (Red, dashed)
    lines!(ax, k_values, pi_single, 
        label = "Single Bubble", 
        color = :red, 
        linewidth = 2, 
        linestyle = :dash
    )

    # Double Bubble (Blue, dashed)
    lines!(ax, k_values, pi_double, 
        label = "Double Bubble", 
        color = :blue, 
        linewidth = 2, 
        linestyle = :dash
    )

    # Add legend
    axislegend(ax, position = :lb)

    # Display figure
    fig
end
