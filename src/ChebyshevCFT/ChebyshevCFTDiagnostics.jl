module ChebyshevCFTDiagnostics

using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
using EnvelopeApproximation.ChebyshevCFT
using CairoMakie
using Statistics
using Random
using LinearAlgebra

function analyze_chebyshev_plan_with_atol(
    f::Function, 
    ks::AbstractVector{<:Real}; 
    a::Float64=-1.0, 
    b::Float64=1.0,
    N_values::Vector{Int}=[7 * 3^n for n in 3:6],  # N must be divisible by P
    P::Int=7,
    α::Float64=1.0,
    plot_results::Bool=true
)
    """
    Analyze the accuracy of the ChebyshevPlanWithAtol for computing Fourier transforms.
    
    Parameters:
    - f: Function to analyze
    - ks: Vector of wavenumbers to test
    - a, b: Domain boundaries
    - N_values: Vector of N values to test (must all be divisible by P)
    - P: The reduction factor for error estimation (must be odd)
    - α: The convergence rate parameter
    - plot_results: Whether to generate plots
    
    Returns:
    - A named tuple with analysis results
    """
    # Verify all N values are divisible by P
    invalid_ns = filter(N -> N % P != 0, N_values)
    if !isempty(invalid_ns)
        error("All N values must be divisible by P=$P. Invalid values: $invalid_ns")
    end
    
    # Function to compute "exact" reference solutions via quadrature
    function exact_ft(f, k, a::Float64, b::Float64)
        result, err = quadgk(x -> f(x) * cis(-k * x), a, b, atol=1e-15)
        return result, err
    end

    # Compute high-precision reference solutions
    println("Computing high-precision reference solutions...")
    reference_solutions = Vector{ComplexF64}(undef, length(ks))
    reference_errors = Vector{Float64}(undef, length(ks))
    
    for (j, k) in enumerate(ks)
        sol, err = exact_ft(f, k, a, b)
        reference_solutions[j] = sol
        reference_errors[j] = err
        println("k = $k: solution = $sol, quadrature error = $err")
    end

    # Store results
    actual_errors = zeros(length(N_values), length(ks))
    estimated_errors = zeros(length(N_values))
    individual_errors = Dict{Int, Matrix{Float64}}()  # To store error at each point for selected N values

    println("\nTesting ChebyshevPlanWithAtol accuracy for various N values")
    println("="^60)

    for (i, N) in enumerate(N_values)
        println("Testing N = $N...")

        # Create ChebyshevPlanWithAtol
        plan = ChebyshevPlanWithAtol{N,P}(α)

        # Calculate Chebyshev coefficients
        chebyshev_coeffs!(f, a, b, plan)

        s = EnvelopeApproximation.ChebyshevCFT.scale(a, b)
        t = EnvelopeApproximation.ChebyshevCFT.translation(a, b)

        # Compute modes for each k and store estimated errors
        modes_and_errors = zeros(ComplexF64, length(ks))
        lower_modes = zeros(ComplexF64, length(ks))
        
        for (j, k) in enumerate(ks)
            # Get both full and lower order modes
            full_mode, lower_mode = fourier_mode(k, plan, s, t)
            modes_and_errors[j] = full_mode
            lower_modes[j] = lower_mode
            
            # Calculate actual error for this k
            actual_errors[i, j] = abs(full_mode - reference_solutions[j])
        end

        # Calculate estimated error using the plan's method
        inf_norm = maximum(abs.(modes_and_errors - lower_modes))
        estimated_errors[i] = inf_norm / (P^α - 1)
        
        println("  Estimated error: $(estimated_errors[i])")
        println("  Max actual error: $(maximum(actual_errors[i, :]))")

        # For all N values, compute and store point-wise errors
        # Create dense grid for pointwise evaluation
        x_grid = range(a, b, length=1000)
        f_exact = @. f(x_grid) * sqrt(1 - ((x_grid - t) / s) ^ 2)
        
        # Store exact function values, reconstructed values and errors
        individual_errors[N] = zeros(1000, 3)
        individual_errors[N][:, 1] = f_exact
        
        # Reconstruct function from Chebyshev coefficients
        for (idx, x) in enumerate(x_grid)
            u = (x - t) / s  # Map to [-1, 1]
            
            # Reconstruct function value from Chebyshev series
            val = 0.0
            for j in 1:N
                # Use the fact that T_j(cos(θ)) = cos(j*θ)
                θ = acos(clamp(u, -1.0, 1.0))
                val += plan.coeffs_buffer[j] * cos((j-1) * θ)
            end
            individual_errors[N][idx, 2] = val
            individual_errors[N][idx, 3] = abs(val - f_exact[idx])
        end
    end

    # Calculate convergence rates
    println("\nCalculating empirical convergence rates:")
    println("="^60)

    # Overall convergence rate
    overall_errors = [maximum(actual_errors[i, :]) for i in 1:length(N_values)]
    rates = Float64[]
    for i in 1:(length(N_values)-1)
        rate = log(overall_errors[i]/overall_errors[i+1]) / log(N_values[i+1]/N_values[i])
        push!(rates, rate)
        println("Rate from N=$(N_values[i]) to N=$(N_values[i+1]): $rate")
    end

    avg_rate = mean(rates)
    println("Average empirical convergence rate: $avg_rate")
    
    # Compare with specified alpha
    if abs(avg_rate - α) > 0.3
        println("\nWARNING: Empirical convergence rate ($avg_rate) differs significantly from specified α ($α).")
        println("Consider adjusting α to match the empirical rate for better error estimation.")
    end

    # Plot results if requested
    if plot_results
        fig = plot_chebyshev_plan_with_atol_results(
            f, N_values, ks, reference_solutions, 
            actual_errors, estimated_errors, individual_errors,
            P, α, a, b
        )
    else
        fig = nothing
    end

    # Return a named tuple with the results
    return (
        N_values = N_values,
        ks = ks,
        P = P,
        α = α,
        reference_solutions = reference_solutions,
        reference_errors = reference_errors,
        actual_errors = actual_errors,
        estimated_errors = estimated_errors,
        individual_errors = individual_errors,
        convergence_rates = rates,
        avg_convergence_rate = avg_rate,
        figure = fig
    )
end

function generate_distinct_colors(n:: Int)
    if n <= 0
        return []
    elseif n == 1
        return [:blue]
    else
        # Use ColorSchemes to generate distinct colors
        # Or manually define a color generation technique
        hues = range(0, 330, length=n) # Distribute hues around the color wheel
        saturations = fill(0.8, n)
        values = fill(0.9, n)
        
        # Convert HSV to RGB
        colors = map(i -> HSV(hues[i], saturations[i], values[i]), 1:n)
        return colors
    end
end

function plot_chebyshev_plan_with_atol_results(
    f::Function,
    N_values::Vector{Int}, 
    ks::AbstractVector{<:Real}, 
    reference_solutions::Vector{ComplexF64}, 
    actual_errors::Matrix{Float64},
    estimated_errors::Vector{Float64},
    individual_errors::Dict{Int, Matrix{Float64}},
    P::Int,
    α::Float64,
    a::Float64,
    b::Float64
)    
    fig = Figure(size=(1600, 1000))

    # Absolute error convergence plot
    ax1 = Axis(fig[1, 1],
        title = "Convergence of Fourier Transform Approximation",
        xlabel = "N (Number of Chebyshev points)",
        ylabel = "Absolute Error",
        xscale = log10,
        yscale = log10)

    # Plot actual error for each k
    for (j, k) in enumerate(ks)
        lines!(ax1, N_values, actual_errors[:, j], label="k = $k")
        scatter!(ax1, N_values, actual_errors[:, j])
    end

    # Plot theoretical O(N^(-α)) convergence
    ref_point = (N_values[end], mean(actual_errors[end, :]))
    theoretical_line = ref_point[2] * (ref_point[1] ./ N_values).^α
    lines!(ax1, N_values, theoretical_line, 
        linestyle=:dash, 
        linewidth=2,
        color=:black,
        label="O(N^(-$α))")

    # Actual Fourier transform values plot
    ax_ft = Axis(fig[1, 2],
        title = "Actual Fourier Transform Values",
        xlabel = "k",
        ylabel = "|F(k)|", 
        xscale = log10)

    # Plot magnitude of reference solutions
    magnitudes = abs.(reference_solutions)
    lines!(ax_ft, ks, magnitudes)
    scatter!(ax_ft, ks, magnitudes)

    # Function approximation plot
    ax_func = Axis(fig[1, 3],
        title = "Function Approximation",
        xlabel = "u",
        ylabel = "f(x(u)) ⋅ √(1 - u²)",)
    
    # Plot the exact function
    x_grid = range(a, b, length=1000)
    t = EnvelopeApproximation.ChebyshevCFT.translation(a, b)
    s = EnvelopeApproximation.ChebyshevCFT.scale(a, b)
    f_exact = @. f(x_grid) * sqrt(1 - ((x_grid - t) / s) ^ 2)
    lines!(ax_func, x_grid, f_exact, 
        linewidth=2, color=:black, label="Exact")
    
    # Plot selected approximations
    # Number of unique N values in individual_errors
    num_n_values = length(keys(individual_errors))
    colors = generate_distinct_colors(num_n_values)
    i = 1
    for N in sort(collect(keys(individual_errors)))
        lines!(ax_func, x_grid, individual_errors[N][:, 2], 
            linewidth=1, color=colors[i], linestyle=:dash, 
            label="N = $N")
        i += 1
    end
    
    axislegend(ax_func, position=:rt)
    
    # Pointwise error plot
    ax_pointwise = Axis(fig[2, 3],
        title = "Pointwise Approximation Error",
        xlabel = "x",
        ylabel = "Absolute Error",
        yscale = log10)

    # Sort N values to get ordered sequence (ascending)
    sorted_N_values = sort(collect(keys(individual_errors)))

    # Select only first, second-to-last, and last N values
    selected_indices = []
    if length(sorted_N_values) >= 1
        push!(selected_indices, 1)  # First
    end
    if length(sorted_N_values) >= 3
        push!(selected_indices, length(sorted_N_values) - 1)  # Second-to-last
    end
    if length(sorted_N_values) >= 2
        push!(selected_indices, length(sorted_N_values))  # Last
    end

    # Generate colors for selected N values
    selected_colors = generate_distinct_colors(length(selected_indices))

    # Plot only the selected N values
    for (color_idx, idx) in enumerate(selected_indices)
        N = sorted_N_values[idx]
        lines!(ax_pointwise, x_grid, individual_errors[N][:, 3], 
            linewidth=2.0, color=selected_colors[color_idx], 
            label="N = $N")
    end

    axislegend(ax_pointwise, position=:rt)

    # Find the maximum absolute error for each N value
    max_absolute_errors = [maximum(actual_errors[i, :]) for i in 1:length(N_values)]
    
    # Extract maximum interpolation errors for available N values
    interp_n_values = Int[]
    max_interp_errors = Float64[]
    for N in N_values
        if haskey(individual_errors, N)
            push!(interp_n_values, N)
            push!(max_interp_errors, maximum(individual_errors[N][:, 3]))
        end
    end

    # Plot showing maximum absolute error as a function of N
    ax_max_abs = Axis(fig[2, 1],
        title = "Maximum Absolute Error vs N (P=$P, α=$α)",
        xlabel = "N (Number of Chebyshev points)",
        ylabel = "Maximum Absolute Error",
        xscale = log10,
        yscale = log10)

    # Plot Fourier errors
    lines!(ax_max_abs, N_values, max_absolute_errors, 
        color=:blue, label="Fourier Error")
    scatter!(ax_max_abs, N_values, max_absolute_errors, color=:blue)

    # Plot estimated errors from ChebyshevPlanWithAtol
    lines!(ax_max_abs, N_values, estimated_errors,
        color=:purple, label="Estimated Error")
    scatter!(ax_max_abs, N_values, estimated_errors, color=:purple)

    # Plot Interpolation errors
    lines!(ax_max_abs, interp_n_values, max_interp_errors, 
        color=:red, label="Interpolation Error")
    scatter!(ax_max_abs, interp_n_values, max_interp_errors, color=:red)

    # Fit power law model for Fourier errors: A⋅N^(-α)
    if length(N_values) >= 3
        log_errors_fourier = log.(max_absolute_errors)
        log_n_fourier = log.(N_values)
        
        # Linear regression on log-log scale
        A_fourier = hcat(ones(length(log_n_fourier)), log_n_fourier)
        params_fourier = A_fourier \ log_errors_fourier
        
        A_coef_fourier = exp(params_fourier[1])
        α_fourier = -params_fourier[2]
        
        # Generate points for the fitted curve
        fit_n_fourier = exp.(range(log(minimum(N_values)), log(maximum(N_values)), length=100))
        fit_errors_fourier = A_coef_fourier ./ fit_n_fourier.^α_fourier
        
        # Add fitted curve
        lines!(ax_max_abs, fit_n_fourier, fit_errors_fourier, 
            color=:blue, linestyle=:dash, 
            label="Fourier Fit: $(round(A_coef_fourier, digits=3))⋅N^(-$(round(α_fourier, digits=3)))")
    end
    
    # Fit power law model for estimated errors
    if length(N_values) >= 3
        log_errors_est = log.(estimated_errors)
        log_n_est = log.(N_values)
        
        # Linear regression on log-log scale
        A_est = hcat(ones(length(log_n_est)), log_n_est)
        params_est = A_est \ log_errors_est
        
        A_coef_est = exp(params_est[1])
        α_est = -params_est[2]
        
        # Generate points for the fitted curve
        fit_n_est = exp.(range(log(minimum(N_values)), log(maximum(N_values)), length=100))
        fit_errors_est = A_coef_est ./ fit_n_est.^α_est
        
        # Add fitted curve
        lines!(ax_max_abs, fit_n_est, fit_errors_est, 
            color=:purple, linestyle=:dash, 
            label="Est Fit: $(round(A_coef_est, digits=3))⋅N^(-$(round(α_est, digits=3)))")
    end

    # Fit power law model for Interpolation errors: A⋅N^(-α)
    if length(interp_n_values) >= 3
        log_errors_interp = log.(max_interp_errors)
        log_n_interp = log.(interp_n_values)
        
        # Linear regression on log-log scale
        A_interp = hcat(ones(length(log_n_interp)), log_n_interp)
        params_interp = A_interp \ log_errors_interp
        
        A_coef_interp = exp(params_interp[1])
        α_interp = -params_interp[2]
        
        # Generate points for the fitted curve
        fit_n_interp = exp.(range(log(minimum(interp_n_values)), log(maximum(interp_n_values)), length=100))
        fit_errors_interp = A_coef_interp ./ fit_n_interp.^α_interp
        
        # Add fitted curve
        lines!(ax_max_abs, fit_n_interp, fit_errors_interp, 
            color=:red, linestyle=:dash, 
            label="Interp Fit: $(round(A_coef_interp, digits=3))⋅N^(-$(round(α_interp, digits=3)))")
    end

    axislegend(ax_max_abs, position=:lb)

    # Create a plot to compare actual vs estimated errors
    ax_error_ratio = Axis(fig[2, 2],
        title = "Error Estimation Accuracy (P=$P, α=$α)",
        xlabel = "N (Number of Chebyshev points)",
        ylabel = "Ratio (Estimated/Actual)",
        xscale = log10)

    # Calculate ratios of estimated to actual errors
    error_ratios = estimated_errors ./ max_absolute_errors
    
    # Create a bar plot of ratios
    barplot!(ax_error_ratio, 1:length(N_values), error_ratios,
            color=:lightblue,
            width=0.6)
    
    # Add a horizontal line at 1.0 for perfect estimation
    hlines!(ax_error_ratio, [1.0], color=:red, linestyle=:dash)
    
    # Add text labels for the ratios
    for i in 1:length(N_values)
        text!(ax_error_ratio, i, error_ratios[i] + 0.1, 
            text = "$(round(error_ratios[i], digits=2))", 
            align = (:center, :bottom))
    end
    
    # Set x-axis ticks to show the N values
    ax_error_ratio.xticks = (1:length(N_values), string.(N_values))
          
    display(fig)
    return fig
end

end