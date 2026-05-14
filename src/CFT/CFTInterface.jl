module CFTInterface

abstract type CFTPlan end

function fourier_modes(f, ks:: AbstractVector{<: Real}, a:: Real, b:: Real, plan:: CFTPlan)
    error("fourier_modes not implemented for plan of type $(typeof(plan))")
end

end