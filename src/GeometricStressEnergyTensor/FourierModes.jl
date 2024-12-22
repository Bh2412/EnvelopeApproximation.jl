function fourier_mode(f:: SphericalIntegrand{Float64}, 
    κ:: Float64; kwargs...):: ComplexF64
    _f(μ:: Float64):: ComplexF64 = cis(-κ * μ) * ∫_ϕ(f, μ)
    return quadgk(_f, -1., 1.; kwargs...)[1]
end

function fourier_mode(bapi:: BubbleArcPotentialIntegrand, 
    κ:: Float64; kwargs...):: ComplexF64
    _f(μ:: Float64):: ComplexF64 = cis(-κ * μ) * ∫_ϕ(bapi, μ)
    μ_lims = polar_limits(bapi.R, bapi.domes)
    return quadgk(_f, μ_lims...; kwargs...)[1]
end

function fourier_mode(f:: SphericalIntegrand{MVector{K, Float64}}, 
    κ:: Float64; kwargs...):: MVector{K, ComplexF64} where K
    _f(μ:: Float64):: MVector{K, ComplexF64} = cis(-κ * μ) * ∫_ϕ(f, μ)
    return quadgk(_f, -1., 1.; kwargs...)[1]
end

function fourier_mode(basi:: BubbleArcSurfaceIntegrand,
    κ:: Float64; kwargs...):: MVector{6, ComplexF64}
    _f(μ:: Float64):: MVector{6, ComplexF64} = cis(-κ * μ) * ∫_ϕ(basi, μ)
    μ_lims = polar_limits(basi.R, basi.domes)
    return quadgk(_f, μ_lims...; kwargs...)[1]
end

function fourier_mode(ba:: BubbleArck̂ik̂j∂iφ∂jφ,
    κ:: Float64; kwargs...):: ComplexF64
    _f(μ:: Float64):: ComplexF64 = cis(-κ * μ) * ∫_ϕ(ba, μ)
    μ_lims = polar_limits(ba.R, ba.domes)
    return quadgk(_f, μ_lims...; kwargs...)[1]
end

function fourier_mode(ba:: BubbleArcŊ,
    κ:: Float64; kwargs...):: ComplexF64
    _f(μ:: Float64):: ComplexF64 = cis(-κ * μ) * ∫_ϕ(ba, μ)
    μ_lims = polar_limits(ba.R, ba.domes)
    return quadgk(_f, μ_lims...; kwargs...)[1]
end
