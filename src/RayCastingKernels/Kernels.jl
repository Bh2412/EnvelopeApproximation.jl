"""
    Kernel

Abstract interface for a kernel evaluated by the ray-casting envelope
integration.

Concrete kernels must implement:

- [`allocate_accumulant(kernel)`](@ref)
- [`accumulate_ray!(accumulant, kernel, source, direction, weight, stop_time)`](@ref)

They may additionally implement:

- [`prepare_kernel!(kernel, source)`](@ref), called once before integrating the
  rays belonging to a source
- [`finalize_accumulant!(accumulant, kernel)`](@ref), called once after all
  sources and rays have been accumulated
"""
abstract type Kernel end

"""
    allocate_accumulant(kernel::Kernel)

Allocate the mutable result accumulated by `kernel`.
"""
function allocate_accumulant end

"""
    prepare_kernel!(kernel::Kernel, source)

Prepare source-dependent kernel state before accumulating the source's rays.
The default implementation does nothing.
"""
prepare_kernel!(::Kernel, source) = nothing

"""
    accumulate_ray!(accumulant, kernel::Kernel, source, direction, weight, stop_time)

Accumulate one surviving ray into `accumulant`.
"""
function accumulate_ray! end

"""
    finalize_accumulant!(accumulant, kernel::Kernel)

Finalize and return an accumulant after all rays have been processed. The
default implementation returns the accumulant unchanged.
"""
finalize_accumulant!(accumulant, ::Kernel) = accumulant

export Kernel,
       allocate_accumulant,
       prepare_kernel!,
       accumulate_ray!,
       finalize_accumulant!
