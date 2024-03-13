# ------ SimpleMotionType
abstract type SimpleMotionType{T<:Real} end

"""
Simple Motion

x = x + ux
y = y + uy
z = z + uz

"""
# -------- SimpleMotion
mutable struct SimpleMotion{S <: SimpleMotionType} <: MotionModel
    types::AbstractVector{S}
end

Base.getindex(motion::SimpleMotion, p::Union{AbstractRange,AbstractVector,Colon}) = motion
Base.getindex(motion::SimpleMotion, p::Union{AbstractRange,AbstractVector,Colon}, 
                                    q::Union{AbstractRange,AbstractVector,Colon}) = motion

function get_spin_coords(motion::SimpleMotion{S}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, t::AbstractArray{T}) where {T<:Real, S<:SimpleMotionType{T}}
    xt = x .+ reduce(.+, map((type) -> displacement_x(type, x, y, z, t), motion.types)) # Periodicity could be included here
    yt = y .+ reduce(.+, map((type) -> displacement_y(type, x, y, z, t), motion.types))
    zt = z .+ reduce(.+, map((type) -> displacement_z(type, x, y, z, t), motion.types))
    return xt, yt, zt
end

# --------- Simple Motion Types: -------------
# Non-periodic types: defined by an initial time (ti), a final time (tf) and a displacement      
include("simplemotion/Translation.jl")
include("simplemotion/Rotation.jl")
# include("simplemotion/Cardiac.jl")
                                    
# Periodic types: defined by the period, the temporal symmetry and a displacement (amplitude)








"""
Idea for motion pipeline (simple motion composition)

positions = x, y, z
for type in motion.types
positions = get_coords(type, positions...)
end
return positions
"""
# +(x::SimpleMotionType, y::SimpleMotionType)

