#This is the main file. In this file, the DSEs are solved. 

#load packages
using LinearAlgebra
using Printf
using Plots
using Interpolations
using FastGaussQuadrature
using Base.Threads
using DelimitedFiles
using Mmap
#local modules
include("Term3.jl")
using .Term3

include("int.jl")
using .integrate2D

#test how many threads are available
Threads.nthreads()



#Integration cutoffs
IR_CUTOFF = 0.0001;       #IR cutoff for momentum and frequency
UV_CUTOFF = 10;        #UV cutoff for momentum
ω_max = 10^6;         #UV cutoff for frequency

#define constants
m2= 1;                      #bare mass 
T= 12;                      #Temperature
const λ= 1;                 #couling tadpole
g = 1;                      #coupling sunrises
const Γ0=1;
const γ=1;
const χ=1;              
const N=1000000;            #N = infinity
zeroC = complex(0.0, 0.0)   #complex 0

#higher level constants
const γ_χ= γ/χ;
const Γ02= Γ0^2;

## Prefactors
term1_constants =   -(im/2) * 8 * γ * g^2/χ^2 * 1/(2*pi)^3 # *(N-1)/N
term2_constants =   +(im/2) * 4 * g^2/χ * 1/(2*pi)^3 #*(N-1)/N
term3_constant =   +(im/2)* λ/3  * 1/(2*pi)^3 #* (N+2)/N
Γ_constant=   4 * g^2 * γ * 1/χ^2 * 1/(2*pi)^3    #*(N-1)/N

#Grid

##dimensions(this needs to match the .bin files)
NUM_NODES_FREQ = 2^7;               #Number of nodes for the frequency integration
NUM_NODES_MOM= 2^7;                 #Number of nodes for the momentum integration   

## Momentum grid
logmomNodes, logmomWeights= gausslegendre(NUM_NODES_MOM)
logmomNodes = 0.5 * (logmomNodes .+ 1) * (log(UV_CUTOFF) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
logmomWeights_zw = 0.5 * (log(UV_CUTOFF) - log(IR_CUTOFF)) * logmomWeights
p_grid = exp.(logmomNodes)                          #none-logarithmic grid
logmomWeights = logmomWeights_zw .* p_grid          #weight (including logarithmic measure)

## Frequency grid
logfreqNodes, logfreqWeights1 = gausslegendre(NUM_NODES_FREQ)
logfreqNodes = 0.5 * (logfreqNodes .+ 1) * (log(ω_max) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
logfreqWeights1 = 0.5 * (log(ω_max) - log(IR_CUTOFF)) * logfreqWeights1
w_grid1 = exp.(logfreqNodes)
w_grid2= -reverse(w_grid1);
w_grid= vcat(w_grid2, w_grid1);                                             #momentum grid is doubled (-ω_max to ω_max)
logfreqWeights2 = reverse(logfreqWeights1);
logfreqWeights= vcat(logfreqWeights2, logfreqWeights1) .* abs.(w_grid)      #weight (including logarithmic measure)

#Define angular solution

##There are two ways of doing this: 
### 1. If N is small enough: save complete matrices on RAM
### 2. If N is too large, then save only chunks of them

#For both cases: load them in the SSD first
c1_file = open("tmp/C1_128.bin")
m = read(c1_file, Int)
n = read(c1_file, Int)
o = read(c1_file, Int)
p = read(c1_file, Int)
length=  m*n*o*p

c1_sol_flat = mmap(c1_file, Vector{Float64}, length)

#Define angular solution
c2_file = open("tmp/C2_128.bin")
m = read(c2_file, Int)
n = read(c2_file, Int)
o = read(c2_file, Int)
p = read(c2_file, Int)
length=  m*n*o*p

c2_sol_flat = mmap(c2_file, Vector{ComplexF64}, length)

### Version 1

#c1_sol = reshape(c1_sol_flat, (p, o, n, m))

#create a 4d chunk
function load_chunk(data)
    chunk = zeros(eltype(data),m, n, o, p)

    for w_out in 1:m
        for p_out in 1:n
            for w_in in 1:o
                for p_in in 1:p
                    # Calculate the index in the flat array
                    idx = p_in + (w_in-1)*p + (p_out-1)*p*o + (w_out-1)*p*o*n 
                    chunk[w_out, p_out, w_in, p_in] = data[idx]
                end
            end
        end
    end


    return chunk
end

c1_sol= load_chunk(c1_sol_flat)

#version 2
#=

#create a 2d chunk
function load_xy_chunk(data, w_out, p_out)
    chunk = zeros(eltype(data), o, p)

    @inbounds for w_in in 1:o
        for p_in in 1:p
            idx = p_in + (w_in-1)*p + (p_out-1)*p*o + (w_out-1)*p*o*n 
            chunk[w_in, p_in] = data[idx]
        end
    end

    return chunk
end


=#



Σ = fill(0.0+0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)
Γ = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)

@time integrate2D.int_freq_mom(Σ, Γ, chunkc1, chunkc2, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)


#close files
close(c1_file)
close(c2_file)