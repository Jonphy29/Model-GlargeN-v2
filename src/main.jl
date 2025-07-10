#Load all packages

using LinearAlgebra
using Printf
using Plots
using Interpolations
using FastGaussQuadrature
using Base.Threads
using DelimitedFiles
using StaticArrays

#using Profile
#using Dierckx
#using QuadGK
#using GLMakie  


#local modules
include("Term3.jl")
using .Term3

include("int.jl")
using .integrate2D

#test how many threads are available
Threads.nthreads()

## IR and UV cutoff 
IR_CUTOFF = 0.0001;       #Same IR cutoff for momentum and frequency
UV_CUTOFF = 10;        #UV cutoff (physical cutoff for momenta) 
ω_max = 10^6;         #Omega is integrated from 0 to ω_max

## Constants
m2= 1;                    #bare mass 
T= 12;                  #Temperature(This needs to be varied for negative m^2, to find the critical point)
const λ= 1;                #couling tadpole
g = 1;                 #coupling sunrises
const Γ0=1;
const γ=1;
const χ=1;              
const N=1000000;               #infinity(The factor is not used in the following, but maybe practical for the next order in the 1/N expansion)

## higher level constants

const γ_χ= γ/χ;
const Γ02= Γ0^2;

## Prefactors
term1_constants =   -(im/2) * 8 * γ * g^2/χ^2 * 1/(2*pi)^3 # *(N-1)/N
term2_constants =   +(im/2) * 4 * g^2/χ * 1/(2*pi)^3 #*(N-1)/N
term3_constant =   +(im/2)* λ/3  * 1/(2*pi)^3 #* (N+2)/N
Γ_constant=   4 * g^2 * γ * 1/χ^2 * 1/(2*pi)^3    #*(N-1)/N

#Grid

NUM_NODES_FREQ = 2^6;           #Number of nodes for the frequency integration
NUM_NODES_MOM= 2^6;            #Number of nodes for the momentum integration   

#### Momentum grid
 
logmomNodes, logmomWeights= gausslegendre(NUM_NODES_MOM)
logmomNodes = 0.5 * (logmomNodes .+ 1) * (log(UV_CUTOFF) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
logmomWeights_zw = 0.5 * (log(UV_CUTOFF) - log(IR_CUTOFF)) * logmomWeights

#logmomNodes, logmomWeights_zw= gauss(NUM_NODES_MOM, log(IR_CUTOFF), log(UV_CUTOFF));  


p_grid = exp.(logmomNodes) #none-logarithmic grid
logmomWeights = logmomWeights_zw .* p_grid      #weight (including logarithmic measure)
                                            
#### Frequency grid

logfreqNodes, logfreqWeights1 = gausslegendre(NUM_NODES_FREQ)
logfreqNodes = 0.5 * (logfreqNodes .+ 1) * (log(ω_max) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
logfreqWeights1 = 0.5 * (log(ω_max) - log(IR_CUTOFF)) * logfreqWeights1

#logfreqNodes, logfreqWeights1= gauss(NUM_NODES_FREQ, log(IR_CUTOFF), log(ω_max));  

w_grid1 = exp.(logfreqNodes)

w_grid2= -reverse(w_grid1);

w_grid= vcat(w_grid2, w_grid1); 

logfreqWeights2 = reverse(logfreqWeights1);
logfreqWeights= vcat(logfreqWeights2, logfreqWeights1) .* abs.(w_grid)

           
#define complex 0
zeroC = complex(0.0, 0.0)

Σ = fill(0.0+0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)
Γ = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)



#Test functions

@time Term3.Int3(Σ, Γ, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)
#Tadpole_it1_exact(Λ)=  λ * Γ0/3 * 1/((2*pi))^3 * T * 2 * pi *  (arctan(Λ/sqrt(m2-x))); #* (2*pi)^4  
#Tadpole_it1_exact(10)

C1[65, 1, 65:end, 1]

w_grid1
scatter(w_grid1, C1[65, 1, 65:end, 1])


@time integrate2D.int_freq_mom(Σ, Γ, C1[65, 1, :, :], C2[65, 1, :, :], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)


Term2.Int2(Σ, Γ, 0.1, 0.1, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term2_constants, Γ_constant, γ_χ, Γ0)

S(w,p)= 0.0+0.0*im
G(w,p)= 2*Γ0*T*im

@time Term1.Int1(S, G, 0.1, 0.1, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term1_constants, γ_χ, Γ0)

function Integrate(Σ, Γ, T, m2)

    #Initialize output Arrays
    Σ_new1 = Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)
    Γ_new1 = Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)
    

    Threads.@threads for i = 1:NUM_NODES_MOM
        
        #outer momentum p
        
        for j = 1:NUM_NODES_FREQ
            
            I2, Γ_new1[j,i] = Term2.Int2(Σ, Γ, p_grid[i], w_grid1[j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term2_constants, Γ_constant, γ_χ, Γ0)

        end

    end

    Γ_new2 = reverse(Γ_new1, dims=1) 
    Γ = vcat(Γ_new2, Γ_new1)

    #Extra/Interpolation
    S_imag = extrapolate(interpolate((w_grid, p_grid), imag(Σ), Gridded(Linear())), Flat())
    S_real = extrapolate(interpolate((w_grid, p_grid), real(Σ), Gridded(Linear())), Flat())
    S(w,p)= complex(S_real(w,p), S_imag(w,p))

    G_imag = extrapolate(interpolate((w_grid, p_grid), imag(Γ), Gridded(Linear())), Flat())
    G(w,p)= complex(0.0, G_imag(w,p))

    #I1 = Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)
    #I2 = Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)

    I3 = Term3.Int3(Σ, Γ, T, m2, p_grid, term3_constant,  logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)

    Threads.@threads for i = 1:NUM_NODES_MOM
        
        #outer momentum p
        
        for j = 1:NUM_NODES_FREQ
            
            I2, G = Term2.Int2(Σ, Γ, p_grid[i], w_grid1[j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term2_constants, Γ_constant, γ_χ, Γ0)
            I1 = Term1.Int1(S, G, p_grid[i], w_grid1[j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term1_constants, γ_χ, Γ0)

            Σ_new1[j,i]=   I3  + I1 + I2 

        end

    end
    
    Σ_new2 = reverse(conj.(Σ_new1), dims=1) 
    Σ_new3= vcat(Σ_new2, Σ_new1)

    Σ_new= complex.(real(Σ_new3), imag(Σ_new3))
    
    return Σ_new, Γ

end

#C1= complex.(-real(A1[1]), imag(A1[1]))


function spectral(Σ ,Γ, T, m2)

    F(i,j) = Γ[i+64,j] /((w_grid1[i]- Γ0 * imag(Σ[i+64,j]))^2 + Γ0^2 * (real(Σ[i+64,j]) + m2 + p_grid[j]^2)^2)
    
    R=Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)
    
    for i in 1:NUM_NODES_FREQ
        for j in 1:NUM_NODES_MOM
            R[i,j]= F(i,j) * w_grid1[i]/(2*pi*im*T)
        end
    end
    
    return real(R)
    
end

function mass(Σ, Γ, m2, T, w_grid, logfreqWeights, P0 = true)

    if P0

        #Extra/Interpolation
        S_imag = extrapolate(interpolate((w_grid, p_grid), imag(Σ), Gridded(Linear())), Flat())
        S_real = extrapolate(interpolate((w_grid, p_grid), real(Σ), Gridded(Linear())), Flat())
        S(w,p)=complex(S_real(w,p), S_imag(w,p))
        
        G_imag = extrapolate(interpolate((w_grid, p_grid), imag(Γ), Gridded(Linear())), Flat())
        G(w,p)= complex(0.0, G_imag(w,p))
        
        #Initialize output Arrays
        Σ_new1 = Vector{ComplexF64}(undef, NUM_NODES_FREQ)
        Γ_new1 = Vector{ComplexF64}(undef, NUM_NODES_FREQ)
        
        I3 = Term3.Int3(Σ, Γ, T, m2, p_grid, term3_constant,  logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)

        for j = 1:NUM_NODES_FREQ
        
            #outer frequency w
                    
            I2, Γ_new1[j] = Term2.Int2(Σ, Γ, 0, w_grid1[j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term2_constants, Γ_constant, γ_χ, Γ0)
            I1 = Term1.Int1(S, G, 0, w_grid1[j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term1_constants, γ_χ, Γ0)
        
            Σ_new1[j]=  I1 + I2 + I3
        
        end

        Σ_new2 = reverse(conj.(Σ_new1)) 
        Σ_new= vcat(Σ_new2, Σ_new1)

        Γ_new2 = reverse(Γ_new1) 
        Γ_new = vcat(Γ_new2, Γ_new1)

    else 

        Γ_new= Γ[:, 1]
        Σ_new= Σ[:, 1]

    end


    int_mass = zeroC

    for f in 1:(2*NUM_NODES_FREQ)
    
        F = Γ_new[f]/((w_grid[f] + imag(Σ_new[f]))^2 + Γ02 * ( -m2 + real(Σ_new[f]))^2)

        int_mass += logfreqWeights[f] * F 

    end

    return 2 * pi * im * T/int_mass 

end


#Test 1: positive mass
m2= 1;
T= 15;

Σ = fill(0.0+0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);
Γ = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);

A=[];
B=[];
m=[];
#I1,I2,I3, Γ_new1 = Integrate(Σ, Γ, T, m2)
#scatter(w_grid1, real(I2[:,1]), label="I1", xlabel="ω", ylabel="I1", xscale = :log10, xticks=(10.0 .^ (-4:6)))

for i in 1:15

    @time Σ_new2, Γ_new2= Integrate(Σ, Γ, T, m2)
    
    println(length(A)+1 , " Iteration:",real(mass(Σ_new2, Γ_new2, m2, T, w_grid, logfreqWeights, false)))

    append!(m, real(mass(Σ_new2, Γ_new2, m2, T, w_grid, logfreqWeights, false)))
    append!(A,[Σ_new2])
    append!(B,[Γ_new2])
    
    Σ=Σ_new2
    Γ=Γ_new2

end 

i=10;
scatter(w_grid1, spectral(A[i] , B[i], T, m2)[:,1], label="ρ", xlabel="ω", ylabel="ρ", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(A[i][65:end,1]), xlabel="ω", ylabel="Im(Σ)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, real(A[i][65:end,1]), xlabel="ω", ylabel="Re(Σ)", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(B[i][65:end,1]), xlabel="ω", ylabel="Γ", xscale = :log10, xticks=(10.0 .^ (-4:6)))





folder="Results_20_5_2024/"

plt = Plots.bar(1:length(m), m, label="", title= "M^2 per iteration(non-critical)")

savefig(plt, folder * "Spectralfunctions.pdf")
# Test 2: delete dynamics

m2= -1;
T= 12.5;

#term3_constant =   +(im/2)* λ/3  * 1/(2*pi)^3 #* (N+2)/N

Σ2 = fill( -0.01 + 0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);
Γ2 = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);
m_2=[];

@time for i in 1:100

    x = Term3.Int3(Σ2, Γ2, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)
    Σ2 = fill(x, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);

    append!(m_2, mass(Σ2, Γ2, m2, T, w_grid, logfreqWeights, false))

end


plt = Plots.bar(1:length(m_2), m_2, label="", title= "M^2 per iteration(non-critical)")
#we expect a value of M^2 = 0.028 at T=12.5

#Test 3: small g

g= 1
m2= -1;
T= 13;

Σ1 = fill(-0.01 + 0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);
Γ1 = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);



Sigma, Γ_new2 = Integrate(Σ1, Γ1, T, m2)
scatter(w_grid1, imag(Γ_new2[65:end,1]), xlabel="ω", ylabel="Γ", xscale = :log10, xticks=(10.0 .^ (-4:6)))


Term2.Int2(Σ, Γ_new2, 0.1, 0.01, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term2_constants, Γ_constant, γ_χ, Γ0)
Term1.Int1(S, G, 0.1, 0.01, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, angularNodes, angularWeights, NUM_NODES_MOM, NUM_NODES_FREQ, NUM_NODES_ANGULAR, term1_constants, γ_χ, Γ0)
Term3.Int3(Σ, Γ_new2, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)

Σ1 = fill(-0.01 + 0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);
Γ1 = fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM);

A1=[];
B1=[];

for i in 1:200
 
    @time Σ_new2, Γ_new2 = Integrate(Σ1, Γ1, T, m2)
    
    println(length(A1)+1 , " Iteration:" ,real(mass(Σ_new2, Γ_new2, m2, T, w_grid, logfreqWeights, false)))
    
    append!(A1,[Σ_new2])
    append!(B1,[Γ_new2])
    
    Σ1=Σ_new2
    Γ1=Γ_new2
    
end 

j=31
scatter(w_grid1, spectral(A1[j] , B1[j], T, m2)[:,1], label=j, xlabel="ω", ylabel="ρ", scale = :log10, xticks=(10.0 .^ (-5:4)), yticks=(10.0 .^ (-6:1)))
scatter(w_grid1, imag(A1[j][65:end,1]), xlabel="ω", ylabel="Im(Σ)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, real(A1[j][65:end,1]), xlabel="ω", ylabel="Re(Σ)", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(B1[j][65:end,1]), xlabel="ω", ylabel="Γ", xscale = :log10, xticks=(10.0 .^ (-4:6)))



plot()
for j in 1:11
    scatter!(w_grid1, spectral(A1[j] , B1[j], T, m2)[:,1], label=j, xlabel="ω", ylabel="ρ", scale = :log10, xticks=(10.0 .^ (-5:4)), yticks=(10.0 .^ (-6:1)))
end
plot!()

σ=  4/3

f(x)=2.3* x^(-σ)

Plots.plot!(f, 10, 1000)
