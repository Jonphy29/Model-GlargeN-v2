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

include("integrate2D.jl")
using .integrate2D

#test how many threads are available
Threads.nthreads()

#Integration cutoffs
IR_CUTOFF = 0.0001      #IR cutoff for momentum and frequency
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
term1_constants =   -(1/2) * 8 * γ * g^2/χ^2 * 1/(2*pi)^3 # *(N-1)/N
term2_constants =   +(im/2) * 4 * g^2/χ * 1/(2*pi)^3 #*(N-1)/N
term3_constant =   +(im/2)* λ/3  * 1/(2*pi)^3 #* (N+2)/N
Γ_constant=   4 * g^2 * γ * 1/χ^2 * 1/(2*pi)^3    #*(N-1)/N

#Grid

##dimensions(this needs to match the .bin files)
NUM_NODES_FREQ = 2^10;              #Number of nodes for the frequency integration
NUM_NODES_MOM= 2^5;                 #Number of nodes for the momentum integration   

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

#=
#For both cases: load them in the SSD first
c1_file = open("data/C1_w1024p32.bin")
num_nodes_pin = read(c1_file, Int)
num_nodes_win = read(c1_file, Int)
num_nodes_pout = read(c1_file, Int)
num_nodes_wout = read(c1_file, Int)
#length= num_nodes_pin * num_nodes_win * num_nodes_pout * num_nodes_wout

C1_chunk=Array{Float32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, num_nodes_wout);
@time read!(c1_file, C1_chunk)

close(c1_file)

C1_chunk[12,1,4,1]

#Define angular solution
c2_file = open("data/C2_w1024p32.bin")
num_nodes_pin = read(c2_file, Int)
num_nodes_win = read(c2_file, Int)
num_nodes_pout = read(c2_file, Int)
num_nodes_wout = read(c2_file, Int)
#length= num_nodes_pin * num_nodes_win * num_nodes_pout * num_nodes_wout
#idx = p_in + (w_in-1)*p + (p_out-1)*p*o + (w_out-1)*p*o*n 

C2_chunk=Array{ComplexF32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, num_nodes_wout)
@time read!(c2_file, C2_chunk)

delete!(C2_chunk)
close(c2_file)
=#
#Integration function



function Integrate_chunked(Σ, Γ, T, m2, Number_of_chunks)

    I3= Int3(Σ, Γ, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)
    #Initialize output Arrays
    Σ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    Γ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)


    c1_file = open("data/C1_w1024p32.bin")
    num_nodes_pin = read(c1_file, Int)
    num_nodes_win = read(c1_file, Int)
    num_nodes_pout = read(c1_file, Int)
    num_nodes_wout = read(c1_file, Int)

    c2_file = open("data/C2_w1024p32.bin")
    num_nodes_pin = read(c2_file, Int)
    num_nodes_win = read(c2_file, Int)
    num_nodes_pout = read(c2_file, Int)
    num_nodes_wout = read(c2_file, Int)

    
    C1_chunk=Array{Float32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, Int(num_nodes_wout/Number_of_chunks))
    C2_chunk=Array{ComplexF32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, Int(num_nodes_wout/Number_of_chunks))

    for k = 1:Number_of_chunks

        read!(c1_file, C1_chunk)
        read!(c2_file, C2_chunk)

        Threads.@threads for j = 1:(Int(num_nodes_wout/Number_of_chunks))
            for i = 1:NUM_NODES_MOM 
            I1, I2, Γ_new1[Int(j +  (k-1) * num_nodes_wout/Number_of_chunks),i] = integrate2D.int_freq_mom(Σ, Γ, C1_chunk[:,:,i,j], C2_chunk[:,:,i,j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)
            Σ_new1[Int(j+ (k-1) * num_nodes_wout/Number_of_chunks),i] = I1 + I2+ I3
            end
        end
    end

    close(c1_file)
    close(c2_file)

    return Σ_new1, Γ_new1

end 

function Integrate_each_term_chunked(Σ, Γ, T, m2, Number_of_chunks)

    #Initialize output Arrays
    I1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    I2 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    Γ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)

    c1_file = open("data/C1_w1024p32.bin")
    num_nodes_pin = read(c1_file, Int)
    num_nodes_win = read(c1_file, Int)
    num_nodes_pout = read(c1_file, Int)
    num_nodes_wout = read(c1_file, Int)

    c2_file = open("data/C2_w1024p32.bin")
    num_nodes_pin = read(c2_file, Int)
    num_nodes_win = read(c2_file, Int)
    num_nodes_pout = read(c2_file, Int)
    num_nodes_wout = read(c2_file, Int)

    
    C1_chunk=Array{Float32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, Int(num_nodes_wout/Number_of_chunks))
    C2_chunk=Array{ComplexF32}(undef, num_nodes_pin, num_nodes_win, num_nodes_pout, Int(num_nodes_wout/Number_of_chunks))

    for k = 1:Number_of_chunks

        read!(c1_file, C1_chunk)
        read!(c2_file, C2_chunk)

        Threads.@threads for j = 1:(Int(num_nodes_wout/Number_of_chunks))
            for i = 1:NUM_NODES_MOM   
                I1[Int(j+ (k-1) * num_nodes_wout/Number_of_chunks),i], I2[Int(j+ (k-1) * num_nodes_wout/Number_of_chunks),i], Γ_new1[Int(j+ (k-1) * num_nodes_wout/Number_of_chunks),i] = integrate2D.int_freq_mom(Σ, Γ, C1_chunk[:,:,i,j], C2_chunk[:,:,i,j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)
            end
        end
    end

    close(c1_file)
    close(c2_file)

    return I1, I2, Γ_new1

end 


function Integrate(Σ, Γ, T, m2)
    
    I3= Int3(Σ, Γ, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)
    #Initialize output Arrays
    Σ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    Γ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)


    Threads.@threads for i = 1:NUM_NODES_MOM 
        for j = 1:(2* NUM_NODES_FREQ)
            
            I1, I2, Γ_new1[j,i] = integrate2D.int_freq_mom(Σ, Γ, C1_chunk[:,:,i,j], C2_chunk[:,:,i,j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)
            Σ_new1[j,i] = I1 + I2+ I3
        end
    end

    #Γ_new2 = reverse(Γ_new1, dims=1) 
    #Γ_new = vcat(Γ_new2, Γ_new1)

    #Σ_new2 = reverse(conj.(Σ_new1), dims=1) 
    #Σ_new= vcat(Σ_new2, Σ_new1)

    return Σ_new1, Γ_new1
end

function Integrate_each_term(Σ, Γ, T, m2)
    
    #Initialize output Arrays
    I1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    I2 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)
    Γ_new1 = Array{ComplexF64}(undef, 2* NUM_NODES_FREQ, NUM_NODES_MOM)

    Threads.@threads for i = 1:NUM_NODES_MOM 
        for j = 1:(2* NUM_NODES_FREQ)
            
            I1[j,i], I2[j,i], Γ_new1[j,i] = integrate2D.int_freq_mom(Σ, Γ, C1_chunk[:,:,i,j], C2_chunk[:,:,i,j], T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)

        end
    end

    #Γ_new2 = reverse(Γ_new1, dims=1) 
    #Γ_new = vcat(Γ_new2, Γ_new1)

    #Σ_new2 = reverse(conj.(Σ_new1), dims=1) 
    #Σ_new= vcat(Σ_new2, Σ_new1)

    return I1, I2, Γ_new1
end


#Computes spectral function for w>0
function spectral(Σ ,Γ, T, m2)

    F(i,j) = Γ[i+NUM_NODES_FREQ,j] /((w_grid1[i] + Γ02 * imag(Σ[i+NUM_NODES_FREQ,j]))^2 + Γ02 * (-real(Σ[i+NUM_NODES_FREQ,j]) + m2 + p_grid[j]^2)^2)
    
    R=Array{ComplexF64}(undef, NUM_NODES_FREQ, NUM_NODES_MOM)
    
    for i in 1:NUM_NODES_FREQ
        for j in 1:NUM_NODES_MOM
            R[i,j]= F(i,j) * w_grid1[i]/(2*pi*im*T)
        end
    end
    
    return real(R)
    
end

function mass_eff(Σ ,Γ, T, m2)
    
    int_mass = zeroC

    for f in 1:(2*NUM_NODES_FREQ)
    
        F = Γ[f]/((w_grid[f] + Γ02 * imag(Σ[f]))^2 + Γ02 * ( +m2 - real(Σ[f]))^2)

        int_mass += logfreqWeights[f] * F 

    end

    return 2 * pi * im * T/int_mass 

end

#####
#Test
m2 = 1.0
T= 12.0

A=[]
B=[]

append!(A, [fill(0.0+0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)])
append!(B, [fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)])

#@time a1,b1 = Integrate_chunked(A[1], B[1], T, m2, 8)

scatter(w_grid1, real(a1[(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Re(I1)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))


#test term 3: Compare:

-λ * T/(12* pi^2)* (UV_CUTOFF - atan(UV_CUTOFF/sqrt(m2))) 

Int3(A[1], B[1], T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)

#test non-critical region

for i in 1:10

    a, b = Integrate_chunked(A[i], B[i], T, m2, 8)

    println(mass_eff(a , b , T, m2))

    append!(A, [a])
    append!(B, [b])

end

i=11
I1, I2, Γ_new =Integrate_each_term_chunked(A[i], B[i], T, m2, 8)

scatter(w_grid1, real(I1[(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Re(I1)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))
#savefig("images/w1024p32/Re_I1.pdf")
scatter(w_grid1, real(I2[(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Re(I2)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))#works i.g.
#savefig("images/w1024p32/Re_I2.pdf")
scatter(w_grid1, imag(I1[(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Im(I1)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))#not antisymmetric
#savefig("images/w1024p32/Im_I1.pdf")
scatter(w_grid1, imag(I2[(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Im(I2)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))#works
#savefig("images/w1024p32/Im_I2.pdf")

256*2
i=11

scatter(w_grid1, spectral(A[i] , B[i], T, m2)[:,1], label="ρ", xlabel="ω", ylabel="ρ", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(A[i][1:NUM_NODES_FREQ,1]), xlabel="ω", ylabel="Im(Σ)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(A[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Im(Σ)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))

scatter(w_grid1, real(A[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Re(Σ)", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(B[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Γ", xscale = :log10, xticks=(10.0 .^ (-4:6)))

#test for causality
for i in 1:NUM_NODES_FREQ

    if imag(A[11][i,1]) + w_grid1[i] < 0.0
        println("Problem at", w_grid1[i], " : ", imag(A[11][i,1]))
    end

end



#####

m2 = -1.0
T= 11
A1=[]
B1=[]

append!(A1, [fill(1.1+0.0*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)])
append!(B1, [fill(0.0+2*Γ0*T*im, 2 * NUM_NODES_FREQ, NUM_NODES_MOM)])

for i in 1:10

    a, b = Integrate(A1[i], B1[i], T, m2)
    println(mass_eff(a , b , T, m2))
    append!(A1, [a])
    append!(B1, [b])

end

i=11

scatter!(w_grid1, spectral(A1[i] , B1[i], T, m2)[:,1], label="ρ", xlabel="ω", ylabel="ρ", xscale = :log10, xticks=(10.0 .^ (-4:6)))
#scatter!(w_grid1, spectral(A[i] , B[i], 12, 1)[:,1], label="ρ", xlabel="ω", ylabel="ρ", xscale = :log10, xticks=(10.0 .^ (-4:6)))

scatter(w_grid1, imag(A1[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Im(Σ)", xscale = :log10,  xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, real(A1[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Re(Σ)", xscale = :log10, xticks=(10.0 .^ (-4:6)))
scatter(w_grid1, imag(B1[i][(NUM_NODES_FREQ+1):end,1]), xlabel="ω", ylabel="Γ", xscale = :log10, xticks=(10.0 .^ (-4:6)))


plot()
for j in 1:11
    scatter!(w_grid1, spectral(A1[j] , B1[j], T, m2)[:,1], label=j, xlabel="ω", ylabel="ρ", scale = :log10, xticks=(10.0 .^ (-5:4)), yticks=(10.0 .^ (-6:1)))
end
plot!()


scatter(w_grid1, spectral(A1[i] , B1[i], T, m2)[:,1], label="ρ", xlabel="ω", ylabel="ρ", scale = :log10, xticks=(10.0 .^ (-4:6)))


σ=  4/3

f(x)=0.9* x^(-σ)
plot!(f, 0.1, 100)



