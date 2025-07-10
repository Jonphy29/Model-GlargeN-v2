#This file computes the analytical theta integrals. It will create som e.bin files which are used in the main.jl file.
#It is necessary to run this file, and create the .bin files with the same grid size as in main.jl.
#Note: This takes a while, and also requires a lot of memory. 
using LinearAlgebra
using FastGaussQuadrature


#constants

const γ=1;
const χ=1;    
const γ_χ= γ/χ;

#define grid(as in main.jl)

NUM_NODES_FREQ = 2^7;     
NUM_NODES_MOM= 2^7;            

IR_CUTOFF = 0.0001;       
UV_CUTOFF = 10;        
ω_max = 10^6;         

logmomNodes, logmomWeights= gausslegendre(NUM_NODES_MOM)
logmomNodes = 0.5 * (logmomNodes .+ 1) * (log(UV_CUTOFF) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
p_grid = exp.(logmomNodes) 
                                            
logfreqNodes, logfreqWeights1 = gausslegendre(NUM_NODES_FREQ)
logfreqNodes = 0.5 * (logfreqNodes .+ 1) * (log(ω_max) - log(IR_CUTOFF)) .+ log(IR_CUTOFF)
w_grid1 = exp.(logfreqNodes)
w_grid2= -reverse(w_grid1);
w_grid= vcat(w_grid2, w_grid1); 

NUM_NODES_FREQ_doubled= 2 * NUM_NODES_FREQ; 
#define complex 0
zeroC = complex(0.0, 0.0)

#define analytic solution

z_plus(w, w1, p, k) = abs(im*(w-w1) + (γ_χ*(p+k)))
z_minus(w, w1, p, k) = abs(im*(w-w1) + (γ_χ*(p-k)))
phi_plus(w, w1, p, k) = (w-w1)/(γ_χ*(p+k)^2)
phi_minus(w, w1, p, k) = (w-w1)/(γ_χ*(p-k)^2)


C1_fct(w, w1, p, k) = 1/(4* γ_χ * p * k) * log(z_minus(w, w1, p, k)^2 / z_plus(w, w1, p, k)^2)
C2_fct(w, w1, p, k) = (k^2-p^2)/(2 * γ_χ * p * k) * (log(z_plus(w, w1, p, k) / z_minus(w, w1, p, k)) - im*atan(phi_plus(w, w1, p, k)) + im* atan(phi_minus(w, w1, p, k)))

#compute solutions on the grid

c1_file= open("data/C1_128.bin", "w") 

write(c1_file, NUM_NODES_FREQ_doubled)
write(c1_file, NUM_NODES_MOM)
write(c1_file, NUM_NODES_FREQ_doubled)
write(c1_file, NUM_NODES_MOM)



for w_out in 1:(NUM_NODES_FREQ_doubled)
    for p_out in 1:NUM_NODES_MOM
        for w_in in 1:(NUM_NODES_FREQ_doubled)
            for p_in in 1:NUM_NODES_MOM

                x = C1_fct(w_grid[w_out], w_grid[w_in], p_grid[p_out], p_grid[p_in])

                if isfinite(x) && !isnan(x)
                    write(c1_file, x)
                else
                    write(c1_file, 0.0)
                end
                  
            end
        end
    end
end

close(c1_file)

c2_file= open("data/C2_128.bin", "w") 

write(c2_file, NUM_NODES_FREQ_doubled)
write(c2_file, NUM_NODES_MOM)
write(c2_file, NUM_NODES_FREQ_doubled)
write(c2_file, NUM_NODES_MOM)

for w_out in 1:(NUM_NODES_FREQ_doubled)
    for p_out in 1:NUM_NODES_MOM
        for w_in in 1:(NUM_NODES_FREQ_doubled)
            for p_in in 1:NUM_NODES_MOM

                x = C2_fct(w_grid[w_out], w_grid[w_in], p_grid[p_out], p_grid[p_in])

                if isfinite(x) && !isnan(x)
                    write(c2_file, x)
                else
                    write(c2_file, zeroC)
                end
                  
            end
        end
    end
end

close(c2_file)