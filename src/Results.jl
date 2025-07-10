using DelimitedFiles
using FastGaussQuadrature
using QuadGK
using Plots
using GLMakie       #3D Plots


IR_CUTOFF = 0.01     
UV= 10

NUM_NODES = 32

# Define Grid, on which it was calculated
loggrid, logweight = gauss(NUM_NODES, log(IR_CUTOFF), log(10))

grid1= exp.(loggrid)





folder="ResultsT0_75N30"

Σ_new10= readdlm(folder *"/Sigma.txt", ',', Complex{Float64})
Γ_new10 = readdlm(folder *"/Gamma.txt", ',', Complex{Float64})


#Γ


plt = scatter(grid1, imag(Γ_new10[:,1]), xscale = :log10, xlabel="ω", ylabel="Im(Γ)", title= "p=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt, folder *"/Gamma_p" * string(IR_CUTOFF) * ".pdf")

plt = Plots.scatter(grid1, imag(Γ_new10[:,32]), xscale = :log10, xlabel="ω", ylabel="Im(Γ)", title= "p=" * string(UV), label = "")
Plots.savefig(plt, folder *"/Gamma_p" * string(UV) * ".pdf")

plt = Plots.scatter(grid1, imag(Γ_new10[1,:]), xscale = :log10, xlabel="p", ylabel="Im(Γ)", title= "ω=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt, folder *"/Gamma_w" * string(IR_CUTOFF) * ".pdf")

plt = Plots.scatter(grid1, imag(Γ_new10[32,:]), xscale = :log10, xlabel="p", ylabel="Im(Γ)", title= "ω=" * string(UV), label = "")
Plots.savefig(plt, folder *"/Gamma_w" * string(UV) * ".pdf")

#im(Σ)
plt2 = Plots.scatter(grid1, imag(Σ_new10[:,1]), xscale = :log10, xlabel="ω", ylabel="Im(Σ)", title = "p=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt2, folder *"/imagSigma_p" * string(IR_CUTOFF) * ".pdf")

plt2 = Plots.scatter(grid1, imag(Σ_new10[:,32]), xscale = :log10, xlabel="ω", ylabel="Im(Σ)" , title="p=" * string(UV), label = "")
Plots.savefig(plt2, folder *"/imagSigma_p" * string(UV) * ".pdf")

plt2 = Plots.scatter(grid1, imag(Σ_new10[1,:]), xscale = :log10, xlabel="p", ylabel="Im(Σ)", title="ω=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt2, folder *"/imagSigma_w" * string(IR_CUTOFF) * ".pdf")

plt2 = Plots.scatter(grid1, imag(Σ_new10[32,:]), xscale = :log10, xlabel="p", ylabel="Im(Σ)", title="ω=" * string(UV), label = "")
Plots.savefig(plt2, folder *"/imagSigma_w" * string(UV) * ".pdf")


#real(Σ)

plt3 = Plots.scatter(grid1, real(Σ_new10[:,1]), xscale = :log10, xlabel="ω", ylabel="real(Σ)", title = "p=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt3, folder *"/realSigma_p" * string(IR_CUTOFF) * ".pdf")

plt3 = Plots.scatter(grid1, real(Σ_new10[:,32]), xscale = :log10, xlabel="ω", ylabel="real(Σ)", title="p=" * string(UV), label = "")
Plots.savefig(plt3, folder *"/realSigma_p" * string(UV) * ".pdf")

plt3 = Plots.scatter(grid1, real(Σ_new10[1,:]), xscale = :log10, xlabel="p", ylabel="real(Σ)", title="ω=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt3, folder *"/realSigma_w" * string(IR_CUTOFF) * ".pdf")

plt3 = Plots.scatter(grid1, real(Σ_new10[32,:]), xscale = :log10, xlabel="p", ylabel="real(Σ)", title="ω=" * string(UV), label = "")
Plots.savefig(plt3, folder *"/realSigma_w" * string(UV) * ".pdf")

#######

grid_outer_mom= exp.(range(log(0.01),log(10), NUM_NODES_OUTER_MOM))
grid_outer_freq= exp.(range(log(0.01),log(10000), NUM_NODES_OUTER_FREQ))



#calculate Greensfunctions:
Γ0=1
m=im
T=0.75   


GR(i,j)= -1/(im * Γ0 * grid_outer_freq[i] - m^2 - grid_outer_mom[j]^2 - Σ_new10[i,j]) #i:w , j:p

GA(i,j) = -1/(-im * Γ0 * grid_outer_freq[i] - m^2 - grid_outer_mom[j]^2 - (real(Σ_new10[i,j])- im * imag(Σ_new10[i,j]))) #i:w , j:p

#spectral function
ρ(i,j)= GR(i,j) * Γ_new10[i,j] * GA(i,j) * grid_outer_freq[i]/(2*pi*im* T)

ρ(1,1)

R=Array{ComplexF64}(undef, NUM_NODES, NUM_NODES)

for i in 1:32
    for j in 1:32
        R[i,j]=ρ(i,j)
    end
end

plt4= Plots.scatter(grid1, real(R[:, 1]), scale = :log10, xlabel="ω", ylabel="Re ρ ", title="p=" * string(IR_CUTOFF), label = "")
#Plots.savefig(plt4, folder *"/real_spectralfunctionp_" * string(IR_CUTOFF) * ".pdf")

plt4= Plots.scatter(grid1, imag(R[:, 1]), xscale = :log10, xlabel="ω", ylabel="Im(ρ)", title="p=" * string(IR_CUTOFF), label = "")
Plots.savefig(plt4, folder *"/imag_spectralfunctionp_" * string(IR_CUTOFF) * ".pdf")


##3D

R[1:10, 1:10]
grid1[1:16]

GLMakie.activate!()

fig= Figure(resolution = (800, 600))
ax = Axis3(fig[1, 1], xlabel = "w", ylabel = "p", zlabel = "Im(Σ)")

GLMakie.scatter!(ax, grid1[1:16], grid1[1:16], real(R[1:16, 1:16]))

display(fig)