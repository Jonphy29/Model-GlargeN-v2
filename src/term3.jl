module Term3

export Int3

function Int3(Σ, Γ, T, m2, p_grid, term3_constant, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, Γ0, Γ02)

    #define complex zero
    zeroC = complex(0.0, 0.0)

    # initialize final integrals(output variables)
    int_term3 = zeroC

    # for loop corresponding to the inner momentum (k) integration
    for r in 1:NUM_NODES_MOM

        # momentum and corresponding weight
        k = p_grid[r] 
        k_weight=logmomWeights[r]

        # initialize variable to save the frequency integrals
        int_term3_freq = zeroC

        #constants
        k2=k*k
        m2k2= m2 + k2

        for f in 1:(2*NUM_NODES_FREQ)

            # frequency and corresponding weight
            w1 = w_grid[f]
            w1_weight=logfreqWeights[f]

            # angular independent parts of the integral
                
            term= im * (w1 + imag(Σ[f, r])) + Γ0* (-real(Σ[f, r]) + m2k2)
    
            #term1_ang_indep= 1#/term_ang_indep
            term3_ang_indep= Γ[f,r]/abs2(term)
            #term3_ang_indep = Γ[f,r]/((w1 + Γ0 * imag(Σ[f,r]))^2 + Γ02 * (-real(Σ[f,r]) + m2k2)^2)

            int_term3_freq += 2 * w1_weight * term3_ang_indep 

        end

        #Momentum integral: function(=freq. integral) * weight
        int_term3 += int_term3_freq * k_weight * term3_constant * k2

    end

    return int_term3

end;


end