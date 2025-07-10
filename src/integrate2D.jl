module integrate2D

export int_freq_mom

function int_freq_mom(Σ, Γ, C1, C2, T, m2, p_grid, logmomWeights, w_grid, logfreqWeights, NUM_NODES_MOM, NUM_NODES_FREQ, term1_constants, term2_constants, Γ_constant, Γ0)
    #Σ: array of size (2 * NUM_NODES_OUTER_FREQ) * NUM_NODES_OUTER_MOM
    #Γ: array of size (2 * NUM_NODES_OUTER_FREQ) * NUM_NODES_OUTER_MOM
    #C1: array of size NUM_NODES_OUTER_MOM * (2 * NUM_NODES_OUTER_FREQ)
    #C2: array of size NUM_NODES_OUTER_MOM * (2 * NUM_NODES_OUTER_FREQ)
    #T: temperature
    #m2: mass squared
    #p_grid: momentum grid
    #logmomWeights: logarithmic measure for momentum integration
    #w_grid: frequency grid
    #logfreqWeights: logarithmic measure for frequency integration
    #NUM_NODES_MOM: number of nodes for momentum integration
    #NUM_NODES_FREQ: number of nodes for frequency integration
    #term1_constants: constant factor for term1
    #term2_constants: constant factor for term2
    #Γ_constant: constant factor for Γ term
    #Γ0: constant 
        
    #define complex zero
    zeroC = complex(0.0, 0.0)

    # initialize final integrals(output variables)
    int_term1 = zeroC
    int_term2 = zeroC
    int_Γ = zeroC
    
    # for loop corresponding to the inner momentum (k) integration
    for r in 1:NUM_NODES_MOM
    
        # momentum and corresponding weight
         k = p_grid[r] 
        k_weight = logmomWeights[r]
    
    
        # initialize variable to save the frequency integrals
        int_term1_freq = zeroC
        int_term2_freq = zeroC
        int_Γ_freq = zeroC
    
        # for loop corresponding to the inner frequency (w1) integration
        ## Note: Integration is from -ω_max to ω_max, so the loop is doubled
    
        #constants
        k2= k*k
        m2k2= Γ0 * (m2 + k2)
            
        for f in 1:(2*NUM_NODES_FREQ)
    
            # frequency and corresponding weight
            w1 = w_grid[f]
            w1_weight=logfreqWeights[f]
    
            term = im * (w1 + imag(Σ[f, r])) - Γ0 * (-real(Σ[f, r]) + m2k2)

            term1= C1[r, f] * 1/term
            term2 = C2[r, f] * Γ[f,r]/abs2(term)
            term_Γ = C1[r, f] * Γ[f,r]/abs2(term)
                
            int_term1_freq += term1 * w1_weight
            int_term2_freq += term2 * w1_weight  
            int_Γ_freq += term_Γ * w1_weight 
    
        end
            
        int_term1= int_term1_freq * k_weight * term1_constants * k2
        int_term2 += int_term2_freq * k_weight * term2_constants * k2
        int_Γ += int_Γ_freq * k_weight * Γ_constant * k2 
    
    end
    
    return int_term1 * T, int_term2, 2*Γ0*T*im + int_Γ * T
    
    end;
    

end