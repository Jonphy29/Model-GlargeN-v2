# Model G - a large N expansion

## 1. Theory
 In the Hohenberg-Halpering classification, Model G is one of the universality classes of dynamic critical behavior. They originally formulated it for an $O(3)$
 isotropic Heisenberg ferromagnet, which has a three component order parameter, reversibly coupled to a 3 component magnetization. In this work, it is used a generalization 
 to a $O(N)$ symmetry, for which it is considered the limit $N\to \infty$. For this, one can derive some Dyson-Schwinger equations:
 
$$
\Sigma^R(\omega, p) = \int_0^\Lambda\frac{dk}{(2\pi)^3}\int_{-\infty}^\infty d\omega' k^2 \left( SR1 + SR2 + TP \right)
$$

$$
\Gamma^{\tilde{\phi}\tilde{\phi}}(\omega, p) = (\omega, p)= 2i\Gamma_0T + \frac{4\gamma T g^2(N-1)}{\chi^2 N}\int_0^\Lambda \frac{dk}{(2\pi)^3}\int_{-\infty}^\infty d\omega'  k^2 (SR3)
$$

with 

$$
 SR1 = -\frac{4\gamma Tg^2(N-1)}{\chi^2 N}\frac{C_1(\omega,\omega', p,k)}{i(\omega'+Im(\Sigma_R(\omega', k))-\Gamma_0(-Re(\Sigma_R(\omega', k))+m^2+ k^2)}
$$

$$
 SR2= \frac{i}{2}\frac{4g^2(N-1)}{\chi N} \frac{C_2(\omega,\omega', p,k) \Gamma^{\tilde{\phi}\tilde{\phi}}(\omega', k)}{(\omega'+\Gamma_0Im(\Sigma_R(\omega', k)))^2+\Gamma_0^2(-Re(\Sigma_R(\omega', k))+m^2+ k^2)^2}
$$

$$
 TP =\frac{i}{2}\frac{\lambda(N+2)}{3N}\frac{2 \Gamma^{\tilde{\phi}\tilde{\phi}}(\omega', k)}{(\omega'+\Gamma_0Im(\Sigma_R(\omega', k)))^2+\Gamma_0^2(-Re(\Sigma_R(\omega', k))+m^2+ k^2)^2}
$$

$$
 SR3 = \frac{C_1(\omega,\omega', p,k)\Gamma^{\tilde{\phi}\tilde{\phi}}(\omega', k)}{(\omega'+\Gamma_0Im(\Sigma_R(\omega', k)))^2+\Gamma_0^2(-Re(\Sigma_R(\omega', k))+m^2+ k^2)^2}
$$

and the functions (Log denotes the complex logarithm)

$$
    C_1(\omega,\omega', p,k) = \frac{1}{4\frac{\gamma}{\chi}pk}\ln \Bigl(\frac{\vert(\omega-\omega')^2+\frac{\gamma}{\chi}(p-k)^2\vert}{\vert(\omega-\omega')^2+\frac{\gamma}{\chi}(p+k)^2\vert}\Bigr)
$$

$$
    C_2(\omega,\omega', p,k) = \frac{k^2-p^2}{2\frac{\gamma}{\chi}pk}\text{Log}\Bigl(i(\omega-\omega')-\frac{\gamma}{\chi}(p^2+k^2-2pkz)\Bigr)|_{z=-1}^{z=1}
$$

To obtain a solution for $\Sigma^R$ and $\Gamma^{\tilde{\phi}\tilde{\phi}} $, one has to iterativly solve both integrals.

## 2. Implementation And Usage

This repository implements the integrals shown above by using Gauß-Legendre integration. Explanation of files and folders:

**\src**
  >**-integrate2D.jl** : returns the integrals over $SR1$, $SR2$ and $SR3$ for one given $p$ and $\omega$. 
  
  >**-int_theta_analytical.jl** : returns the matrices $C1$ and $C2$, and saves them in the folder \data. Note: since these binary files get large for a big grid, they are not saved, but rather have to be produced
                              seperatly. This will take some time, but only has to be done once, since they dont change per iteration.  
                              
  >**-term3.jl** : return the integral $TP$, which is independent of $\omega$ and $p$
  
  >**-main.jl** : Implements a grid for $p$ and $\omega$, and iterativly solves the integral to obtain $\Sigma^R$ and $\Gamma^{\tilde{\phi}\tilde{\phi}} $. Also one can compute the desired 
              spectral function, and some tests are implemented.
              
  >**-results.jl** : Not working at the moment! In the future, this will be there to obtain some resulting images. 

**\images**: all produces images are saved in this file.

**\data**: the matrices C1 and C2 are saved here.





