using gibbshierarchical
using Distributions
using Random
using Plots
using ParetoSmooth
using PDMats
using LinearAlgebra
using Statistics
using StatsBase 
using SpecialFunctions  
Random.seed!(123)
I = 100  # nombre d'individus
T = 100  # nombre d'observations
D = 5    # dimension des données
s = [1.0  0.0  0.4  0.0  0.2
     0.0  1.0  0.6  0.0  0.0
     0.4  0.6  1.0  0.2  0.0
     0.0  0.0  0.2  1.0  0.1
     0.2  0.0  0.0  0.1  1.0]

s2 = [1.0  0.7  0.0  0.4  0.0
      0.7  1.0  0.0  0.0  0.0
      0.0  0.0  1.0  0.0  0.0
      0.4  0.0  0.0  1.0  0.0
      0.0  0.0  0.0  0.0  1.0]

Phi_0= [1.0  0.  0.0  0.  0.0
0.  1.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0
0.  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  1.0]

# Préparation des données
Y = zeros(I, T, D)
slist = [s, s2]

for i in 1:I
    ind = i <= I/2 ? 1 : 2
    Sigma = rand(InverseWishart(30,slist[ind]))
    Y[i, :, :] = rand(MvNormal(zeros(D), Sigma), T)'
end

m=modele_gibbs(Y,7,7,Phi_0,nu_max=100)
plot(m.chain_nu)
m1=modele_gibbs(Y[1:50,:,:],7,7,Phi_0,nu_max=100)
plot(m1.chain_nu)
m2=modele_gibbs(Y[51:100,:,:],7,7,Phi_0,nu_max=100)
plot(m2.chain_nu)

loo_all=psis_loo(reshape(transpose(m.loglik_vec), size(transpose(m.loglik_vec))..., 1))
loglik_vcat = vcat(transpose(m1.loglik_vec), transpose(m2.loglik_vec))
loo_v=psis_loo(reshape(loglik_vcat, size(loglik_vcat)..., 1))
loo_compare(loo_v,loo_all)