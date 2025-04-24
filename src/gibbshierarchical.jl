module gibbshierarchical
export modele_gibbs
using Distributions
using Random
using Plots
using ParetoSmooth
using PDMats
using LinearAlgebra
using Statistics
using StatsBase 
using SpecialFunctions  

function loglik_vector(Y, Sigma::Array{Float64,3})
    I = length(Y)  # Nombre de groupes
    T = size(Y[1], 1)  # Nombre d'observations temporelles
    p = size(Y[1], 2)  # Dimension des données
    
    # Initialisation de la matrice des log-vraisemblances
    loglik_mat = zeros(T, I)
    
    for i in 1:I
        # Extraction de la matrice de covariance pour le groupe i
        Sigma_i = @view Sigma[:,:,i]
        
        # Création de la distribution multivariée normale
        dist = MvNormal(zeros(p), Symmetric(Sigma_i))  # moyenne nulle
        
        # Calcul des log-densités pour chaque observation
        for t in 1:T
            loglik_mat[t, i] = logpdf(dist, @view Y[i][t,:])
        end
    end
    
    return vec(loglik_mat)  # Vectorisation colonne-major
end

function log_gamma_multivariate(a::Real, D::Int)
    sum(lgamma.(a .+ 0.5 .* [1 .- i for i in 1:D])) + 0.25 * D * (D - 1) * log(pi)
end

function sample_nu_mh(Sigma::Array{Float64}, Phi::Matrix{Float64}, nu_current::Int, 
    lambda::Real, nu_max::Int)
    D=size(Phi,1)
    function log_posterior(nu)
        (nu < D+2 || nu > nu_max) && return -Inf

        I = size(Sigma, 3)
        log_det_Phi = logdet(Phi)
        prior_term = (nu - (D + 2)) * log(lambda - D) - lgamma(nu - (D + 2) + 1)
        gamma_terms = I * ((nu/2) * log_det_Phi - (nu*D/2)*log(2) - log_gamma_multivariate(nu/2, D))

        # Somme des log-déterminants (version optimisée pour array 3D)
        log_det_Sigma_sum = sum(logdet(Sigma[:,:,i]) for i in 1:I)
        sigma_term = -0.5 * (nu + D + 1) * log_det_Sigma_sum

        return prior_term + gamma_terms + sigma_term
    end

    # Proposition symétrique (±1)
    nu_proposed = nu_current + rand((-1,1))

    # Ratio d'acceptation
    log_alpha = log_posterior(nu_proposed) - log_posterior(nu_current)

    # Acceptation/rejet
    if log(rand()) < log_alpha
        return nu_proposed
    else
        return nu_current
    end
end

function modele_gibbs(data::Array{Float64,3}, nu_0::Int, lambda::Int, Phi_0::Matrix{Float64};
    nu_max::Int=100, max_iter::Int=2000, fix::Bool=false, 
    savesigma::Bool=false)

    # Dimensions des données
    I, T, D = size(data)

    # Initialisation des paramètres
    nu = lambda
    Phi = copy(Phi_0)
    Sigma = Array{Float64}(undef, D, D, I)
    Y = [view(data, i, :, :) for i in 1:I]

    # Structures pour stockage des résultats
    chain_nu = zeros(max_iter)
    chain_Phi = Array{Float64}(undef, D, D, max_iter)
    loglik_vec = zeros(max_iter, I*T)

    if savesigma
        chain_Sigma = Array{Float64}(undef, D, D, I, max_iter)
    end


    tstart = time()

    for iter in 1:max_iter
    # 1. Mise à jour des Sigma_i
        for i in 1:I
            S = Y[i]' * Y[i]
            Sigma[:,:,i] = rand(InverseWishart(nu + T, Phi + S))

            savesigma && (chain_Sigma[:,:,i,iter] = Sigma[:,:,i])
        end

        # Calcul des log-vraisemblances
        #loglik_vec[iter,:] = lolog_det_Sigma_sum = sum(logdet(S) for S in Sigma)glik_vec(Y, Sigma)

        # 2. Mise à jour de Phi
        sum_Sinv = Symmetric(sum(inv.(eachslice(Sigma, dims=3))))
        Phi = rand(Wishart(nu_0 + I*nu, Matrix(inv(inv(Symmetric(Phi_0)) + sum_Sinv))))

        # 3. Mise à jour de nu
        nu = sample_nu_mh(Sigma,Phi,nu,lambda,nu_max)#sample_nu(Sigma, Phi, lambda, nu_max, fix)

        # Stockage des valeurs
        chain_nu[iter] = nu
        chain_Phi[:,:,iter] = Phi
        loglik_vec[iter,:] = loglik_vector(Y, Sigma)
        end

    tend = time()

        # Construction du résultat sous forme de NamedTuple
    result = (
    chain_nu = chain_nu,
    chain_Phi = chain_Phi,
    time = (tstart, tend),
    loglik_vec = loglik_vec,
    chain_Sigma = savesigma ? chain_Sigma : nothing
    )

    return result
end

end


