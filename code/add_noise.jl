function add_noise(X::Array{Float64,2}, sigma::Float64)

    M,N = size(X)
		noisy = zeros(M,N)
    @assert size(noisy) == size(X)
		noisy = X + rand(Normal(0,sigma),M,N)
	return noisy::Array{Float64,2}
end
