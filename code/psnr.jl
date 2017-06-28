function psnr(X::Array{Float64,2}, Y::Array{Float64,2})
	M,N = size(X);
	vmax = maximum(X);
	MSE = sum((X-Y).^2)/M/N;
	p = 10log10(vmax^2/MSE);

	return p::Float64
end
