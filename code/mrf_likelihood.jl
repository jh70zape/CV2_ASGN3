function mrf_denoise_nllh(x::Array{Float64,2}, y::Array{Float64, 2}, sigma)

    return (x-y).^2/2/sigma^2;
end

function grad_mrf_denoise_nllh(x::Array{Float64,2}, y::Array{Float64,2}, sigma)

    return (x-y)/sigma^2;
end
