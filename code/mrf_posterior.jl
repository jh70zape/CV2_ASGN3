function mrf_denoise_nlposterior(x::Array{Float64,2},y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)

  prior = mrf_nlprior(x, sigma, alpha);
  lh    = mrf_denoise_nllh(x, y, sigma_noise);
  post  = prior+lh;

    return post;
end

function grad_mrf_denoise_nlposterior(x::Array{Float64,2}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)

  grad_prior = grad_mrf_nlprior(x, sigma, alpha);
  grad_lh    = grad_mrf_denoise_nllh(x, y, sigma_noise);
  grad_post  = grad_prior+grad_lh;

   return grad_post;

end

function mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)

    return m.* mrf_nlprior(x, sigma, alpha);
end

function grad_mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)

    return m.* grad_mrf_nlprior(x,sigma,alpha);
end
