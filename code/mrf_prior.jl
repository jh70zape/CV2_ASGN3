function mrf_nlprior(x::Array{Float64,2}, sigma::Float64, alpha::Float64)

      h,w = size(x)

      # horizontal compatibility
      ph = [studentt(x[:,2:end]-x[:,1:end-1], sigma, alpha) zeros(h,1)];

      # vertical compatibility
      pv = [studentt(x[2:end,:]-x[1:end-1,:], sigma, alpha); zeros(1,w)];

      return -(ph+pv);
end

function grad_mrf_nlprior(x::Array{Float64,2}, sigma::Float64, alpha::Float64)
  height,width = size(x);

  # horizontal compatibility
  h  = x[:,1:end-1] - x[:,2:end];
  dh = grad_studentt(h, sigma, alpha);
  ph = hcat(dh, zeros(height,1)) - hcat(zeros(height,1), dh);

  # vertical compatibility
  v  = x[1:end-1,:] - x[2:end,:];
  dv = grad_studentt(v, sigma, alpha);
  pv = vcat(dv, zeros(1,width)) - vcat(zeros(1,width), dv);

    return -(ph+pv);
end
