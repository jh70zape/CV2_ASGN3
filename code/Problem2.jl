using Images
using PyPlot
using Optim

function impaint(x0,m,sigma,alpha)
  x = copy(x0);

  function f(x)
      return sum(mrf_inpaint_nlposterior(reshape(x, size(m)), m,sigma, alpha));
  end

  function g!(x, storage)
      dx = grad_mrf_inpaint_nlposterior(reshape(x, size(m)), m, sigma, alpha)
      storage[:] = dx[:];
  end

      options = Optim.Options(iterations=500, show_trace=false);
      result = optimize(f, g!, x, GradientDescent(), options);
      return reshape(Optim.minimizer(result), size(left));
end
x=255.*channelview(Gray.(float64.(load("../data/castle.png"))));
n = length(x)
ind = randperm(n)[1:n*0.5]
x[randperm(n)[1:p]] = 127;
M = ones(size(x))
M(ind) = 0.2;
sigma = 10.0; alpha = 1.0
x_res = impaint(x_res,M,sigma,alpha);
