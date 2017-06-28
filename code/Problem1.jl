using Images
using PyPlot
using Optim

function denoise(y,sigma_noise,sigma,alpha)
    x = copy(y);

    function f(x)
        return sum(mrf_denoise_nlposterior(reshape(x, size(y)),y, sigma_noise, sigma, alpha));
    end

    function g!(x, storage)
        dx = grad_mrf_denoise_nlposterior(reshape(x, size(y)), y, sigma_noise, sigma, alpha)
        storage[:] = dx[:];
    end

    options = Optim.Options(iterations=100, show_trace=false);
    result = optimize(f, g!, x, GradientDescent(), options);
    return reshape(Optim.minimizer(result), size(y));
end

x = 255.*channelview(Gray.(float64.(load("../data/la.png"))))
figure(1);
imshow(x);
y = add_noise(x,15.0)
figure(2);
imshow(y);
x_res = denoise(y,15.0,sigma,alpha)
