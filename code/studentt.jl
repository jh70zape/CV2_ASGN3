function studentt(d, sigma::Float64, alpha::Float64)
    return -alpha * log(1+d.^2/2*sigma^2); # log student t
end

function grad_studentt(d, sigma::Float64, alpha::Float64)
    return -2 * alpha*d./ (2sigma^2 + d.^2);
end
