% gaussiantrain(X, y, score_fcn)
function [ gaussian_model ] = gaussiantrain(X, y, score_fcn)
    if (~exist('score_fcn', 'var') || isempty(score_fcn))
        score_fcn = 'aucscore';
    end
    
    [mu sigma2] = estimateGaussian(X);
    gaussian_model.mu = mu;
    gaussian_model.sigma2 = sigma2;
    prob = multivariateGaussian(X, mu, sigma2);
    gaussian_model.thres = selectThreshold(y, prob, score_fcn);
end

