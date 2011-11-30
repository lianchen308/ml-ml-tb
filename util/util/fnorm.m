%fnorm Normalizes the features in X 
%	usages fnorm(X), fnorm(X, dim), fnorm(X, mu, sigma), fnorm(X, mu, sigma, dim)
%   fnorm returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%	Params:
%	mu - mean value along dimension dim
%	sigma - std value along dimension dim
%	dim - dimension to normalize, default value is 1
function [X_norm, mu, sigma] = fnorm(X, varargin)
	nargs = length(varargin);
	dim = 1;
    if (nargs == 1)
        dim = varargin{1};
    elseif (nargs == 3)
        dim = varargin{3};
    end
	
    if (nargs < 2)
        mu = mean(X, dim);
        sigma = std(X, 0, dim);
    else
        mu = varargin{1};
        sigma = varargin{2};
    end

    sz = [2 1];
    sz = size(X, sz(dim));
    X_norm = zeros(size(X));
    for i = 1:sz
        if (dim == 1)
            X_norm(:, i) = (X(:, i) - mu(:, i))/sigma(:, i);
        else
            X_norm(i, :) = (X(i, :) - mu(i, :))/sigma(i, :);
        end
    end

end
