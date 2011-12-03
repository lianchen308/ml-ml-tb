% function [model] = svmtrainw(X, y, C, gamma, weights_or_cv, cv)
function [model] = svmtrainw(X, y, C, gamma, weights_or_cv, cv)

	if (exist('weights_or_cv', 'var') && ~isempty(weights_or_cv))
		if (isscalar(weights_or_cv))
			cv = weights_or_cv;
		else
			weights = weights_or_cv;
		end
    end
			
    if (~exist('weights', 'var') || isempty(weights))
        weights = ones(size(y));
    end
	
    if (~exist('cv', 'var') || isempty(cv))
        cv = 0;
    end
	
    if (~exist('weigths', 'var') || isempty(weights))
        weights = ones(size(y));
    end
	
	
	options = sprintf('-q -h 0 -m 1024 -b 1 -c %s -g %s', num2str (C), num2str (gamma));
	if (cv > 0) 
		options = sprintf('-v %d %s', cv, options);
	end
	[model] = libsvmtrain(weights, y, X, options);

end