% function [model] = svmtrainw(X, y, options)
function [model] = svmtrainw(X, y, opt)

    if (~exist('opt', 'var') || isempty(opt))
        opt.dummy = 1;
    end

    if (~isfield(opt, 'c'))
        opt.c = 1;
    end
    
    if (isfield(opt, 'weights'))
        opt.weights = weights;
    else
        opt.weights = ones(size(y));
    end
    
    if (isfield(opt, 'pos_weights'))
        opt.weights(y == 1) = opt.pos_weights;
    end
    if (isfield(opt, 'neg_weights'))
        opt.weights(y == -1)  = opt.neg_weights;
    end

    
    if (~isfield(opt, 'g')) 
        opt.g = 1/size(X,2);
    end
    
    if (~isfield(opt, 'cv'))
        opt.cv = 0;
    end
	
	svmlibopt = sprintf('-q -h 1 -m 1024 -b 1 -c %s -g %s', num2str (opt.c), num2str (opt.g));
	if (opt.cv > 0) 
		svmlibopt = sprintf('-v %d %s', opt.cv, svmlibopt);
	end
	[model] = libsvmtrain(opt.weights, y, X, svmlibopt);

end