function [bestEpsilon bestScore] = selectThreshold(yval, pval, scoreFcn)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%
    bestEpsilon = 0;
    bestScore = 0;
    stepsize = (max(pval) - min(pval)) / 1000;
    for epsilon = min(pval):stepsize:max(pval)

        ypred = zeros(size(yval));
        ypred(pval < epsilon) = 1; 
        ypred(pval >= epsilon) = 0;
        [score] = feval(scoreFcn, yval, ypred);

        if score > bestScore
           bestScore = score;
           bestEpsilon = epsilon;
        end
    end

end
