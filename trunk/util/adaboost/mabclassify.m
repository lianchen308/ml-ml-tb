%   Classify Implements classification data samples by already built
%   boosted commitee
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
%    result = mabclassify(learners, weights, X)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           learners - cell array of learners
%           weights  - vector of learners weights
%           X        - Data to be classified. Should be DxN matrix, 
%                       where D is the dimensionality of data, and N 
%                       is the number of data samples.
%    Return:
%           result   - vector of real valued commitee outputs for Data. 

function result = mabclassify(learners, weights, X, reescale)

result = 0;

for i = 1 : length(weights)
  lrn_out = feval(learners{i}.predict, learners{i}.model, (X));
  result = result + lrn_out * weights(i);
end

if (exist('reescale', 'var') && reescale == 1)
    result(result < 0) = result(result < 0)/abs(min(result));
    max_val = max(result);
    if (max_val > 0)
        result(result > 0) = result(result > 0)/max(result);
    else
        result = result + 1;
        result = ((result/max(result)) - 0.5)*2;
    end
end
