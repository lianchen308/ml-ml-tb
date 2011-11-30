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

function result = mabclassify(learners, weights, X)

result = zeros(1, size(X, 2));

for i = 1 : length(weights)
  lrn_out = learners{i}.predict(X);
  result = result + lrn_out * weights(i);
end