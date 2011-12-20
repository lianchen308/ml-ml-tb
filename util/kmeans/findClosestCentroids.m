function idx = findClosestCentroids(X, centroids, is_discrete)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

    if (~exist('is_discrete', 'var') || isempty(is_discrete))
        is_discrete = [];
    end

    [dist] = pdistmixed(X, centroids, is_discrete);
    [~, idx] = min(dist, [], 2);

% =============================================================

end

