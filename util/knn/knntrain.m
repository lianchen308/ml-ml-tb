% [ knn_model ] = knntrain(X, knnargs )
% knnargs is in the same format of knnimpute
function [ knn_model ] = knntrain(X, y, K, distance_fn)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
    end
    if (~exist('distance_fn', 'var') || isempty(distance_fn))
        distance_fn = 'euclidean';
    end
    knn_model.data = [X y];
    knn_model.distance_fn = distance_fn;
    knn_model.K = K;
end

