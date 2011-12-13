% [ knn_model ] = knntrain(X, y, K)
% knnargs is in the same format of knnimpute
function [ knn_model ] = knntrain(X, y, K)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
    end
    knn_model.data = [X y];
    knn_model.K = K;
end

