% [y_pred, y_prob, acc, auc] = knnpredict(knn_model, X, y)
function [y_pred, y_prob, acc, auc] = knnpredict(knn_model, X, y)

    n_rows = size(X, 1);
    impute_data = [X (NaN*ones(n_rows, 1))];
    
    y_pred = knnimputeext([impute_data; knn_model.data], knn_model.K, knn_model.distance_fn);
    y_pred = y_pred(1:n_rows, end); 
    y_prob = (y_pred + 1)/2;
    
    if (~exist('y', 'var') || isempty(y))
        acc = -1;
        auc = -1;
    else
        y_class = zeros(size(y_pred));
        y_class(y_pred < 0)  = -1;
        y_class(y_pred >= 0) = 1;
        acc = (length(find(y_class == y))/length(y));
        auc = aucscore(y, y_pred);
    end

end

