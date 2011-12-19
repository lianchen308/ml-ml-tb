% [y_pred, y_prob, acc, score] = knnpredict(knn_model, X, y, score_fcn)
function [y_pred, y_prob, acc, score] = knnpredict(knn_model, X, y, score_fcn)

    n_rows = size(X, 1);
    impute_data = [X (NaN*ones(n_rows, 1))];
    
    y_pred = knnimputeext([impute_data; knn_model.data], knn_model.K);
    y_pred = y_pred(1:n_rows, end); 
    y_prob = (y_pred + 1)/2;
    
    if (~exist('y', 'var') || isempty(y))
        acc = -1;
        score = -1;
    else
        y_class = zeros(size(y_pred));
        y_class(y_pred < 0)  = -1;
        y_class(y_pred >= 0) = 1;
        acc = (length(find(y_class == y))/length(y));
        if (exist('score_fcn', 'var') && ~isempty(score_fcn))
            score = feval(score_fcn, y, y_prob);
        end
    end

end

