% [y_pred, y_prob, acc, auc] = rfpredict(model, X, y)
function [y_pred, y_prob, acc, auc] = rfpredict(model, X, y)

    y_class = [];
    if (strcmp('CompactTreeBagger', class(model)) ...
            || strcmp('TreeBagger', class(model))) 
        [~, y_prob] = predict(model, X);
    elseif (length(model) > 1)
        [y_class y_prob]= eval_Stochastic_Bosque(X, model, 'oobe', 1);
        y_prob = y_prob'- 1;
        y_prob = sum(y_prob, 2)/size(y_prob,2);
        y_prob = [(1-y_prob) y_prob];
    else
        [y_class, y_prob] = classRF_predict(X, model);
        y_prob = bsxfun(@rdivide, y_prob, sum(y_prob, 2));
    end
    
    y_prob = y_prob(:, 2);
    
    y_pred = (y_prob - 0.5)*2;
    
    if (~exist('y', 'var') || isempty(y))
        acc = -1;
        auc = -1;
    else
        if (isempty(y_class))
            y_class = zeros(size(y_pred));
            y_class(y_pred < 0)  = -1;
            y_class(y_pred >= 0) = 1;
        end
        acc = (length(find(y_class == y))/length(y));
        auc = aucscore(y, y_pred);
    end

end