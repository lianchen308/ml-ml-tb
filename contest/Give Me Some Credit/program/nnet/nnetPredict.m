function [y_pred, y_prob, acc, auc] = nnetPredict(model, X, y)
        
   % Probability
    y_out = sim(model, X);
    
    if (~exist('y', 'var') || isempty(y))
        y = -1*ones(size(y_out));
    end
    
    [y_pred, y_prob, acc] = predlabel(y, y_out);

    % Auc
    auc = aucscore(y, y_prob);

end
