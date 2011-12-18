% [y_pred, y_prob, acc, auc] = nnetPredict(model, X, y)
function [y_pred, y_prob, acc, score] = nnetPredict(model, X, y, score_fcn)
        
   % Probability
    y_out = sim(model, X);
    
    if (~exist('y', 'var') || isempty(y))
        y = -1*ones(size(y_out));
    end
    
    [y_pred, y_prob, acc] = predlabel(y, y_out);

    % Auc
    if (exist('score_fcn', 'var') && ~isempty(score_fcn))
        score = feval(score_fcn, y, y_prob);
    end

end
