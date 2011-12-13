% [y_pred, y_prob, acc, auc] = gaussianpredict(gaussian_model, X, y)
function [y_pred, y_prob, acc, auc] = gaussianpredict(gaussian_model, X, y)

    prob = multivariateGaussian(X, gaussian_model.mu, gaussian_model.sigma2);
    y_pred = zeros(size(prob));
    thrs = gaussian_model.thres;
    y_pred(prob >= thrs) = -((prob(prob >= thrs) - thrs)/max(prob));
    y_pred(prob < thrs) = 1 - prob(prob < thrs)/thrs;
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

