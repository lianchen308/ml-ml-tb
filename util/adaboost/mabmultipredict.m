function [y_pred] = mabmultipredict(model, X)

    y_pred = mabclassify(model.learners, model.weights, X);
    max_val = max(abs(min(y_pred)), abs(max(y_pred)));
    y_pred = y_pred/max_val;
    
end
