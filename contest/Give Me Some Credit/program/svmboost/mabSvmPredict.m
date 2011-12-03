function [y_pred] = mabSvmPredict(model, X)

    [~, ~, y_pred, ~] = svmpredictw(model, X');
    y_pred = y_pred';
    
end
