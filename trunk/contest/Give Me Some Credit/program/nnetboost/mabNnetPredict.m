function [y_pred] = mabNnetPredict(model, X)

    y_pred = sim(model, X);
    
end
