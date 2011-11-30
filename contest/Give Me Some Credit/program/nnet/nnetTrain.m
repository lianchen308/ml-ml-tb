function [nn_model] = nnetTrain(X, y, tp, af)
    
    nn_config = newff(minmax(X), minmax(y), tp, af);
    nn_config.trainParam.max_fail = 30;
    nn_config.trainParam.min_grad = 1e-15;
    [nn_model] = train(nn_config, X, y);

end