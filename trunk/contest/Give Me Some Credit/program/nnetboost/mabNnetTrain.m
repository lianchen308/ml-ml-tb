function [nn_model] = mabNnetTrain(X_train, y_train, weigths, tp, af)

    nn_config = newff(minmax(X_train), minmax(y_train), tp, af);
    nn_config.trainParam.max_fail = 30;
    nn_config.trainParam.min_grad = 1e-15;

    [nn_model] = train(nn_config, X_train, y_train, [], [], weigths);
    
end
