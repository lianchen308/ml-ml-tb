function [nn_model] = nnetTrain(X, y, tp, af, w)
    
    nn_config = newff(minmax(X), minmax(y), tp, af);
    nn_config.trainParam.max_fail = 30;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.6; 
    nn_config.divideParam.valRatio = 0.25;
    nn_config.divideParam.testRatio = 0.15;
    
    [nn_model] = train(nn_config, X, y, [], [], w);

end