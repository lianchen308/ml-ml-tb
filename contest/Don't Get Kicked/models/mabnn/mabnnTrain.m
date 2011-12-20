function [nn_model] = mabnnTrain(X, y, weigths, ~)
    X = X';
    y = y';
    nn_config = newff(minmax(X), minmax(y), 60, {'tansig', 'tansig', 'tansig'});
    nn_config.trainParam.max_fail = 8;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.5; 
    nn_config.divideParam.valRatio = 0.5;
    nn_config.divideParam.testRatio = 0.0;
    
    nn_weigths = ones(size(y));
    nn_weigths(y == 1) = 5.849788;
    
    if (~isscalar(weigths))
        nn_weigths = nn_weigths.*weigths';
    end
    [nn_model] = {train(nn_config, X, y, [], [], nn_weigths)};
    
end
