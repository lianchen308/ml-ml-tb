function [nn_model] = nnetTrain(X, y, weigths_or_iteration, ~)
    X = X';
    y = y';
    nn_config = newff(minmax(X), minmax(y), [30 10], {'tansig', 'tansig', 'tansig'});
    nn_config.trainParam.max_fail = 5;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.6; 
    nn_config.divideParam.valRatio = 0.30;
    nn_config.divideParam.testRatio = 0.10;
    
    %nn_weigths = ones(size(y));
    nn_weigths = deftrainweight(y); % ones(size(y)); 
    nn_weigths(y == 1) = nn_weigths(y == 1)*0.9;
    
    if (~isscalar(weigths_or_iteration))
        nn_weigths = nn_weigths.*weigths_or_iteration';
    end
    [nn_model] = {train(nn_config, X, y, [], [], nn_weigths)};
    
end
