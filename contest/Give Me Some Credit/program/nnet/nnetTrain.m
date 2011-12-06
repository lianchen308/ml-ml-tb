function [nn_model] = nnetTrain(X, y, tp, af)
    
    nn_config = newff(minmax(X), minmax(y), tp, af);
    nn_config.trainParam.max_fail = 60;
    nn_config.trainParam.min_grad = 1e-30;
    
    w = ones(size(y));
    ratio = length(find(y==1))/length(y);
    w(y==-1) = 1/(1-ratio);
    w(y==1) = (1/ratio);
    
    [nn_model] = train(nn_config, X, y, [], [], w);

end