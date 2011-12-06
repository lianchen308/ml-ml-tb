function [nn_model] = mabNnetTrain(X, y, weigths, ~)

    nn_config = newff(minmax(X), minmax(y), 30, {'tansig', 'tansig', 'tansig'});
    nn_config.trainParam.max_fail = 60;
    nn_config.trainParam.min_grad = 1e-30;
    
    weigths_ratio = ones(size(y));
    ratio = length(find(y==1))/length(y);
    weigths_ratio(y==-1) = 1/(1-ratio);
    weigths_ratio(y==1) = (1/ratio);

    [nn_model] = {train(nn_config, X, y, [], [], weigths.*weigths_ratio)};
    
end
