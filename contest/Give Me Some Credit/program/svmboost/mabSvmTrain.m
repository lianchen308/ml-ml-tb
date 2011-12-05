function [svm_model] = mabSvmTrain(X, y, weigths, ~)
    
    C = 483293.0238571752; 
    gamma = 0.0000615848;
    %C = 7742636.8268112773;
    %gamma = 0.0000215443;
    n_train = 10000;
    rand_idx = randperm(length(y), n_train);
    X = X(:, rand_idx);
    y = y(:, rand_idx);
    weigths = weigths(:, rand_idx)/min(weigths);

    svm_model = {svmtrainw(X', y', C, gamma, weigths')};
    
end
