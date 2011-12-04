function [svm_model] = mabSvmTrain(X, y, weigths, ~)
    
    %C = 483293.0238571752; 
    %gamma = 0.0000615848;
    C = 7742636.8268112773;
    gamma = 0.0000215443;
    
    %weigths = weigths/mean(weigths);
    svm_model = {svmtrainw(X', y', C, gamma, weigths')};
    
end
