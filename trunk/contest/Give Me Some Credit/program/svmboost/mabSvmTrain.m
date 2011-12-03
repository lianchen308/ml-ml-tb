function [svm_model] = mabSvmTrain(X, y, weigths, ~)
    
    C = 483293.0238571752; 
    gamma = 0.0000615848;
    svm_model = {svmtrainw(X', y', C, gamma, weigths')};
    
end
