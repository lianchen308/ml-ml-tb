function [y_nnetboost, y_svm, y_rf, y_knn, y_svmru, y_nnet, y_gaussian] = loadModelData( X )

    load ../nnetboost/binaryAdaboostModelData.mat;
    y_nnetboost = mabclassify(learners, weights, X, 1);
    
    load ../nnet/binaryNnetModelData.mat;
    [~, y_nnet] = nnetPredict(nn_model, X');
    y_nnet = ((y_nnet - 0.5)*2)';
    
    load ../gaussian/binaryGaussianModelData.mat;
    y_gaussian = gaussianpredict(gaussian_model, X);

    load ../svm/binarySvmModelData.mat;
    y_svm = svmpredictw(svm_model, X);
    
    load ../svm/binarySvmRUModelData.mat
    y_svmru = svmpredictw(svm_ru_model, X);

    load ../rf/binaryRfModelData.mat;
    y_rf = rfpredict(rf_model, X);
    
    load ../knn/binaryKnnModelData.mat;
    y_knn = knnpredict(knn_model, X);
    
end

