function [y_nnetboost, y_svm, y_rf, y_knn] = loadModelData( X )

    
    load ../nnetboost/binaryAdaboostModelData.mat;
    y_nnetboost = mabclassify(learners, weights, X, 1);

    load ../svm/binarySvmModelData.mat;
    y_svm = svmpredictw(svm_model, X);

    load ../rf/binaryRfModelData.mat;
    y_rf = rfpredict(rf_model, X);
    
    load ../knn/binaryKnnModelData.mat;
    y_knn = knnpredict(knn_model, X);
end

