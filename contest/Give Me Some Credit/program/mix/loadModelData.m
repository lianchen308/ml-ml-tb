function [y_nnetboost, y_svm, y_rf] = loadModelData( X )

    load ../nnetboost/binaryAdaboostModelData.mat;
    load ../svm/binarySvmModelData.mat;
    load ../rf/binaryRfModelData.mat;

    y_nnetboost = mabclassify(learners, weights, X, 1);

    y_svm = svmpredictw(svm_model, X);

    y_rf = rfpredict(rf_model, X);
end

