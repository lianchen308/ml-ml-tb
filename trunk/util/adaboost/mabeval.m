% [y_test_auc, y_test_acc] = mabeval(learners, weights, X_train, y_train, X_test, y_test)
function [y_test_auc, y_test_acc] = mabeval(learners, weights, X_train, y_train, X_test, y_test) 
    
    [y_train_acc, y_train_auc] = mabscore(learners, weights, X_train, y_train);
    fprintf('Train result: acc = %1.4f, auc = %1.4f\n', y_train_acc, y_train_auc);
    [y_test_acc, y_test_auc] = mabscore(learners, weights, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);
    
    y_test_pred = mabclassify(learners, weights, X_test, 1);
    [y_test_negacc, y_test_posacc] = classaccuracy(y_test, y_test_pred);
    fprintf('Test acc: neg acc = %1.4f, pos acc = %1.4f\n', y_test_negacc, y_test_posacc);
    
end