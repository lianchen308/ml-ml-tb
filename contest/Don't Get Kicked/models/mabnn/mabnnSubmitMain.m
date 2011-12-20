clc; clear;
load('../../data/parsedData.mat');
load mabNnModelData.mat;

%% Saving submit
fprintf('Running nn model on submission data...\n');
[y_submit_out] = mabclassify(mab_nn_model.learners, mab_nn_model.weights, data.x_submit);
[y_submit_pred, y_submit_prob] = predlabel(y_submit_out);
savesubmitdata('csvMabNnSubmitData.csv', {'RefId,IsBadBuy'}, ...
    data.ref_id_submit, y_submit_prob);
