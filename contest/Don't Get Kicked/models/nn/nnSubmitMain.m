clc; clear;
load('../../data/parsedData.mat');
load nnModelData.mat;

%% Saving submit
fprintf('Running nn model on submission data...\n');
[~, y_submit_prob] = nnetPredict(nn_model.model, data.x_submit');
savesubmitdata('csvNnSubmitData.csv', {'RefId,IsBadBuy'}, ...
    data.ref_id_submit, y_submit_prob);
