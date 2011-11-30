fprintf('Loading submission data...\n');
% loading data
load binaryNnetModelData; % nn_model
load ('../data/binarySubmitData.mat'); % X_submit

% NNet model
fprintf('Running nnet model on submission data...\n');
[~, y_nn_prob, ~, ~] = nnetPredict(nn_model, X_submit);
fprintf('Saving nnet submission data...\n');
nn_data_submit = [(1:length(y_nn_prob))'  y_nn_prob'];
dlmwrite('csvSubmitNnetData.csv',nn_data_submit,'delimiter',',','precision',6);
fprintf('Nn model ran...\n');
