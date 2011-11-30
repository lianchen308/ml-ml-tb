clear; clc;

%train_data = csvread('../../raw data/cs-training-parsed.csv');
train_data = shuffle(csvread('../../raw data/cs-training-parsed.csv'));
% features = size(train_data, 2);
% unknown_f_index = [];
% for i=1:features
%     unknown_f_index =[unknown_f_index; find(train_data(:, i) == -1)];
% end
% unknown_f_index = unique(unknown_f_index);
% train_data(unknown_f_index, :) = [];

submit_data = csvread('../../raw data/cs-test-parsed.csv');
% features = size(submit_data, 2);
% unknown_f_index = [];
% for i=1:features
%     unknown_f_index =[unknown_f_index; find(submit_data(:, i) == -1)];
% end
% unknown_f_index = unique(unknown_f_index);
% submit_data(unknown_f_index, :) = [];


y_t = train_data(:, 1)';
y_t(y_t == 0) = -1;

unsec_ratio_sat = 10;
unsec_ratio_logbase = 1000;
debt_ratio_sat = 10;
income_sat = 100000;

X_t = train_data(:, 2:11)';
X_t(1, :) = logsat(X_t(1, :), unsec_ratio_sat, unsec_ratio_logbase);
X_t(4, :) = logsat(X_t(4, :), debt_ratio_sat);
X_t(5, :) = logsat(X_t(5, :), income_sat);

X_s = submit_data';
X_s(1, :) = logsat(X_s(1, :), unsec_ratio_sat, unsec_ratio_logbase);
X_s(4, :) = logsat(X_s(4, :), debt_ratio_sat);
X_s(5, :) = logsat(X_s(5, :), income_sat);

%for i=[10 9 8 7 6 3 2]
%    X_s(i, :) = [];
%    X_t(i, :) = [];
%end

X_all = [X_t X_s];
x_mu = mean(X_all, 2);
x_sigma = std(X_all, 0, 2);
X_t_norm = fnorm(X_t, x_mu, x_sigma, 2);
X_submit = fnorm(X_s, x_mu, x_sigma, 2);
n = length(y_t);
n_cut = round(0.6*n);
X_train = X_t_norm(:, 1:n_cut);
y_train = y_t(:, 1:n_cut);

n_cut2 = n_cut + round(0.2*n);
X_val = X_t_norm(:, n_cut+1:n_cut2);
y_val = y_t(:, n_cut+1:n_cut2);

X_test = X_t_norm(:, n_cut2+1:end);
y_test = y_t(:, n_cut2+1:end);

save binaryData.mat X_train y_train X_val y_val X_test y_test;
save binarySubmitData.mat X_submit;
fprintf('Data parsed and saved\n');
%mean_std_train = [mean(X_train, 2) std(X_train, 0, 2)]
%mean_std_submit = [mean(X_submit, 2) std(X_submit, 0, 2)]