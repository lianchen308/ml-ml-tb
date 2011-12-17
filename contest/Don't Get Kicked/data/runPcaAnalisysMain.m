clear; clc;

%% loading values
tic;

if (~exist('parsedData.mat', 'file'))
    runParseDataMain;
end
load parsedData.mat;
pcadata = data;

%% running pca
var_retained = 0.99;
fprintf('Runing PCA to achieve at least %3.2f%% of variance\n\n', var_retained*100);
[~, zu, ~, zk, z_cum_sigma] = pca([pcadata.x_train1; pcadata.x_test1; pcadata.x_train2; pcadata.x_test2; pcadata.x_submit], ...
    var_retained);
pcadata.x_train1 = projectdata(pcadata.x_train1, zu, zk);
pcadata.x_test1 = projectdata(pcadata.x_test1, zu, zk);
pcadata.x_train2 = projectdata(pcadata.x_train2, zu, zk);
pcadata.x_test2 = projectdata(pcadata.x_test2, zu, zk);
pcadata.x_submit = projectdata(pcadata.x_submit, zu, zk);

%% displaying results
z_cum_sigma(:, 2) = 100*z_cum_sigma(:, 2);
fprintf('Features reduced to %d features retaining %3.2f%% of variance.\n', zk, z_cum_sigma(zk, 2));
fprintf('Variance by dimension:\n\n');
disp(z_cum_sigma);


%% Finishing
fprintf('Saving data...\n');
pcadata = struct(pcadata);
save pcaParsedData.mat pcadata;

fprintf('Data parsed and saved\n\n');
toc;


