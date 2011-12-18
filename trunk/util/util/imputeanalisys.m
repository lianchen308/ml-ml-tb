% imputeanalisys(x_train, is_discrete, ks)
function imputeanalisys(x_train, is_discrete, options)
    if (~exist('options', 'var') || isempty(options))
        options.k = [];
    end
    if (~isfield(options, 'k') || isempty(options.k))
        options.k = [1 5 10 15 20 50 100];
    end

    %% Normalizing data
    fprintf('Normalizing data...\n');
    [x_train, ~, ~] = fnorm(x_train);

    %% Getting nan statistics
    fprintf('Getting nan statistics...\n');
    x_train_nans = isnan(x_train);
    nan_per_col = sum(x_train_nans);
    missing_ratio = sum(nan_per_col)/length(x_train_nans);
    fprintf('Missing values: %3.2f%%...\n', missing_ratio*100);
    
    %% Removing nan rows
    fprintf('Removing nan rows...\n');
    x_train_nan_rows = any(isnan(x_train), 2);
    x_train_non_nan = x_train;
    x_train_non_nan(x_train_nan_rows, :) = [];

    %% Injecting random nans based on statistics collected
    fprintf('Injecting random nans based on statistics collected...\n\n');
    n_non_nan = size(x_train_non_nan, 1);
    x_train_inject_nan = x_train_non_nan;
    nan_cols = find(nan_per_col > 0);
    for col=nan_cols
        rand_idx = randperm(n_non_nan, nan_per_col(col));
        x_train_inject_nan(rand_idx, col) = NaN;
    end
    injected_nans = isnan(x_train_inject_nan);

    %% Building impute (continuous or discrete) vector
    impute_types =  repmat('c', 1, length(is_discrete));
    dicrete_idx = find(is_discrete);
    impute_types(dicrete_idx) = repmat('d', 1, length(dicrete_idx));

    %% knn neighboors imputation
    for k=options.k
        fprintf('Running knn neighboors (k=%d)...\n', k);

        x_impute = knnimputeext(x_train_inject_nan, k, is_discrete); 
        x_imputeloss = imputationLossMixed(x_train_non_nan, ...
            x_impute, injected_nans, impute_types);
        fprintf('Weighted knn neighboors (k=%d) loss: %f...\n', k, x_imputeloss);

%         x_impute = knnimputeext(x_train_inject_nan, k, is_discrete, ones(k,1)); 
%         x_imputeloss = imputationLossMixed(x_train_non_nan, ...
%             x_impute, injected_nans, impute_types);
%         fprintf('Non-weighted knn neighboors (k=%d) loss: %f...\n', k, x_imputeloss);
        fprintf('\n');
    end

    %% mean value imputation
    fprintf('Running mean value imputation...\n');
    x_impute = meanValueImputation(x_train_inject_nan, impute_types);
    x_imputeloss = imputationLossMixed(x_train_non_nan, ...
            x_impute, injected_nans, impute_types);
    fprintf('Mean value imputation loss: %f...\n\n', x_imputeloss);

    %% EM imputation
    fprintf('Running EM imputation...\n');
    emopt.maxit = 5;
    emopt.disp = 0;
    x_impute = regem(x_train_inject_nan, emopt);
    x_imputeloss = imputationLossMixed(x_train_non_nan, ...
            x_impute, injected_nans, impute_types);
    fprintf('EM imputation loss: %f...\n\n', x_imputeloss);
    %% Finishing
    fprintf('Done!\n\n');
end