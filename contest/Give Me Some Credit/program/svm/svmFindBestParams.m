function [C, gamma] = svmFindBestParams(X_train, y_train, X_val, y_val, c_values, gamma_values)

    % common vars
    c_n 		= length(c_values);
    gamma_n 	= length(gamma_values);

    if (c_n == 1 && gamma_n == 1)
        C = c_values;
        gamma = gamma_values;
        return;
    end

    c_col = 1;
    gamma_col = 2;
    score_col = 3;

    % building param table: [C gamma score acc] for each column
    param_matrix = zeros(c_n*gamma_n, 3);
    z = 1;
    for i=1:c_n
        for j=1:gamma_n
            param_matrix(z,c_col) = c_values(i);
            param_matrix(z,gamma_col) = gamma_values(j);
            z = z + 1;
        end
    end


    % training models
    n_params = size(param_matrix, 1);
    for i=1:n_params
        % getting params
        C = param_matrix(i,c_col);
        gamma = param_matrix(i,gamma_col);
        %training and evaluating
        [svm_model] = svmTrainWrapper(X_train, y_train, C, gamma);
        [~, y_val_acc, ~, y_val_auc] = svmPredictWrapper(X_val, y_val, svm_model);
        %saving results
        param_matrix(i,score_col) = y_val_auc;
        fprintf('P %d of %d: C=%8.8f, g=%5.10f, acc:%1.4f, auc:%1.4f\n', ...
                i, n_params, C, gamma, y_val_acc, y_val_auc);
    end

    % find max accuracy
    auc		= max(param_matrix(:, score_col));
    max_i	= find(param_matrix(:, score_col) == auc);
    for i=1:length(max_i)
        C(i) = param_matrix(max_i(i), c_col);
        gamma(i) = param_matrix(max_i(i), gamma_col);
    end

end