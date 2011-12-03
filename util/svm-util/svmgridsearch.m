% function [svm_model, svm_auc, svm_c, svm_gamma] = ...
%	svmgridsearch(X_train, y_train, X_val, y_val, X_test, y_test, ...
%		c_values, gamma_values, cross_validation, n_find_params, n_actual_train)
function [svm_model, svm_auc, svm_c, svm_gamma] = svmgridsearch(X_train, y_train, X_val, y_val, X_test, y_test, ...
	c_values, gamma_values, cv, n_find_params, n_actual_train)
	
    fprintf('Finding SVM params...\n');
    n_train = size(y_train, 1);
	
	if (~exist('cv', 'var') || isempty(cv))
		cv = 10;
	end
    
	if (~exist('n_find_params', 'var') || isempty(n_find_params))
		n_find_params = 2000;
	end
    if (n_find_params >= n_train)
        n_find_params = n_train;
		X_val_find = X_val;
		y_val_find = y_val;
	else
		X_val_find = [X_train(n_find_params+1:end, :); X_val];
		y_val_find = [y_train(n_find_params+1:end, :); y_val];
    end
	if (~exist('n_actual_train', 'var') || isempty(n_actual_train))
		n_actual_train = 20000;
	end
    if (n_actual_train >= n_train)
        n_actual_train = n_train;
    end

    [svm_c, svm_gamma] = dogridsearch(X_train(1:n_find_params, :), y_train(1:n_find_params, :), ...
        X_val_find, y_val_find, c_values, gamma_values);
	svm_c = svm_c(1); 
	svm_gamma = svm_gamma(1);
	
    fprintf('Params selected: C = %8.8f, gamma = %5.10f\n', ...
                svm_c(1), svm_gamma(1));

    fprintf('Training SVM params...\n');

    [svm_model] = svmtrainw(X_train(1:n_actual_train, :), y_train(1:n_actual_train, :), svm_c(1), svm_gamma(1), cv);
    [~, ~, ~, y_train_auc] = svmpredictw(svm_model, X_train, y_train);
    [~, ~, ~, y_val_auc] = svmpredictw(svm_model, X_val, y_val);
    [~, ~, ~, y_test_auc] = svmpredictw(svm_model, X_test, y_test);
    fprintf('AUC: train auc = %1.4f, val auc = %1.4f, test auc = %1.4f\n', y_train_auc, y_val_auc, y_test_auc);
    svm_auc = min([y_train_auc y_val_auc y_test_auc]);

end

function [C, gamma] = dogridsearch(X_train, y_train, X_val, y_val, c_values, gamma_values)

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
    fid = fopen('svmgridsearch.txt','a');
    for i=1:n_params
        % getting params
        C = param_matrix(i,c_col);
        gamma = param_matrix(i,gamma_col);
        %training and evaluating
        [svm_model] = svmtrainw(X_train, y_train, C, gamma);
        [~, y_val_acc, ~, y_val_auc] = svmpredictw(svm_model, X_val, y_val);
        %saving results
        param_matrix(i,score_col) = y_val_auc;
        performance = sprintf('P %d of %d: C=%10.10f, g=%10.10f, acc:%1.4f, auc:%1.4f\n', ...
                i, n_params, C, gamma, y_val_acc, y_val_auc);
        fprintf(performance);
        fprintf(fid, performance);
    end
    fclose(fid);

    % find max accuracy
    auc		= max(param_matrix(:, score_col));
    max_i	= find(param_matrix(:, score_col) == auc);
    for i=1:length(max_i)
        C(i) = param_matrix(max_i(i), c_col);
        gamma(i) = param_matrix(max_i(i), gamma_col);
    end

end