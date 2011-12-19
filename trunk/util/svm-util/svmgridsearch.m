% function [svm_model, svm_auc, svm_c, svm_gamma] = ...
%	[svm_model, svm_score, svm_c, svm_gamma] = svmgridsearch(X_train, y_train, X_test, y_test, ...
%       c_values, gamma_values, svm_opt, n_find_params, n_actual_train)
function [svm_model, svm_score, svm_c, svm_gamma] = svmgridsearch(x_train, y_train, x_test, y_test, ...
	c_values, gamma_values, svm_opt, n_find_params, n_actual_train)
	
    %%
    fprintf('Finding SVM params...\n');
    n_train = size(y_train, 1);
    
    if (~exist('n_find_params', 'var') || isempty(n_find_params))
        n_find_params = 2000;
    end
    
    if (n_find_params >= n_train)
        n_find_params = n_train;
    end
	if (~exist('n_actual_train', 'var') || isempty(n_actual_train))
		n_actual_train = 20000;
	end
    if (n_actual_train >= n_train)
        n_actual_train = n_train;
    end
    
    if (~isfield(svm_opt, 'score_fcn') || isempty(svm_opt.score_fcn))
		svm_opt.score_fcn = 'aucscore';
    end

    %%
    [svm_c, svm_gamma] = dogridsearch(x_train(1:n_find_params, :), y_train(1:n_find_params, :), ...
        x_test, y_test, c_values, gamma_values, svm_opt);
	svm_c = svm_c(1); 
	svm_gamma = svm_gamma(1);
	
    fprintf('Params selected: C = %f, gamma = %f\n', ...
                svm_c(1), svm_gamma(1));

    %%
    fprintf('Training SVM params...\n');

    cur_svm_opt = svm_opt;
    cur_svm_opt.c = svm_c(1);
    cur_svm_opt.g = svm_gamma(1);
    [svm_model] = svmtrainw(x_train(1:n_actual_train, :), y_train(1:n_actual_train, :), cur_svm_opt);
    [~, ~, ~, y_train_score] = svmpredictw(svm_model, x_train, y_train, svm_opt.score_fcn);
    [~, ~, ~, y_test_score] = svmpredictw(svm_model, x_test, y_test, svm_opt.score_fcn);
    fprintf('%s: train = %1.4f, test = %1.4f\n', ...
        svm_opt.score_fcn, y_train_score, y_test_score);
    svm_score = y_test_score;

end

function [C, gamma] = dogridsearch(x_train, y_train, x_test, y_test, ...
    c_values, gamma_values, svm_opt)

    %% common vars
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

    %% building param table: [C gamma score acc] for each column
    param_matrix = zeros(c_n*gamma_n, 3);
    z = 1;
    for i=1:c_n
        for j=1:gamma_n
            param_matrix(z,c_col) = c_values(i);
            param_matrix(z,gamma_col) = gamma_values(j);
            z = z + 1;
        end
    end


    %% training models
    n_params = size(param_matrix, 1);
    fid = fopen(sprintf('svmgridsearch.%s.txt', datestr(now, 'yyyy-mm-dd.HH-MM')),'a');
	fprintf(fid,'%%\tC\t\tgamma\t\tacc\t\t%s\n', svm_opt.score_fcn);
    for i=1:n_params
        % getting params
        C = param_matrix(i,c_col);
        gamma = param_matrix(i,gamma_col);
        %training and evaluating
        cur_svm_opt = svm_opt;
        cur_svm_opt.c = C;
        cur_svm_opt.g = gamma;
        [svm_model] = svmtrainw(x_train, y_train, cur_svm_opt);
        [~, y_test_acc, ~, y_test_score] = svmpredictw(svm_model, ...
            x_test, y_test, svm_opt.score_fcn);
        %saving results
        param_matrix(i,score_col) = y_test_score;
        fprintf('P %d of %d: C=%10.10f, g=%10.10f, acc:%1.4f, %s:%1.4f\n', ...
                i, n_params, C, gamma, y_test_acc, svm_opt.score_fcn, y_test_score);
        fprintf(fid, '%s\t%s\t%s\t%s\n', num2str(C), num2str(gamma), ...
            num2str(y_test_acc), num2str(y_test_score));
    end
    fclose(fid);

    %% find max accuracy
    score		= max(param_matrix(:, score_col));
    max_i	= find(param_matrix(:, score_col) == score);
    for i=1:length(max_i)
        C(i) = param_matrix(max_i(i), c_col);
        gamma(i) = param_matrix(max_i(i), gamma_col);
    end

end