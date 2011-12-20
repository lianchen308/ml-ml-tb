% [X_resampled, y_resampled] = ggsvmru(X_train, y_train, X_val, y_val, c, gamma, ...
%   score_fcn, n_batch, n_min_it)
function [X_resampled, y_resampled] = gsvmru(X_train, y_train, X_val, y_val, c, gamma, ...
    score_fcn, n_batch, n_min_it)
    
    if (~exist('n_batch', 'var') || isempty(n_batch))
        n_batch = 10000;
    end
    if (n_batch > size(y_train, 1))
        n_batch = size(y_train, 1);
    end
    
    if (~exist('n_min_it', 'var') || isempty(n_min_it))
        n_min_it = 5;
    end
    
    X_psv = X_train(y_train == 1, :);
    y_psv = y_train(y_train == 1, :);
    
    i = 1;
    best.acc = 0;
    best.score = -1;
    X_nsv = [];
    y_nsv = [];
    while (true)
        [X_train, y_train, X_nsv, y_nsv, agg_acc, agg_score] = aggresample(X_train, y_train, ...
            X_val, y_val, X_psv, y_psv, X_nsv, y_nsv, c, gamma, score_fcn, n_batch, i);
        if (agg_score > best.score || i <= n_min_it)
            best.acc = agg_acc;
            best.score = agg_score;
            [best.X_agg, best.y_agg] = uniquex([X_psv; X_nsv], [y_nsv; y_nsv]);
        else
            break;
        end
        i = i + 1;
    end
    
    X_resampled = best.X_agg;
    y_resampled = best.y_agg;
    [X_resampled, idx] = shuffle(X_resampled);
    y_resampled = y_resampled(idx, :);
    
end

function [X_train, y_train, X_nsv_agg, y_nsv_agg, agg_acc, agg_score] = aggresample(X_train, y_train, X_val, y_val, ...
    X_psv, y_psv, X_nsv_agg, y_nsv_agg, c, gamma, score_fcn, n_batch, i)
    
    fprintf('GSVM-RU iteration %d...\n', i);
    [X_nsv, y_nsv] = svmresample(X_train, y_train, c, gamma, n_batch);
    [X_train, y_train, ~, ~] = movesample(X_train, y_train, [], [], X_nsv);
	X_nsv_agg = [X_nsv_agg; X_nsv];
    y_nsv_agg = [y_nsv_agg; y_nsv];
    
    fprintf('Training resampling...\n');
    [X_agg, y_agg] = uniquex([X_psv; X_nsv_agg], [y_psv; y_nsv_agg]);
    [X_agg, idx] = shuffle(X_agg);
    y_agg = y_agg(idx, :);
    opt.c = c;
    opt.gama = gamma;
    [svm_model] = svmtrainw(X_agg, y_agg, opt);
    [~, agg_acc, ~, agg_score] = svmpredictw(svm_model, X_val, y_val, score_fcn);
    fprintf('Validation result: acc = %s, %s = %s ...\n', num2str(agg_acc), score_fcn, num2str(agg_score));
    
end


