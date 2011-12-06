%   mabboost Implements boosting process based on "Modest AdaBoost"
%   algorithm
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
%    [learners, weights, final_hyp] = maboost(learn_params, X, y, 
%       old_weights, old_learners)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           learn_params - the learning obj config. 
%                       Must have functions:
%						[models] = learn_params.train(X, y, weigths)
%							where models is a cell array of models
%						[y_predicted] = learn_params.predict(model, X);
%                       Additional properties:
%                        X_val: X used for validation stop. Defaults to 
%                           20% of training set randomly picked
%                        y_val: y to be used with X_val.
%                        show_progress: Print progress to output
%                        max_fail: max iteration without validation
%                           performance increase.
%                        max_iter: max number of absolute iterations
%                       
%           X   - training data. Should be DxN matrix, where D is the
%                       dimensionality of data, and N is the number of
%                       training samples.
%           y   - training labels. Should be 1xN matrix, where N is
%                       the number of training samples.
%           max_iter  - number of iterations
%           old_weights      - weights of already built commitee (used for training 
%                       of already built commitee)
%           old_learners    - learnenrs of already built commitee (used for training 
%                       of already built commitee)
%    Return:
%           learners  - cell array of constructed learners 
%           weights   - weights of learners
%           final_hyp - output for training data
function [learners, weights, final_hyp] = maboost(learn_params, X, y, old_weights, old_learners)

	if (hasfield(learn_params, 'X_val') && hasfield(learn_params, 'y_val'))
		X_val = learn_params.X_val;
		y_val = learn_params.y_val;
	else
		%n = length(y);
		%val_size = round(n*0.3);
		%rand_index = randperm(n);
		%X_val = X(:, rand_index(1:val_size));
		%X = X(:, rand_index(val_size+1:end));
		%y_val = y(:, rand_index(1:val_size));
		%y = y(:, rand_index(val_size+1:end));
        X_val = X;
        y_val = y;
	end
	
    best.auc = -1;
	best.fails = 0;
    
    if (nargin == 3)
        learners = {};
        weights = [];
		n = length(y);
        distr = ones(1, n) / n;
        final_hyp = zeros(1, n);
    elseif (nargin > 4)
        learners = old_learners;
        weights = old_weights;
        final_hyp = mabclassify(learners, weights, X);
        distr = exp(- (y .* final_hyp));  
        distr = distr / sum(distr);
        
        % store best values
        y_val_result = mabclassify(learners, weights, X_val);
        [~, y_val_prob, val_acc] = predlabel(y_val, y_val_result);
        [val_auc] = aucscore(y_val, y_val_prob);
        best.auc = val_auc;
        best.acc = val_acc;
        best.learners = learners;
        best.weights = weights;
        best.final_hyp = final_hyp;
        
    else
        error('Incorrect param number');
    end
    
    if (~hasfield(learn_params, 'show_progress'))
        learn_params.show_progress = 1;
    end
	
    if (~hasfield(learn_params, 'max_fail'))
        learn_params.max_fail = 4;
    end
    
    if (~hasfield(learn_params, 'max_iter'))
        learn_params.max_iter = 1000;
    end
	
    for it = 1 : learn_params.max_iter

        %chose best learner
		models = feval(learn_params.train, X, y, distr, it);
    
        % calc error
        rev_distr = ((1 ./ distr)) / sum ((1 ./ distr));

        for i = 1:length(models)
            curr_model = models{i};

			step_out = feval(learn_params.predict, curr_model, X);
      
			s1 = sum( (y ==  1) .* (step_out) .* distr);
			s2 = sum( (y == -1) .* (step_out) .* distr);
			  
			s1_rev = sum( (y == 1) .* (step_out) .* rev_distr);    
			s2_rev = sum( (y == -1) .* (step_out) .* rev_distr);  
			
			alpha = s1 * (1 - s1_rev) - s2 * (1 - s2_rev);    
		   
			if(sign(alpha) ~= sign(s1 - s2) || (s1 + s2) == 0)
                best.fails = best.fails + 1;
                if (learn_params.show_progress)
                    fprintf('Iter %d (fail): invalid step out.\n', it);
                end
                continue;
			end
			
            final_hyp = final_hyp + step_out .* alpha;
			
            weights(end+1) = alpha; %#ok<AGROW>
            
            learner.model = curr_model;
            learner.predict = learn_params.predict;
            learners{end+1} = learner; %#ok<AGROW>
			
			% performance calc
            y_val_result = mabclassify(learners, weights, X_val);
			[~, y_val_prob, val_acc] = predlabel(y_val, y_val_result);
			[val_auc] = aucscore(y_val, y_val_prob);
			
            
			
			% stop condition
            if ((val_auc > best.auc) || (val_auc == best.auc && val_acc > best.acc))
                best.auc = val_auc;
                best.acc = val_acc;
                best.learners = learners;
                best.weights = weights;
                best.final_hyp = final_hyp;
                best.fails = 0;
            else
                best.fails = best.fails + 1;
            end		
			
            if (learn_params.show_progress)
                if (best.fails == 0)
                    status = 'improved';
                else
                    status = sprintf('fail %d', best.fails);
                end
                [~, y_prob, acc] = predlabel(y, final_hyp);
                fprintf('Iter %d (%s): train acc=%1.4f, train auc=%1.4f, val acc=%1.4f, val auc=%1.4f\n', ...
                    it, status, acc, aucscore(y, y_prob), val_acc, val_auc);
            end
        end
		
		if (best.fails >= learn_params.max_fail)
            if (learn_params.show_progress)
                fprintf('Validation stop.\n');
            end
            break;
		end

		distr = exp(- 1 * (y .* final_hyp));
		Z = sum(distr);
		distr = distr / Z; 

    end
	
	learners  = best.learners;
	weights   = best.weights;
	final_hyp =	best.final_hyp;
end
