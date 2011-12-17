% imputed = knnimputeext(data, K, knnargs)
% knnargs is in the same format of knnimpute
function imputed = knnimputeext(data, K, is_discrete, weights)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
    end
    
    if (~exist('is_discrete', 'var') || isempty(is_discrete))
        is_discrete = false(1, size(data, 2));
    end
    
    calc_weights = false;
    if (~exist('weights', 'var') || isempty(weights))
        calc_weights = true;
    end
    
    
    
    nanVals = isnan(data);
    noNans = sum(nanVals,2) == 0;
    nans = sum(nanVals,2) > 0;
    dataNoNans = data(noNans, :);
    dataNans = data(nans, :);
    
    % set nancols to 0
    nanVals = isnan(dataNans);
    dataNans(nanVals) = 0;
    
    % find nancols
    [~, nan_cols] = find(nanVals);
    nan_cols = unique(nan_cols);
    
    for col=nan_cols
        [idx, ~] = knnsearch(dataNoNans, dataNans, ...
            'k', K, 'dist', 'euclidean', 'IncludeTies', true);
    end
    
    for r=1:size(dataNans, 1)
        nan_idx = find(nanVals(r, :));
        if (calc_weights)
            weights = dataNoNans(idx{r, :}, :);
            weights(:, nan_idx) = 0;
            weights = sum(bsxfun(@minus, weights, dataNans(r, :)).^2, 2);
            weights = 1./weights;
        end
        for c=nan_idx
            dataNans(r, c) = wnanmean(dataNoNans(idx{r, :}, c), weights, is_discrete(c));
        end
    end

    imputed = data;
    imputed(nans, :) = dataNans;

end

function m = wnanmean(x, weights, is_discrete)
    %WNANMEAN Weighted Mean value, ignoring NaNs, infs are special

    % Find NaNs and set them to zero
    x = x(:); weights = weights(:);
    nans = isnan(x);
    weights(nans) = [];
    x(nans) = [];
    infs = isinf(weights);
    if any(infs)
        if (is_discrete)
            m = mode(x(infs));
            return;
        end
        m = nanmean(x(infs));
        return 
    end
    if (is_discrete)
        m = mode(x);
        return;
    end
    % normalize the weights
    weights = weights./sum(weights);
    m = weights'*x;
end