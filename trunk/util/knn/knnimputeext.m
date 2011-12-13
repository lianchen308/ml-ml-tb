% imputed = knnimputeext(data, K, knnargs)
% knnargs is in the same format of knnimpute
function imputed = knnimputeext(data, K)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
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
        dist_vals = dataNoNans(idx{r, :}, :);
        dist_vals(:, nan_idx) = 0;
        dist_vals = sum(bsxfun(@minus, dist_vals, dataNans(r, :)).^2, 2);
        dist_vals = 1./dist_vals;
        for c=nan_idx
            dataNans(r, c) = wnanmean(dataNoNans(idx{r, :}, c), dist_vals);
        end
    end

    imputed = data;
    imputed(nans, :) = dataNans;

end

function m = wnanmean(x,weights)
    %WNANMEAN Weighted Mean value, ignoring NaNs, infs are special

    % Find NaNs and set them to zero
    x = x(:); weights = weights(:);
    nans = isnan(x);
    infs = isinf(weights);
    if all(nans)
        m = NaN;
        return
    end
    if any(infs)
        m = nanmean(x(infs));
        return 
    end
    % Sum up non-NaNs, and divide by the number of non-NaNs.
    x(nans) = 0;
    weights(nans) = 0;
    % normalize the weights
    weights = weights./sum(weights);
    m = weights'*x;
end