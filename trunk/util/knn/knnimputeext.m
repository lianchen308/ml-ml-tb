% imputed = knnimputeext(data, K, knnargs)
% knnargs is in the same format of knnimpute
function imputed = knnimputeext(data, K, distance_fn)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
    end
    if (~exist('distance_fn', 'var') || isempty(distance_fn))
        distance_fn = 'euclidean';
    end
    
    nanVals = isnan(data);
    noNans = sum(nanVals,2) == 0;
    nans = sum(nanVals,2) > 0;
    dataNoNans = data(noNans, :);
    dataNans = data(nans, :);
    nanVals = isnan(dataNans);
    dataNans(nanVals) = 0;
    
    [idx, dist] = knnsearch(dataNoNans, dataNans, 'dist', distance_fn, 'k', K);
    dist = 1 ./ dist;
    
    for r=1:size(dataNans, 1)
        nan_idx = find(nanVals(r, :));
        for c=nan_idx
            dataNans(r, c) = wnanmean(dataNoNans(idx(r, :), c), dist(r, :));
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