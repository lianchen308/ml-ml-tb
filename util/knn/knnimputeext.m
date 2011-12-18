% imputed = knnimputeext(data, K, knnargs)
% knnargs is in the same format of knnimpute
function imputed = knnimputeext(data, K, isDiscrete, weights)
    if (~exist('K', 'var') || isempty(K))
        K = 1;
    end
    
    if (~exist('isDiscrete', 'var') || isempty(isDiscrete))
        isDiscrete = false(1, size(data, 2));
    end
    
    calcWeights = false;
    if (~exist('weights', 'var') || isempty(weights))
        calcWeights = true;
    end
    
    
    dataStd = nanstd(data);
    nanVals = isnan(data);
    noNans = sum(nanVals,2) == 0;
    nans = sum(nanVals,2) > 0;
    dataNoNans = data(noNans, :);
    dataNans = data(nans, :);
    
    % set nancols to 0
    nanVals = isnan(dataNans);
    dataNans(nanVals) = 0;
    
    [~, idx] = pdist2(dataNoNans, dataNans, 'euclidean', 'Smallest', min(100*K, size(dataNoNans, 1)) );   
        
    for r=1:size(dataNans, 1)
        knnIdx = idx(:, r);
        % continuous distance
        contNoNan = dataNoNans(knnIdx, ~isDiscrete);
        contNoNan(:, nanVals(r, ~isDiscrete)) = 0;
        contDist = bsxfun(@minus, contNoNan, dataNans(r, ~isDiscrete));
        contDist = bsxfun(@rdivide, contDist, dataStd(~isDiscrete));
        
        contDist = sum(abs(contDist), 2);
        dist = contDist; 
        
        % discrete distance
        discrNoNan = dataNoNans(knnIdx, isDiscrete);
        discrNoNan(:, nanVals(r, isDiscrete)) = 0;
        discrDist = bsxfun(@ne, discrNoNan, dataNans(r, isDiscrete));
        discrDist = sum(discrDist, 2);
        dist = dist + discrDist; 
        
        % trim to first k
        [dist, iDistSorted] = sort(dist);
        knnIdx = knnIdx(iDistSorted);
        dist = dist(1:K);
        knnIdx = knnIdx(1:K);
        if (calcWeights)
           weights = 1./dist;
        end
        nanIdx = find(nanVals(r, :));
        for c=nanIdx
            dataNans(r, c) = wnanmean(dataNoNans(knnIdx, c), weights, isDiscrete(c));
        end
    end

    imputed = data;
    imputed(nans, :) = dataNans;

end

function m = wnanmean(x, weights, isDiscrete)
    %WNANMEAN Weighted Mean value, ignoring NaNs, infs are special

    % Find NaNs and set them to zero
    x = x(:); weights = weights(:);
    nans = isnan(x);
    weights(nans) = [];
    x(nans) = [];
    infs = isinf(weights);
    if any(infs)
        if (isDiscrete)
            m = mode(x(infs));
            return;
        end
        m = nanmean(x(infs));
        return 
    end
    if (isDiscrete)
        [opt, ~, optIdx] = unique(x,'rows');
        opt_weight = zeros(length(opt), 1);
        for i=1:length(opt_weight)
            opt_weight(i) = sum(weights(optIdx == i));
        end
        [~, mIdx] = max(opt_weight);
        m = opt(mIdx);
        return;
    end
    % normalize the weights
    weights = weights./sum(weights);
    m = weights'*x;
end