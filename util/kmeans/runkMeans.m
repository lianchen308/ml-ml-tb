function [centroids, idx] = runkMeans(X, K, ...
                                      max_iters, is_discrete, plot_progress)
%RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
%is a single example
%   [centroids, idx] = runkMeans(X, K, ...
%       max_iters, is_discrete, plot_progress) 
%   runs the K-Means algorithm on data matrix X, where each 
%   row of X is a single example. It uses initial_centroids used as the
%   initial centroids. max_iters specifies the total number of interactions 
%   of K-Means to execute. plot_progress is a true/false flag that 
%   indicates if the function should also plot its progress as the 
%   learning happens. This is set to false by default. runkMeans returns 
%   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

    % Set default value for plot progress
    if ~exist('plot_progress', 'var') || isempty(plot_progress)
        plot_progress = false;
    end
    
    if (~exist('is_discrete', 'var') || isempty(is_discrete))
        is_discrete = [];
    end
    
    if isscalar(K)
        initial_centroids = kMeansInitCentroids(X, K);
    else
        initial_centroids = K;
    end
    
    
    % Plot the data if we are plotting progress
    if plot_progress
        figure;
        hold on;
    end

    % Initialize values
    [m, ~] = size(X);
    K = size(initial_centroids, 1);
    centroids = initial_centroids;
    previous_centroids = centroids;
    idx = zeros(m, 1);

    % Run K-Means
    for i=1:max_iters

        % Output progress
        fprintf('K-Means iteration %d/%d...\n', i, max_iters);

        % For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids, is_discrete);

        % Optionally, plot progress here
        if plot_progress
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
            fprintf('Press enter to continue.\n');
            pause;
        end

        % Given the memberships, compute new centroids
        previous_centroids = centroids;
        centroids = computeCentroids(X, idx, K);
        if (min(min(previous_centroids == centroids)) == 1)
            break;
        end
    end

    % Hold off if we are plotting progress
    if plot_progress
        hold off;
    end

end

