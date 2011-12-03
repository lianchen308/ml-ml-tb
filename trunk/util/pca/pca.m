function [Z, U, S, K, cum_sigma] = pca(X, sigma)
%PCA Run principal component analysis on the dataset X
%   [Z, U, S, K, cum_sigma] = pca(X, sigma) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

	[U, S] = pcadecompose(X);
	cumvar = cumsum(cumsum(S), 2);
	cumvar = cumvar(:, end);
    cumvar = cumvar/cumvar(end);
    cum_sigma = [(1:length(cumvar))' cumvar];
    
	K =  find(cumvar >= sigma);
	K = K(1);
	Z = projectdata(X, U, K);
	
end
