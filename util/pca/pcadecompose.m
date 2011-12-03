function [U, S] = pcadecompose(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pcadecompose(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

	[m, ~] = size(X);
	[U, S] = svd((1/m)*(X'*X));

end
