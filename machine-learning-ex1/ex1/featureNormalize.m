function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% mean operator will now provide us a vector, 
% where each element is the mean of its corresponding feature
mu = mean(X); 

% in order to subtract the mean of each feature from each feature, 
% we will create a matrix where each row is the mean vector.
% i.e. each collumn is the mean of this feature.
X_mu = X - repmat(mu, [size(X,1) 1]);

% repeating the same method in mean, 
% but now we will divide the std from each element in order 
% to normalize it
sigma = std(X_mu);
X_norm = X_mu./repmat(sigma, [size(X,1) 1]);
% ============================================================

end
