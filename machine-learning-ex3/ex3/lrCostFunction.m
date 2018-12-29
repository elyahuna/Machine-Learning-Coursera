function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% vectorized computation of the h_theta(x) using the sigmoid function. We now have m*1 matrix (m = num of examples):
h_theta_x = sigmoid(X*theta);

%I will seperate the calculation into to term: [sum(y^i*log(h_theta_x^i))] and [sum((1-y^i)*log(1-h_theta_x^i))]
% I will perform addition of the term and multiply them by (-1/m):
J = (-1/m) * ( (y' * log(h_theta_x)) + ((1-y)' * log(1-h_theta_x)) );

%the gradient is much easier. the summation among X_j will be perfomed by matrix multiplication.
grad = 1/m * (X' * (h_theta_x - y));

%Regularization part:
%%%%%

%theta_zero is theta with the first element as zero, to easily eliminate calculation for theta(0):
theta_zero = [0;theta(2:length(theta))];
J = J + (0.5*lambda/m) * (theta_zero')*(theta_zero);
grad = grad + (lambda/m) * theta_zero;

% =============================================================

grad = grad(:);

end
