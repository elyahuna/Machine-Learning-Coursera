function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%%%%%%%%%%%%
%% Step 1: Calculating the cost
%%%%%%%%%%%%%%

% Calculating the regularization term, without element theta(1) - the bias element.
% Using scalar multiplication we can get the sum of the sqaure values of the vector.
thetaReg = [ 0 ; theta(2:end)];
regTerm = lambda * thetaReg'*thetaReg;

% Calculating the Err vector between h(x) and y.
h_X_err = X*theta - y;

% Calculating J(theta) using both of the term calculated earlier.
J = 1/(2*m) * (h_X_err'*h_X_err + regTerm);


%%%%%%%%%%%%%
%% Step 2: Calculating the Gradient
%%%%%%%%%%%%%%

% Using vector multiplication we can the grad term.
% X'*h_err(x) will give us the left term of the grad, 
% where for each value of theta the err of each example is being multiplied by the corresponsing X_i(j)
grad = 1/m * (X'*h_X_err + lambda*thetaReg);


% =========================================================================

grad = grad(:);

end
