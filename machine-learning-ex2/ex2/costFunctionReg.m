function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% the base for the calcultion of J and grad is still the cost function from ex2
[J, grad] = costFunction(theta, X, y);

% in order to elimination regularization on 1st theta,
% we will create a new vector of theta that will use us to add the regularization term.
theta_1st_item_zero = theta;
theta_1st_item_zero(1) = 0;

% the the sum of theta.^2 is actually theta'*theta (but without 1st element).
J = J + lambda/(2*m) * (theta_1st_item_zero' * theta_1st_item_zero);
% each item of the gradient will be added with it's corresponding regularization term (except to the 1st item)
grad = grad + ((lambda/m) * theta_1st_item_zero');

% =============================================================

end
