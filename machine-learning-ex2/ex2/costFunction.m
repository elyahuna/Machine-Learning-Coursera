function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

% Calculating h_theta(X) using the sigmoid function from the previous exercise
h_theta_x = sigmoid(X*theta);

%The calculation of the J and the cost is gonna be vectorized

%Ycomb will be the vector [-y_1, -(1-y_1),....,-y_m, -(1-y_m)]
Ycomb((1:m)*2 - 1) 	= -y;
Ycomb((1:m)*2) 		= -(1 - y);

%Xcomb will be the vector [log(h_theta_x_1), log(1 - h_theta_x_1),....., log(h_theta_x_m), log(1 - h_theta_x_m)]
Xcomb((1:m)*2 - 1) 	= log(h_theta_x);
Xcomb((1:m)*2)		= log(1 - h_theta_x);

% now we will multiply this vector to receive the sum over cost of each example
% i.e: Ycomb(0)*Xcomb(0)+...+Ycomb(m)*Xcomb(m) = -y_1*log(h_theta_x_1) -(1-y_1)*log(1-h_theta_x_1) + .....
J = (1/m) * Ycomb*Xcomb';

% vectorized operation of the gradient:
grad = (1/m) * ((h_theta_x - y)'*X);

% =============================================================

end
