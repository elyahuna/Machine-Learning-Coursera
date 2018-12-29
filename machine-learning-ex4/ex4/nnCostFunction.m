function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Part1:
%%%%%%%%%%%%%%%%%%%%%%%%

% converting y into a hot one vector, i.e. putting one in the corresponding label index
yHotOne = zeros(m,num_labels);
for i = 1:m
	if y(i) == 0
		yHotOne(i,10) = 1;
	else
		yHotOne(i,y(i)) = 1;
	end
end

% performing simple fwd prop
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% calculating J(Theta) with vectoriztion
J = -1/m * sum ( sum( (yHotOne .* log(h2) ) + ( (1-yHotOne) .* log(1-h2) ) ) );

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Part2:
%%%%%%%%%%%%%%%%%%%%%%%%

% getting Theta1 and Theta2 matrices that are not including the first columns,
% in order to calculate the regularization term of the cost function
th1reg = Theta1(:,2:end); 
th2reg = Theta2(:,2:end);

% calculating the regularization element
reg4J = lambda/(2*m) * ( th1reg(:)'*th1reg(:) + th2reg(:)'*th2reg(:) );

% Adding the regularization element to J, to get regularized cost
J = J + reg4J;

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Part3:
%%%%%%%%%%%%%%%%%%%%%%%%

% This part will be completed using vectorization and not using for loop.
% Calculation each matrix so it will be used later for the gradient calculation

A1 = [ones(m,1) X];

Z2 = A1 * Theta1';

A2 = [ones(m,1) sigmoid(Z2)];

Z3 = A2 * Theta2';

A3 = sigmoid(Z3);

delta3 = A3 - yHotOne;

delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(Z2);

% Calculating the regularization elements for Thetas
Theta14Reg = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta24Reg = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

% Calculating the gradients with the regularization elements.
Theta1_grad = 1/m * delta2' * A1 + (lambda/m)*Theta14Reg;
Theta2_grad = 1/m * delta3' * A2 + (lambda/m)*Theta24Reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
