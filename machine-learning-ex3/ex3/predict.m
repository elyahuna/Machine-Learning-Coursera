function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Adding ones collumn to X
X_ones = [ones(m,1),X];

%Computing the input of the hidden layer by using the first theta matrix
A1 = sigmoid(X_ones*Theta1');

%Calculating the num of rows in order to add 1's collumn to the input of the hidden layer
m_a1 = size(A1,1);

%Adding 1's collumn to the inpout of the hiddfen layer.
A1_ones = [ones(m_a1,1),A1];

%Calculating the output of the hidden layer, i.e. the ouput of the system.
A2 = sigmoid(A1_ones*Theta2');

%Retreving the index from each row, of the symbol with the highest output value.
[~,p] = max(A2, [], 2);


% =========================================================================


end
