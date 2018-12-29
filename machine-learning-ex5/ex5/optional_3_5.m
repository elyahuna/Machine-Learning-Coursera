function optional_3_5(X, y, Xval, yval, lambda, numOfItersPer_m)

% This function implments part 3.5 in ex5.
% It plots a learning curve,
% where for number of examples the error is calculated by 
% averaging the error of thetea
% that was learning using (i) randomly selected examples.

% Input:
%	X, y - the whole training set
%	Xval, yval - the CV set
%	lambda - the lambda that should be used for the regularization.
%   
	
% number of train examples:	
mTrain = length(X);

%initializing the error matrices (for each number of examples, for each iteration
error_train = zeros(length(y), numOfItersPer_m);
error_val = error_train;

% looping through number of examples:
for i=1:mTrain

	%looping through the size of group we would like to average
	for j=1:numOfItersPer_m
	
		%creating a vector of random indices in the required size and range
		trainIndices = randperm(mTrain,i);
		
		%retrieving the current used training set according
		% to the current train indices:
		curr_X_train = X(trainIndices,:);
		curr_y_train = y(trainIndices,:);
		
		%training the algrithm and calculating theta
		theta = trainLinearReg(curr_X_train, curr_y_train, lambda);
		
		% calculating the error for the current training set and CV set:
		[error_train(i,j), ~] = linearRegCostFunction(curr_X_train, curr_y_train, theta, 0);
		[error_val(i,j), ~] = linearRegCostFunction(Xval, yval, theta, 0);
	end
end

% averagin the error through the iterations
avgErrorTrain = mean(error_train, 2);
avgErrorVal   = mean(error_val, 2);

%plotting the learning curve:
figure(10);

plot(1:mTrain, avgErrorTrain, 1:mTrain, avgErrorVal);

title(sprintf('Polynomial Regression Average Learning Curve (lambda = %f)', lambda));
xlabel('Number of random training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
end
