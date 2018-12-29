function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Creating two vector for sigma and C to evaluate
sigma_Vec = C_vec = [0.01,0.03,0.1,0.3,1,3,10,30];
paramVecSize = length(C_vec);

% Creating an error matric for all evaluated values of sigma and C
CV_ERR = zeros(paramVecSize, paramVecSize);

% Evaluating each values combiantion of sigma and C
for sigmaIndex = 1:paramVecSize
	for cIndex = 1:paramVecSize
		
		%Training the model classifier with the current c and sigma
		model= svmTrain(X, y, C_vec(cIndex), ...
		@(x1, x2) gaussianKernel(x1, x2, sigma_Vec(sigmaIndex)));
		
		%Using the trained model to predict the CV data set
		pred = svmPredict(model, Xval);
		
		%calculating the mean error on the cv data set
		CV_ERR(sigmaIndex, cIndex) = mean(double(pred ~= yval));
	end
end

% Indexes for the min value of of the error
sigmaIdxMin = 1;
cIdxMin = 1;

% retreiving the min value indexes using fund function
[sigmaIdxMin cIdxMin] = find(CV_ERR == min(min(CV_ERR)),1);

% assigning the min value using the indexes
sigma = sigma_Vec(sigmaIdxMin);
C = C_vec(cIdxMin);


% =========================================================================

end
