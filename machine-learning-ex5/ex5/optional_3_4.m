function [errorTest] = ...
    optional_3_4(X_poly_train, y_train, X_poly_test, ytest, lambda)

% This function implements optional part 3.4 in ex5.
% Calculation of the test error with a given lambda.
% in order for it to be independed of previous step, the theta is being calculated locally.

% Input: 
%		X_poly_train and y_train - the training set
%		X_poly_test and ytest - the tests set.
%		lambda - the given value of lambda for which this function will return the test error.
% Output:
%		errorTest - the test error for the given lambda.
	
theta = trainLinearReg(X_poly_train, y_train, lambda);
[errorTest, ~] = linearRegCostFunction(X_poly_test, ytest, theta, 0);
fprintf('The test error with lambda=%f is %f\n', lambda, errorTest);

end
