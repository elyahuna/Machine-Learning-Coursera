function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
	
	%calculating y values prediction according the current epsilon
	y_pred = (pval < epsilon);
	
	%calculating tp by all y's which are ones in actual and in prediction
	tp = sum(yval .* y_pred);
	%false positive is given by when predict is 1 but actual is 0, so def is -1.
	fp = sum( (yval - y_pred) == -1);
	%false negative is when actual is 1 but predict is zero, def is 1.
	fn = sum( (yval - y_pred) == 1);
	
	%calculating prec and rec according to the equations.
	%in case of zero in the denominator we will set the value to 1
	if (tp + fp) == 0
		prec = 1;
	else
		prec = tp / (tp + fp);
	end
	
	if (tp + fn) == 0
		rec = 1;
	else
		rec = tp / (tp + fn);
	end
	
	%calculating F1 score according to the equation.
	F1 = 2*prec*rec / (prec+rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
