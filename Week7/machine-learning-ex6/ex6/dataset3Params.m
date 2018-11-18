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
choices = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Matrix for storing error
%prediction_error = zeros(length(choices), length(choices));
Cindex = 0;
sigmaIndex = 0;
lowestError = 10000;
% For loop iterating over choices for C
for i = 1:8;
  tempC = choices(i);
% For loop iterating over choices for sigma
  for j = 1:8;
    tempSigma = choices(j);
    % Train the model with our trial C and sigma
    model= svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    if prediction_error < lowestError;
      lowestError = prediction_error;
      Cindex = i;
      sigmaIndex = j;
  end
end

sigma = choices(sigmaIndex);
C = choices(Cindex);

% Find the minimum error
%[minRow, Crow] = min(prediction_error);
%[minCol, sigmaCol] = min(prediction_error(minRow));
%C = choices(Crow);
%sigma = choices(sigmaCol);


% =========================================================================

end
