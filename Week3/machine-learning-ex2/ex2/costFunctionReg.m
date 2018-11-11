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


h = X * theta;
sig = sigmoid(h); % This is my hypothesis vector
error = sig - y;
gradOne = (1/m) * (X' *error);

insideLeft = -y' * (log(sig));
insideRight = (1 - y)' * log(1 - sig);
insideAll = insideLeft - insideRight;
unRegularizedCost = insideAll / m;

%%  Not sure I understand why I am doing this
%% It is for excluding the bias feature?
theta(1) = 0;
sqTheta = theta' * theta;
regularizationTerm = (lambda / (2*m)) * sqTheta;
regularizedCost = unRegularizedCost + regularizationTerm;

scaledTheta = (lambda / m) * theta;

grad = gradOne + scaledTheta;
J = regularizedCost;



% =============================================================

end
