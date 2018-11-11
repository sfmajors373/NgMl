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

h = X * theta;
sig = sigmoid(h); % This is my hypothesis vector
error = sig - y;
grad = (1/m) * (X' *error);  % This is the gradient


insideLeft = -y' * (log(sig));
insideRight = (1 - y)' * log(1 - sig);
insideAll = insideLeft - insideRight;
unRegularizedCost = 1/m * insideAll;

%%  Not sure I understand why I am doing this
%% It is for excluding the bias feature?
%theta(1) = 0;
%sqTheta = theta' * theta;
%regularizationTerm = (lambda / (2*m)) * sqTheta;
%regularizedCost = unRegularizedCost + regularizationTerm;

J = unRegularizedCost;


% =============================================================

end
