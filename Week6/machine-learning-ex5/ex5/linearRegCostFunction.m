function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate the cost (unregularized)
h = X * theta;
error = h - y;
error_sq = error .^ 2;
q = sum(error_sq);
unreg_cost = (1/(2*m)) * q;

% Calculate gradient (unregularized)
h = X * theta;
errors = h - y;
a = X' * errors;
unreg_grad = (1/m) * a;


% Cost regularization
theta(1) = 0;
b = sum(theta .^2);
reg_term = (lambda/(2*m)) * b;

J = unreg_cost + reg_term;

% Gradient regularization
grad_reg_term = (lambda/m) * theta;
grad = grad_reg_term + unreg_grad;










% =========================================================================

grad = grad(:);

end
