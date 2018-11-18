function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
normA = norm(x1-x2);
norm_sq = normA ^ 2;
sigma_sq = sigma^2;
two_sigma_sq = 2 * sigma_sq;
c = norm_sq / two_sigma_sq;
sim = exp(-c);




% =============================================================
    
end
