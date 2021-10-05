function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sumCost = 0;

for i = 1:m
  xi = X(i,:);
  yi = y(i);
  hyp = sigmoid(xi * theta);
  sumCost = sumCost + (-yi * log(hyp) - (1 - yi) * log(1 - hyp));
  grad = grad + ((hyp - yi) .* xi)';
 end

J = sumCost / m;
grad = grad / m;


% =============================================================

end
