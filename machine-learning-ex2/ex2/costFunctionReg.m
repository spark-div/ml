function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
szt = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

   Hth = sigmoid(X*theta);
   term1 = ((-y)')*(log(Hth));
   term2 = ((1.-y)')*log(1.-Hth);
   SigMega= term1-term2;
   J1st = sum(SigMega)/m;
   regthet = theta(2:szt, 1);
   SumSqrd = sum(regthet.^2);
   Jreg = (lambda*SumSqrd)/(2*m);
   J = J1st + Jreg;

   C = Hth - y;
   D = (C'*X)';
   L = lambda*ones(szt, 1);
   L(1) = 0;
   RegTheta = L/m;
   RegTheta = RegTheta .* theta;
   grad = D/m + RegTheta; 

% =============================================================

end
