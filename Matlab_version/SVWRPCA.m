function [L, S] = SVWRPCA(X,lambda, tol, max_iter)
    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
    % - mu - the augmented lagrangian parameter, default = 10*lambda
    % - tol - reconstruction error tolerance, default = 1e-6
    % - max_iter - maximum number of iterations, default = 1000

    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');

    % default arguments
    if nargin < 2
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 3
        tol = 1e-6;
    end
    if nargin < 4
        max_iter = 1e4;
    end
    
    % initial solution
    L = X;
    S = zeros(M, N);
    Y = zeros(M, N);
    mu = 10*lambda;
   [~, S0,~] = svd(L, 'econ');
        
        S_Vec         =   max(max(S0)',0);
        T=mean(S_Vec);
    for iter = (1:max_iter)
        [~, S0,~] = svd(L, 'econ');
        
        S_Vec         =   max(max(S0)',0);
        W_Vec   =  T./( S_Vec+eps );   % Weight vector
        % ADMM step: update L and S
        tau=W_Vec/mu;
        %tau=W_Vec;
        X1= X - S + (1/mu)*Y;
         % shrinkage operator for singular values
        [U, S, V] = svd(X1, 'econ');
        So = sign(S) .* max(abs(S) - diag(tau), 0);
        A=max(abs(S) - diag(tau), 0);
        L = U*So*V';
       
        
        
        tau2=lambda/mu';
        temp=X - L + (1/mu)*Y;
        S=sign(temp) .* max(abs(temp) - tau2, 0);
     
        % and augmented lagrangian multiplier
        Z = X - L - S;
        Z(unobserved) = 0; % skip missing values
        Y = Y + mu*Z;
        
        err = norm(Z, 'fro') / normX;
%         if (err <= tol)|| (mod(iter, 100) == 0)||(iter>max_iter)
%              fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
%                      iter, err, rank(L), nnz(S(~unobserved)));
%         end
        if (err <= tol||iter>max_iter) 
            break; 
        end
        
    end
end

