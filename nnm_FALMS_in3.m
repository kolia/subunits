function [X,err,obj] = nnm_FALMS_in3(A,b,rho,rep,Xt,initX)

% solve the problem (fast algorithm)
% argmin 0.5||A(X) - b||^2 + rho*||X||_*
% A is p x 1 cell
% b is p x 1 vector

p = length(b);
m = size(A{1},1);
n = size(A{1},2);

Av = zeros(p,m*n);
for i = 1:p
    Av(i,:) = (A{i}(:))';
end

% initialization

im = norm(Av*Av'); mu = 1/im; 
 % largest eigenvalue of Av*Av'. For large numbers can use expected values

Xn = initX; %randn(m,n); 
Y = Xn; Z = Xn; Ym = Y; Ymm = Y;
[U,G,V] = svd(Y);
svd_z = sum(diag(G));

L = -U(:,1:nnz(G))*V(:,1:nnz(G))';

% Woodbury calculation of quadratic term in (1)
iX = mu*eye(m*n) - mu^2*Av'*((eye(p) + mu*(Av*Av'))\Av);
iXAvb = iX*Av'*b;
tk = 1; tkm = tk;

skip = 0;
obj = zeros(rep,1);
err = zeros(rep,1);
for k = 1:rep    
    
    % optimize (1)
    Xn = reshape(iXAvb + iX*(L(:) + im*Z(:)),m,n);

    % see pseudocode
    sd = sum(svd(Xn,'econ'));
    if rho*sd > rho*svd_z - L(:)'*(Xn(:)-Z(:)) + 0.5*im*norm(Xn-Z,'fro')^2
        if skip == 0
            tk = (1 + sqrt(1+8*tkm^2))/2;
        else
            tk = (1 + sqrt(1+4*tkm^2))/2;
        end
        Xn = Ym + (tkm - 1)/tk*(Ym - Ymm);
        skip = 1;
        Z = Xn;

        % just so we can calculate obj(k)
        sd = sum(svd(Xn,'econ'));
    else
        skip = 0;
    end
    
    % just for tracking 
    obj(k) = sum((Av*Xn(:) - b).^2) + rho*sd;

    % F = X - mu grad(f(X))
    F = Xn;
    for i = 1:p
        F = F - mu*(A{i}(:)'*Xn(:)-b(i))*A{i};
    end
    
    % shrinkage of nuclear norm by mu*rho : optimize (2)
    [U,G,V] = svd(F);
    G(1:min(m,n),1:min(m,n)) = diag(max(diag(G(1:min(m,n),1:min(m,n))) - mu*rho,0));
    Y = U*G*V';    

    % pseudocode
    if all(Z(:)==Xn(:))
      tkp = (1 + sqrt(1+2*tk^2))/2;  
    else
      tkp = (1 + sqrt(1+4*tk^2))/2;  
    end
    Z = Y + ((tk-1)/tkp)*(Y - Ym);
    
    % subgradient of nuclear norm
    [U,G,V] = svd(Z);
    L = -rho*U(:,1:nnz(G))*V(:,1:nnz(G))';

    tkm = tk;
    tk = tkp;
    Ymm = Ym;
    Ym = Y;
    
    err(k) = norm(Xt-Xn,2);
    
    % for next iteration skipping condition
    svd_z = sum(diag(G));
end
    
X = Xn;    