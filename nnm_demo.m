clear all; clc;
m = 20;
n = 30; % dimensions
r = 2;  % rank
rep = 500;
A1 = randn(m,r);
A2 = randn(r,n);
X = A1*A2;

p = 4*(m+n)*r;
A = cell(p,1);
b = zeros(p,1);
w = .3;

for i = 1:p
    Ar = randn(m,n);
    Ar = Ar/norm(Ar,'fro');
    A{i} = Ar;
    b(i) = trace(Ar'*X) + w*randn;
end

rho = 1/n; %1/sqrt(n);
e = w;

Xin = randn(m,n);

tic; [Xr,err,obj] = nnm_FALMS_in3(A,b,rho,e,rep,X,Xin); toc

nn = min([X(:);Xr(:)]);
mm = max([X(:);Xr(:)]);

figure;
    subplot(321); imagesc(X,[nn,mm]); title('Original'); colorbar; axis square;
    subplot(322); imagesc(Xr,[nn,mm]); title('Recovered'); colorbar; axis square;
    subplot(3,2,3:4); plot(1:rep,err/norm(X,'fro')); xlabel('Iteration #'); ylabel('Relative Error'); 
    subplot(3,2,5:6); semilogy(1:rep,obj); ylabel('Objective Function');