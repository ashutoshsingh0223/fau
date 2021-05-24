function [out_lsd]= LSD_Classify(X_train,X_valid, K)

% perform LSD clustering on training set
[out_lsd_train] = LSD_Cluster(X_train,K);

C_train = out_lsd_train.C;
gamma_train = out_lsd_train.gamma;
[D_train,T_train] = size(X_train);
L_train = (1/(T_train*D_train))* norm(X_train - C_train*gamma_train,'fro')^2;


% find gamma for validation set - solve gamma problem
C_valid = C_train; % C is the same (i.e. trained)

gamma_myeps = 1e-6;
gamma_maxit = 500;
gamma_valid = EvaluateGamma(X_valid,C_valid,[],gamma_myeps,gamma_maxit);

[D_valid,T_valid] = size(X_valid);
L_valid = (1/(T_valid*D_valid))* norm(X_valid - C_valid*gamma_valid,'fro')^2;

% X_train: set output variables
out_lsd.C = out_lsd_train.C;
out_lsd.gamma = out_lsd_train.gamma;
out_lsd.err_discr = L_train;

% X_valid: set output variables
out_lsd.C_valid = C_valid;
out_lsd.gamma_valid = gamma_valid;
out_lsd.err_discr_valid = L_valid;

end

function [out_lsd] = LSD_Cluster(X,K)
% implementation of Hierarchical Rotation-based LSD Algorithm
% from
% "R. Arora, M.R. Gupta, A. Kapila, M. Fazel: Similarity-based Clustering by
% Left-Stochastic Matrix Factorization", Journal of Machine Learning
% Research 14 (2013), pp. 1417-1452
% algorithm can be found on page 1427
%
%

[n,T]=size(X);

P = zeros(K,T);

% assemble similarity matrix
KK = get_similarity_matrix(X);

%options.tol = 1e-16;
%options.disp = 1;
%options.p = 2*K;
options.maxit = 1e5;

[V,D,myflag] = eigs(KK,K,'lm',options);
%[V,~,myflag] = eigs(KK,K,'la',options);
%[V,~,myflag] = eigs(KK,K);

if myflag ~= 0
    keyboard
end

% Lukas: this is my fix - original formula was not working
%cstar = norm(inv(V'*V)*V'*ones(T,1),2)^2/K;
%cstar = norm(V'*ones(T,1),1)/K; 
%cstar = cstar/sqrt(T);
cstar = K;

KK = cstar*KK;

idx = 1:T;
W = zeros(2,1);

idx_K2 = cell(2,1);

for i=1:K-1
    % compute LSD for two clusters
    P_K2 = LSD_Rotate_K2(KK(idx,idx));
    
    % write new probabilities to output matrix
    if i == 1
       P(1:2,idx) = P_K2;
    else
       % Lukas: take previous probabilities and split them into two new rows 
       % to satisfy left-stochastic condition for P
       P(i+1,idx) = P_K2(2,:).*P(i,idx);
       P(i,idx) = P_K2(1,:).*P(i,idx); % do not change the order of those two rows!
    end

    idx_K2{1} = find(P(i,:) >= 0.5); % points in first clusters
    idx_K2{2} = find(P(i+1,:) > 0.5); % points in the second cluster

    % compute average within-cluster similarity
    for j=1:2
        nm = length(idx_K2{j});
        W(j) = 0;
        for jj=1:length(idx_K2{j})
            for ii=1:jj
                W(j) = W(j) + KK(idx_K2{j}(ii),idx_K2{j}(jj));
% Lukas idea: no probabilities? W(j) = W(j) + P(i,jj)*P(i,ii)*KK(idx_K2{j}(ii),idx_K2{j}(jj));
            end
        end
        W(j) = W(j)/(nm*(nm - 1));
    end
    
    % which cluster to split next?
    % the one with smaller average within-cluster similarity
    if W(1) <= W(2)
       idx = idx_K2{1}; 
    else
       idx = idx_K2{2}; 
    end
    
    if isempty(idx)
        break;
    end
end

% fix: if gamma_k=0 then P*P' is singular
Psum = sum(P,2);
idx = find(Psum > 0);

% find matrix F solving unconstrained min | X - FP |_F
F = zeros(n,K);
F(:,idx) = ((P(idx,:)*P(idx,:)')\(P(idx,:)*X'))';

out_lsd.gamma = P;
out_lsd.C = F;
out_lsd.L = norm(X - F*P,'fro')^2/(T*n);

end

function [P] = LSD_Rotate_K2(KK)

T = size(KK,1);

% this is variant for two clusters
K = 2;

% compute rank-k eigendecomposition
%options.tol = 1e-16;
%options.disp = 1;
%options.p = 2*K;
options.maxit = 1e5;

[V,D,myflag] = eigs(KK,K,'lm',options);
%[V,D,myflag] = eigs(KK,K,'la',options);
%[V,D,myflag] = eigs(KK,K);

if myflag ~= 0
    keyboard
end

% Lukas fix: maybe numerically D < 0?
D = max(D,0);

% compute M
M = zeros(K,T);
for k=1:K
    M(k,:) = sqrt(D(k,k))*V(:,k)';
end

% compute normal to least-squares hyperplane fit to columns of M
m = (M*M')\sum(M,2); % = inv(M*M')*M*ones(T,1);

% project columns of M onto the hyperplane normal to m that passes through
% the origin
Mtilde = (eye(K) - m*m'/norm(m))*M;

% shift columns 1/sqrt(K) units in direction of m
Mtilde = Mtilde + 1/(sqrt(K)*norm(m))*kron(ones(1,size(M,2)),m);

% compute a rotation (see Subrutine 1 or if K=2 formula in Section 3.2.2)
u = 1/sqrt(K)*ones(K,1); % unit normal to probability simplex
Rs = RotateK2(m,u);

% compute matrix Q
Q = Rs*Mtilde;

Ru = eye(K);

% compute the column-wise Euclidean projection onto the simplex
P = projection_simplexes(Ru*Q);

end


function [Rs] = RotateK2(m,u)
% see page 1425 formula in the end of the page
uTm = u'*m;
mTm = m'*m;
v = u - (uTm/mTm)*m;
U = [m/norm(m), v/norm(v)];
RG = [uTm, -1+(uTm)^2; 1-(uTm)^2, uTm];

Rs = U*RG*U';
end

function KK = get_similarity_matrix(X)
sigma = 1;

%T = size(X,2);
%KK = zeros(T,T);
%for t1=1:T
%    for t2=1:T
%        KK(t1,t2) = exp(-norm(X(:,t1) - X(:,t2),2)^2/(2*sigma^2) );
%    end
%end

%KK = X'*X;%
KK = exp(-squareform(pdist(X','minkowski').^2)/(2*sigma^2));

end

