function [gamma]=EvaluateGammaParallel(in)
Data=in.X;
C=in.C;
T=in.T;
d=in.d;
K=in.K;
eps=in.eps;
xt2=0;
Size_FEM=1;
H=C'*C;H=(H+H');%H=H+reg_const*(K*eye(K)-ones(K));
A=-eye(K);
b=zeros(K,1);
Aeq=ones(1,K);beq=1;
L=0;
%gamma=zeros(K,T);
options = optimoptions('quadprog','Display','off');
options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.TolX = 1e-12;
int_T=1:Size_FEM:T;
if int_T(length(int_T))~=T
   int_T=[int_T T]; 
end
for n_dim=1:d+1
    if n_dim==1
        X=Data;
    else
        X(n_dim-1,:)=X(n_dim-1,:)+eps;
    end
    for t=1:(length(int_T)-1)
        if t<(length(int_T)-1)
            ttti=int_T(t):(int_T(t+1)-1);
        else
            ttti=int_T(t):(int_T(t+1));
        end
        opt{t}.A=A;
        opt{t}.b=b;
        opt{t}.Aeq=Aeq;
        opt{t}.beq=beq;
        opt{t}.H=H*length(ttti);
        opt{t}.f=(-2*C'*sum(X(:,ttti),2))';
        opt{t}.options=options;
        if ~isempty(gamma)
            opt{t}.gamma=gamma(:,int_T(t))';
        else
            opt{t}.gamma=[];
        end
    end
    for t=1:(length(int_T)-1)
        %f=(-2*C'*X(:,t))';
        [x{t},fval{t}] = quadprog(opt{t}.H,opt{t}.f,[],[],...
            opt{t}.Aeq,opt{t}.beq,opt{t}.b,[],opt{t}.gamma,opt{t}.options);
    end
    for t=1:(length(int_T)-1)
        L=L+fval{t};
        if t<(length(int_T)-1)
            ttti=int_T(t):(int_T(t+1)-1);
        else
            ttti=int_T(t):(int_T(t+1));
        end
        for tt=1:length(ttti)
            gamma(:,ttti(tt))=x{t};
        end
    end
end
end

