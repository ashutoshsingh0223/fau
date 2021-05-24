function [out]=Evaluate_dGamma_dX_v2(in)
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
gamma=zeros(K,T,d+1);
for n_dim=1:(d+1)
     X=Data;
    if n_dim>1
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
        opt{t}.gamma=[];
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
            gamma(:,ttti(tt),n_dim)=x{t};
        end
    end
end
dGamma_dX=zeros(d,d);
dL_dX=zeros(1,d);
for j=1:d
    for t=1:T
        temp=C*(squeeze(gamma(:,t,j+1))-squeeze(gamma(:,t,1)))/eps;
        L_orig=norm(Data(:,t)-C*gamma(:,t,1),2);
        e_d=zeros(d,1);e_d(j)=eps;
        L_eps=norm((Data(:,t)+e_d)-C*gamma(:,t,j+1),2);
        dL_dX(j)=dL_dX(j)+((L_eps-L_orig)/eps)^2/T;
        for i=1:d
            dGamma_dX(i,j)=dGamma_dX(i,j)+temp(i)^2;
        end
    end
end
dGamma_dX=dGamma_dX/T;
dGamma_dX=sum(dGamma_dX,1);
% for n_dim=1:d
%     sss=0;
%     for t=1:T
%         temp=squeeze(gamma(:,t,n_dim+1))-squeeze(gamma(:,t,1));
%         sss=sss+temp'*H*temp;
%         %dGamma_dX(n_dim,k)=sum(gamma(k,:,1).*((squeeze(gamma(k,:,n_dim+1))-squeeze(gamma(k,:,1)))/eps).^2)/sum(gamma(k,:,1));
%     end
%     dGamma_dX(n_dim)=sum(abs(sss/eps))/T;
% end
out.dGamma_dX=dGamma_dX./sum(dGamma_dX);
out.gamma=gamma(:,:,1);
out.dL_dX=dL_dX;
end

