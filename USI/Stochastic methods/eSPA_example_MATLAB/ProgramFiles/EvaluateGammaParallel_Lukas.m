function [gamma,L]=EvaluateGammaParallel_Lukas(X,C,T,d,K,gamma,xt2)

eps=1e-10; % this should be the same as outer eps (?)

solve_QP = false;
if isempty(gamma)
    solve_QP = true;
end

% use one step just to decrease objective function
if ~solve_QP
    CTC = C'*C;
    alpha_bar = 1/eigs(CTC,1);
    
    % store the original value of objective function
    L_old = norm(X - C*gamma,'fro')^2/(T*d);

    % gradient of objective function
    mygrad = 2*CTC*gamma - 2*C'*X;
    
    % Dostal-Schoberl step
    gamma = projection_simplexes(gamma - alpha_bar*mygrad); 

    % compute new objective function
    L=norm(X - C*gamma,'fro')^2/(T*d);
    
    % if the decrease is not sufficient, then solve the QP
    if abs(L - L_old) < eps
        solve_QP = true;
    end
    
end

if solve_QP
    L = 0;
    
    options = optimoptions('quadprog','Display','off');
    H=C'*C;H=(H+H');%H=H+reg_const*(K*eye(K)-ones(K));
    lb=zeros(K,1);
    Aeq=ones(1,K);beq=1;
    
    if true
        % solve one by one
        CTX = -2*C'*X;
        
        for t=1:T
            f = CTX(:,t);
            
            if or(size(gamma,1) ~= K,size(gamma,2) ~= T)
                gamma0 = [];
            else
                gamma0 = gamma(:,t); % reuse provided gamma
            end
            
            [gamma(:,t),L_new] = quadprog(H,f,[],[],...
                Aeq,beq,lb,[],gamma0,options);
            L = L + L_new;
        end
    end
    
    if false % just the test
        % solve one big QP
        T = size(X,2);
        
        CX = -2*C'*X;
        f_big = reshape(CX',K*T,1);
        H_big = kron(eye(T),H);
        lb_big = zeros(K*T,1);
        Aeq_big = kron(ones(1,K),eye(T));
        beq_big = ones(T,1);
        gamma0_big = [];
        
        [gamma_big,L_new] = quadprog(H_big,f_big,[],[],...
            Aeq_big,beq_big,lb_big,[],gamma0_big,options);
        
        gamma = reshape(gamma_big,T,K)';
        L = L + L_new;
    end
    
    LL=norm(X - C*gamma,'fro')^2/(T*d);
    L = LL;
%    abs(LL - L) % there is a difference and I don't know why

% now I know why, there is a test:
%    xt2_2 = 0;
%    for t=1:T
%       xt2_2 = xt2_2 + X(:,t)'*X(:,t); 
%    end
%    abs(xt2_2 - xt2)
    
end


end

