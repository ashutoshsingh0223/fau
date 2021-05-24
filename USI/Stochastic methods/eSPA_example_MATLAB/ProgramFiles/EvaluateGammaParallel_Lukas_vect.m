function [gamma,L]=EvaluateGammaParallel_Lukas_vect(X,C,T,d,K,gamma,xt2)

myeps=1e-9; % this should be the same as outer eps (?)

solve_QP = true;%true;%
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
    if abs(L - L_old) < myeps
        solve_QP = true;
    end
    
end

if solve_QP
    
    H=C'*C;H=(H+H');%H=H+reg_const*(K*eye(K)-ones(K));
    CTX = -2*C'*X;

    if false
        % solve using vectorised SPGQP
        normH = eigs(H,1);
        
        if or(size(gamma,1) ~= K,size(gamma,2) ~= T)
            gamma0 = -CTX; % something as init approximation
        else
            gamma0 = gamma; % reuse provided gamma
        end
        
        [gamma,it] = spgqp_vec(H,-CTX, gamma0, normH, myeps, 1e4);
    end
    
    if true
        % solve one by one using spgqp
        normH = eigs(H,1);
        
        for t=1:T
            f = CTX(:,t);
            
            if or(size(gamma,1) ~= K,size(gamma,2) ~= T)
                gamma0 = -f; % something as init approximation
            else
                gamma0 = gamma(:,t); % reuse provided gamma
            end
            
            gamma(:,t) = spgqp_vec(H,-f, gamma0, normH, myeps, 1e4);
        end
    end

    if false
        % solve one by one using quadprog
        options = optimoptions('quadprog','Display','off');
        lb=zeros(K,1);
        Aeq=ones(1,K);beq=1;
        
        for t=1:T
            f = CTX(:,t);
            
            if or(size(gamma,1) ~= K,size(gamma,2) ~= T)
                gamma0 = -f; % something as init approximation
            else
                gamma0 = gamma(:,t); % reuse provided gamma
            end
            
            [gamma(:,t),L_new] = quadprog(H,f,[],[],...
                Aeq,beq,lb,[],gamma0,options);
            
        end
    end

    L=norm(X - C*gamma,'fro')^2/(T*d);
    
end


end

