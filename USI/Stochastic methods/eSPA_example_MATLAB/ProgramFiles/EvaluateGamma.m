function [gamma,L]=EvaluateGamma(X,C,gamma,myeps,maxit)


[d,T] = size(X);
K = size(C,1);

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
    if abs(L - L_old) < myeps
        solve_QP = true;
    end
end


if solve_QP
    H=C'*C;H=(H+H');
    CTX = -2*C'*X;
    
    % solve using vectorised SPGQP
    normH = eigs(H,1);
    
    if or(size(gamma,1) ~= K,size(gamma,2) ~= T)
        gamma0 = -CTX; % something as init approximation
    else
        gamma0 = gamma; % reuse provided gamma
    end
    
    [gamma,it] = spgqp_vec(H,-CTX, gamma0, normH, myeps, maxit);
    
    L=norm(X - C*gamma,'fro')^2/(T*d);
    
end


end

