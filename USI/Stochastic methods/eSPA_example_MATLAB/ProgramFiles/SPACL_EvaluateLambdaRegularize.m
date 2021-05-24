function [Lambda]=SPACL_EvaluateLambdaRegularize(pi,gamma,m,K);
    Lambda=pi*gamma';
    for k=1:K
        ss=sum(Lambda(:,k));
        if ss>0
         Lambda(:,k)=Lambda(:,k)./sum(Lambda(:,k));
        else
         Lambda(:,k)=1/m;   
        end
    end

end

