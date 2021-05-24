function Px = projection_simplex( x,n)

       %n = length(x); 

       [y,idx] = sort(x); 
       
       t_hat = 0;
       
       i = n-1;
       while i >= 1 
           ti = (sum(y(i+1:n))-1)/(n-i);
           if ti >= y(i)
               t_hat = ti;
               break
           else
               i = i - 1;
           end
       end
       
       if i == 0
           t_hat = (sum(y(1:n))-1)/n;
       end
       
       Px = max(x-t_hat,zeros(n,1));

end

