function [C_qual,C_cpu]=PlotScalingResults_SPACL(out,I,flag)
N=size(out,3);J=size(out,2);
C_qual=zeros(I,J);C_cpu=C_qual;
if flag==1
    for i=1:I
        for j=1:J
    for ii=1:N;a(ii)=out{i,j,ii}{1}.L_pred_valid_Markov;b(ii)=out{i,j,ii}{1}.time_Markov;end;
    C_qual(i,j)=mean(a);C_cpu(i,j)=mean(b);
        end
    end
else
    for i=1:I
        for j=1:J
    for ii=1:N;a(ii)=out{i,j,ii}.L_pred_valid;b(ii)=out{i,j,ii}.time;end;
    C_qual(i,j)=mean(a);C_cpu(i,j)=mean(b);
        end
    end    
end
    