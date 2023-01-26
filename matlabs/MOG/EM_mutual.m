% test of mutual infomation theory
% Mutual information theory for adaptive mixture models

close all
clear all
% change input data here
data=[1.25*randn(500,3)+10; 1.5*randn(1000,3)-2; randn(1500,3)+3; 5*randn(500,3)+30;];

figure
hist(data);
% intilize EM paramters
K=7;
disp('The initial no. of components is:'); disp(K);
TH=0.00001;
max_fai=1000;
I_total=1000;
[n D] = size(data);
p = ones(1,K)/K;      % mixing proportions
mu = randn(D,K);      % means
s2 = zeros(D,D,K);

for i=1:K
  s2(:,:,i) = -100*diag(log(rand(D,1))); % variances
end

while (max_fai>0)%&(I_total>TH)
[p, mu, s2]=my_mog(data,K,1000, p, mu, s2);
%[N, D]=size(data);
clear p_inter H H2 I I_total fai fai2
for i=1:K
    for j=1:K
            p_inter(i,j)=(2*pi)^(-0.5*D)*det(s2(:,:,j))^(-0.5)*exp(-0.5*(mu(:,i)-mu(:,j))'*inv(s2(:,:,j))*(mu(:,i)-mu(:,j)))*p(i);
            
    end
end
mm=max(p_inter, [],2);
for i=1:K
p_inter(i,:)=p_inter(i,:)./mm(i);
end

for i=1:K
    for j=1:K
        if abs(p_inter(i,j))>TH
        fai(i,j)=p(j)*p_inter(i,j)*log(p_inter(i,j)/p(i));
        fai2(i,j)=p(i)*p_inter(j,i)*log(p_inter(j,i)/p(j));
       else
        fai(i,j)=0;
        fai2(i,j)=0;
       end
    end
end
% compute I total in another way
I_total=0;
for i=1:K
    for j=1:K
        if (i~=j)
        I_total=I_total+p(i)*fai(i,j);
        end
    end
end

max_fai=-1000;
index=[0 0];
for i=1:K
    for j=1:K
        if (i~=j)
            if fai(i,j)>max_fai
                max_fai=fai(i,j);
                index=[i,j];
            end
        end
    end
end
if max_fai>0
    K=K-1;
    if p(index(1))>p(index(2))
        p(index(2))=[];
        mu(:,index(2))=[];
        s2(:,:,index(2))=[];
    else
        p(index(1))=[];
        mu(:,index(1))=[];
        s2(:,:,index(1))=[];
    end
end
disp(I_total);%disp(max_fai);
end  % end of while
disp('The optimal no. of component found is:'); disp(K);           

