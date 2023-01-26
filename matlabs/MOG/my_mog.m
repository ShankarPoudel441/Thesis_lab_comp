function [prior_op, mean_op, cov_op] = mog(x,k,niter, p, mu, s2)

% x is the data , k is the components, iter is the no. of iterations

% Simple script to do EM for a mixture of Gaussians.
% -------------------------------------------------
% Oct 15, 2001 - Rasmussen and Ghahramani

% Load some data:

% Initialise parameters

[n D] = size(x);        % number of observations (n) and dimension (D)
%k = 6;                % number of components
%p = ones(1,k)/k;      % mixing proportions
%mu = randn(D,k);      % means
%s2 = zeros(D,D,k);    % covariance matrices
%niter=1000;             % number of iterations

% initialize variances from independent exponential on diagonal
for i=1:k
  s2(:,:,i) = -100*diag(log(rand(D,1))); % variances
end

set(gcf,'Renderer','zbuffer');

clear Z;
% run EM for niter iterations
for t=1:niter,
    %fprintf('t=%d\n',t);
    % Do the E-step:
    for i=1:k
      Z(:,i) = p(i)*det(s2(:,:,i))^(-0.5)*exp(-0.5*sum((x'-repmat(mu(:,i),1,n))'*inv(s2(:,:,i)).*(x'-repmat(mu(:,i),1,n))',2));
    end
    Z = Z./repmat(sum(Z,2),1,k);
    
    % Do the M-step:
    for i=1:k
      mu(:,i) = (x'*Z(:,i))./sum(Z(:,i));
      s2(:,:,i) = (x'-repmat(mu(:,i),1,n))*(repmat(Z(:,i),1,D).*(x'-repmat(mu(:,i),1,n))')./sum(Z(:,i));
      p(i) = mean(Z(:,i));
    end
    if (D==3)
    clf
    hold on
    plot3(x(:,1),x(:,2),x(:,3),'.');
    for i=1:k
      plot_gaussian(s2(:,:,i),mu(:,i),i,11);
    end
    drawnow;
end
end
mean_op=mu;
cov_op = s2;
prior_op = p;
return