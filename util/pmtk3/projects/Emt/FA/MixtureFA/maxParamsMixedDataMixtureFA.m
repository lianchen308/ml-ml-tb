function params = maxParamsMixtureFA(ss, data, params, options)
% maximize params for mixutre of FAs

  [Dm,Nm] = size(data.categorical);
  [Dc,Nc] = size(data.continuous);
  N = max([Nc Nm]);
  [Dz,K] = size(params.mean);

  params.mixProb = (ss.sumMixProb + params.alpha0)/(N + K*params.alpha0);
  params.mean = bsxfun(@times, ss.sumMean, 1./ss.sumMixProb(:)');
  t = 0;
  for k = 1:K
    if options.estimateCovMat
      % haven't chcked this part yet
      if ~options.regCovMat
        params.covMat(:,:,k) = ss.sumCovMat(:,:,k)/ss.sumMixProb(k) - params.mean(:,k)*params.mean(:,k)';
      else
        den = params.nu0 + Dz + 1 + ss.sumMixProb(k);
        params.covMat(:,:,k) = (params.S0 + ss.sumCovMat(:,:,k))/den - N*params.mean*params.mean'/den;
      end
      params.precMat(:,:,k) = inv(params.covMat(:,:,k));
    end

    % beta
    if options.estimateBeta
      params.beta(:,:,k) = ss.sumLhs(:,:,k)*inv(ss.sumCovMat(:,:,k));
      if Dc >0
        params.betaCont(:,:,k) = params.beta(1:Dc,:,k);
        t = t + ss.sumYY(:,k) - diag(params.betaCont(:,:,k)*ss.sumLhs(1:Dc,:,k)'); % for Phi
      end
      if Dm > 0
        params.betaMult(:,:,k) = params.beta(Dc+1:end,:,k);
      end
    end
  end
  if Dc >0
    % estimate noise variance for continuous variables
    if options.estimateBeta
      params.noiseCovMat = diag(2*params.b + t)./(N + 2*(params.a + 1));
    else
      % Phi update without substituting new beta
      numl = ss.sumPhi + 2*params.b;
      params.noiseCovMat = diag(numl./(N + 2*(params.a + 1)));
    end
    params.noisePrecMat = diag(1./diag(params.noiseCovMat));
  end

  % variational params
  if Dm > 0
    params.psi = ss.psi;
  end

