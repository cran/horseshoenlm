#' Horseshoe shrinkage prior in Bayesian linear regression
#'
#'
#' This function employs the algorithm provided by van der Pas et. al. (2016) for 
#' linear model to fit Bayesian regression. 
#'
#'  The model is:
#'  \eqn{y_i} is response,
#'  \eqn{y_i=X\beta+\epsilon, \epsilon \sim N(0,\sigma^2)}.
#'
#' @references Maity, A. K., Carroll, R. J., and Mallick, B. K. (2019) 
#'             "Integration of Survival and Binary Data for Variable Selection and Prediction: 
#'             A Bayesian Approach", 
#'             Journal of the Royal Statistical Society: Series C (Applied Statistics).
#'             
#'             Maity, A. K., Bhattacharya, A., Mallick, B. K., & Baladandayuthapani, V. (2020). 
#'             Bayesian data integration and variable selection for pan cancer survival prediction
#'             using protein expression data. Biometrics, 76(1), 316-325.
#'             
#'             Stephanie van der Pas, James Scott, Antik Chakraborty and Anirban Bhattacharya (2016). horseshoe:
#'             Implementation of the Horseshoe Prior. R package version 0.1.0.
#'             https://CRAN.R-project.org/package=horseshoe
#'             
#'             Enes Makalic and Daniel Schmidt (2016). High-Dimensional Bayesian Regularised Regression with the
#'             BayesReg Package arXiv:1611.06649
#'
#'@param y Response vector.
#'@param X Matrix of covariates, dimension \eqn{n*p}.
#'@param method.tau Method for handling \eqn{\tau}. Select "truncatedCauchy" for full
#' Bayes with the Cauchy prior truncated to [1/p, 1], "halfCauchy" for full Bayes with
#' the half-Cauchy prior, or "fixed" to use a fixed value (an empirical Bayes estimate,
#' for example).
#'@param tau  Use this argument to pass the (estimated) value of \eqn{\tau} in case "fixed"
#' is selected for method.tau. Not necessary when method.tau is equal to "halfCauchy" or
#' "truncatedCauchy". The default (tau = 1) is not suitable for most purposes and should be replaced.
#'@param method.sigma Select "Jeffreys" for full Bayes with Jeffrey's prior on the error
#'variance \eqn{\sigma^2}, or "fixed" to use a fixed value (an empirical Bayes
#'estimate, for example).
#'@param Sigma2 A fixed value for the error variance \eqn{\sigma^2}. Not necessary
#'when method.sigma is equal to "Jeffreys". Use this argument to pass the (estimated)
#'value of Sigma2 in case "fixed" is selected for method.sigma. The default (Sigma2 = 1)
#'is not suitable for most purposes and should be replaced.
#'@param burn Number of burn-in MCMC samples. Default is 1000.
#'@param nmc Number of posterior draws to be saved. Default is 5000.
#'@param thin Thinning parameter of the chain. Default is 1 (no thinning).
#'@param alpha Level for the credible intervals. For example, alpha = 0.05 results in
#'95\% credible intervals.
#'@param Xtest test design matrix.
#'
#'@return 
#' \item{yHat}{Predictive response}
#' \item{BetaHat}{Posterior mean of Beta, a \eqn{p} by 1 vector}
#' \item{LeftCI}{The left bounds of the credible intervals}
#' \item{RightCI}{The right bounds of the credible intervals}
#' \item{BetaMedian}{Posterior median of Beta, a \eqn{p} by 1 vector}
#' \item{LambdaHat}{Posterior samples of \eqn{\lambda}, a \eqn{p*1} vector}
#' \item{Sigma2Hat}{Posterior mean of error variance \eqn{\sigma^2}. If method.sigma =
#' "fixed" is used, this value will be equal to the user-selected value of Sigma2
#' passed to the function}
#' \item{TauHat}{Posterior mean of global scale parameter tau, a positive scalar}
#' \item{BetaSamples}{Posterior samples of \eqn{\beta}}
#' \item{TauSamples}{Posterior samples of \eqn{\tau}}
#' \item{Sigma2Samples}{Posterior samples of Sigma2}
#' \item{LikelihoodSamples}{Posterior samples of likelihood}
#' \item{DIC}{Devainace Information Criterion of the fitted model}
#' \item{WAIC}{Widely Applicable Information Criterion}
#'
#' @importFrom stats dnorm pnorm rbinom rnorm var dbinom
#' @examples
#'
#' burnin <- 500
#' nmc    <- 1000
#' thin <- 1
#' y.sd   <- 1  # standard deviation of the response
#' 
#' p <- 100  # number of predictors
#' ntrain <- 100  # training size
#' ntest  <- 50   # test size
#' n <- ntest + ntrain  # sample size
#' q <- 10   # number of true predictos
#' 
#' beta.t <- c(sample(x = c(1, -1), size = q, replace = TRUE), rep(0, p - q))  
#' x <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(p))    
#' 
#' tmean <- x %*% beta.t
#' y <- rnorm(n, mean = tmean, sd = y.sd)
#' X <- scale(as.matrix(x))  # standarization
#' 
#' # Training set
#' ytrain <- y[1:ntrain]
#' Xtrain <- X[1:ntrain, ]
#' 
#' # Test set
#' ytest <- y[(ntrain + 1):n]
#' Xtest <- X[(ntrain + 1):n, ]
#' 
#' posterior.fit <- lmhs(y = ytrain, X = Xtrain, method.tau = "halfCauchy",
#'                        method.sigma = "Jeffreys", burn = burnin, nmc = nmc, thin = 1,
#'                        Xtest = Xtest)
#'                              
#' posterior.fit$BetaHat
#' 
#' # Posterior processing to recover the true predictors
#' cluster <- kmeans(abs(posterior.fit$BetaHat), centers = 2)$cluster
#' cluster1 <- which(cluster == 1)
#' cluster2 <- which(cluster == 2)
#' min.cluster <- ifelse(length(cluster1) < length(cluster2), 1, 2)
#' which(cluster == min.cluster)  # this matches with the true variables
#' 
#'
#' @export





lmhs <- function(y, X, method.tau = c("fixed", "truncatedCauchy","halfCauchy"), tau = 1,
                  method.sigma = c("fixed", "Jeffreys"), Sigma2 = 1,
                  burn = 1000, nmc = 5000, thin = 1, alpha = 0.05,
                  Xtest = NULL)
{
  
  method.tau = match.arg(method.tau)
  
  method.sigma = match.arg(method.sigma)
  
  
  
  N=burn+nmc
  
  effsamp=(N-burn)/thin
  
  n=nrow(X)
  
  p=ncol(X)
  
  if(is.null(Xtest))
    
  {
    Xtest  <- X
    
    ntest  <- n
    
  } else {
    
    ntest <- nrow(Xtest)
    
  }
  
  
  
  ## parameters ##
  
  Beta     <- rep(0,p)
  
  lambda   <- rep(1,p)
  
  sigma_sq <- Sigma2
  
  
  
  ## output ##
  
  betaout          <- matrix(0, p, effsamp)
  
  lambdaout        <- matrix(0, p, effsamp)
  
  tauout           <- rep(0, effsamp)
  
  sigmaSqout       <- rep(1, effsamp)
  
  likelihoodout    <- matrix(0, n, effsamp)
  
  loglikelihoodout <- rep(0, effsamp)
  
  responseout      <- matrix(0, ntest, effsamp)
  
  
  
  
  ## matrices ##
  
  I_n=diag(n)
  
  l0=rep(0,p)
  
  l1=rep(1,n)
  
  l2=rep(1,p)
  
  if(p < n)
    
  {
    
    Q_star=t(X)%*%X
    
  }
  
  
  
  
  
  ## start Gibb's sampling ##
  
  message("Markov chain monte carlo is running")
  
  for(i in 1:N)
    
  {
    
    
    
    ## update beta ##
    
    if((p > n) || (p == n))
      
    {
      
      lambda_star=tau*lambda
      
      U=as.numeric(lambda_star^2)*t(X)
      
      ## step 1 ##
      
      u=stats::rnorm(l2,l0,lambda_star)
      
      v=X%*%u + stats::rnorm(n)
      
      ## step 2 ##
      
      v_star=solve((X%*%U+I_n),((y/sqrt(sigma_sq))-v))
      
      Beta=sqrt(sigma_sq)*(u+U%*%v_star)
      
    } else {
      
      lambda_star=tau*lambda
      
      L=chol((1/sigma_sq)*(Q_star+diag(1/as.numeric(lambda_star^2),p,p)))
      
      v=solve(t(L),t(t(y)%*%X)/sigma_sq)
      
      mu=solve(L,v)
      
      u=solve(L,stats::rnorm(p))
      
      Beta=mu+u
      
    }
    
    
    
    ## update lambda_j's in a block using slice sampling ##
    
    eta = 1/(lambda^2)
    
    upsi = stats::runif(p,0,1/(1+eta))
    
    tempps = Beta^2/(2*sigma_sq*tau^2)
    
    ub = (1-upsi)/upsi
    
    # now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
    
    Fub = 1 - exp(-tempps*ub) # exp cdf at ub
    
    Fub[Fub < (1e-4)] = 1e-4;  # for numerical stability
    
    up = stats::runif(p,0,Fub)
    
    eta = -log(1-up)/tempps
    
    lambda = 1/sqrt(eta);
    
    
    
    ## update tau ##
    
    ## Only if prior on tau is used
    
    if(method.tau == "halfCauchy"){
      
      tempt = sum((Beta/lambda)^2)/(2*sigma_sq)
      
      et = 1/tau^2
      
      utau = stats::runif(1,0,1/(1+et))
      
      ubt = (1-utau)/utau
      
      Fubt = stats::pgamma(ubt,(p+1)/2,scale=1/tempt)
      
      Fubt = max(Fubt,1e-8) # for numerical stability
      
      ut = stats::runif(1,0,Fubt)
      
      et = stats::qgamma(ut,(p+1)/2,scale=1/tempt)
      
      tau = 1/sqrt(et)
      
    }#end if
    
    
    
    if(method.tau == "truncatedCauchy"){
      
      tempt = sum((Beta/lambda)^2)/(2*sigma_sq)
      
      et = 1/tau^2
      
      utau = stats::runif(1,0,1/(1+et))
      
      ubt_1=1
      
      ubt_2 = min((1-utau)/utau,p^2)
      
      Fubt_1 = stats::pgamma(ubt_1,(p+1)/2,scale=1/tempt)
      
      Fubt_2 = stats::pgamma(ubt_2,(p+1)/2,scale=1/tempt)
      
      #Fubt = max(Fubt,1e-8) # for numerical stability
      
      ut = stats::runif(1,Fubt_1,Fubt_2)
      
      et = stats::qgamma(ut,(p+1)/2,scale=1/tempt)
      
      tau = 1/sqrt(et)
      
    }
    
    
    
    ## update sigma_sq ##
    
    if(method.sigma == "Jeffreys"){
      
      if((p > n) || (p == n))
        
      {
        
        E_1=max(t(y-X%*%Beta)%*%(y-X%*%Beta),(1e-10))
        
        E_2=max(sum(Beta^2/((tau*lambda))^2),(1e-10))
        
      } else {
        
        E_1=max(t(y-X%*%Beta)%*%(y-X%*%Beta),1e-8)
        
        E_2=max(sum(Beta^2/((tau*lambda))^2),1e-8)
        
      }
      
      sigma_sq= 1/stats::rgamma(1, (n + p)/2, scale = 2/(E_1+E_2))
      
    }
    
    
    
    # likelihood
    
    likelihood    <- dnorm(y, mean = X %*% Beta, sd = sqrt(sigma_sq))
    
    loglikelihood <- sum(log(likelihood))
    
    
    
    
    ## Prediction ##
    
    yhat <- Xtest %*% Beta
    
    
    
    if (i%%500 == 0)
      
    {
      
      message("iteration = ", i)
      
    }
    
    
    
    
    
    
    
    if(i > burn && i%%thin== 0)
      
    {
      
      betaout[ ,(i-burn)/thin]          <- Beta
      
      lambdaout[ ,(i-burn)/thin]        <- lambda
      
      tauout[(i - burn)/thin]           <- tau
      
      sigmaSqout[(i - burn)/thin]       <- sigma_sq
      
      likelihoodout[ ,(i - burn)/thin]  <- likelihood
      
      loglikelihoodout[(i - burn)/thin] <- loglikelihood
      
      responseout[ ,(i - burn)/thin]    <- yhat
      
    }
    
  }
  
  
  
  pMean          <- apply(betaout,1,mean)
  
  pMedian        <- apply(betaout,1,stats::median)
  
  pLambda        <- apply(lambdaout,1,mean)
  
  pSigma         <- mean(sigmaSqout)
  
  pTau           <- mean(tauout)
  
  pResponse      <- apply(responseout, 1, mean)
  
  pLikelihood    <- apply(likelihoodout, 1, mean)
  
  pLoglikelihood <- mean(loglikelihoodout)
  
  
  
  loglikelihood.posterior <- sum(c(dnorm(y, mean = X %*% pMean, sd = sqrt(pSigma), log = TRUE)))
  
  DIC  <- -4 * pLoglikelihood + 2 * loglikelihood.posterior
  lppd <- sum(log(pLikelihood))
  WAIC <- -2 * (lppd - 2 * (loglikelihood.posterior - pLoglikelihood))
  
  
  
  #construct credible sets
  
  left <- floor(alpha*effsamp/2)
  
  right <- ceiling((1-alpha/2)*effsamp)
  
  
  
  BetaSort <- apply(betaout, 1, sort, decreasing = F)
  
  left.points <- BetaSort[left, ]
  
  right.points <- BetaSort[right, ]
  
  
  
  result=list("yHat" = pResponse, "BetaHat"= pMean, "LeftCI" = left.points,
              
              "RightCI" = right.points, "BetaMedian"= pMedian, "LambdaHat" = pLambda,
              
              "Sigma2Hat"=pSigma,"TauHat"=pTau, "BetaSamples"=betaout,
              
              "TauSamples" = tauout, "Sigma2Samples" = sigmaSqout, "LikelihoodSamples" = likelihoodout,
              
              "DIC" = DIC, "WAIC" = WAIC)
  
  return(result)
  
}