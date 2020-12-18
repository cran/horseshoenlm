#' Horseshoe shrinkage prior in Bayesian Logistic regression
#'
#'
#' This function employs the algorithm provided by Makalic and Schmidt (2016) for 
#' binary logistic model to fit Bayesian logistic regression. The observations are updated 
#' according to the Polya-Gamma data augmentation of approach of Polson, Scott, and Windle (2014).
#' 
#'
#' The model is:
#' \eqn{z_i} is response either 1 or 0,
#' \eqn{\log \Pr(z_i = 1) = logit^{-1}(X\beta)}.
#' 
#' 
#'
#' @references Stephanie van der Pas, James Scott, Antik Chakraborty and Anirban Bhattacharya (2016). horseshoe:
#'             Implementation of the Horseshoe Prior. R package version 0.1.0.
#'             https://CRAN.R-project.org/package=horseshoe
#'             
#'             Enes Makalic and Daniel Schmidt (2016). High-Dimensional Bayesian Regularised Regression with the
#'             BayesReg Package arXiv:1611.06649
#'             
#'             Polson, N.G., Scott, J.G. and Windle, J. (2014) The Bayesian Bridge.
#'             Journal of Royal Statistical Society, B, 76(4), 713-733.
#'
#'             
#'
#'
#'@param z Response, a \eqn{n*1} vector of 1 or 0.
#'@param X Matrix of covariates, dimension \eqn{n*p}.
#'@param method.tau Method for handling \eqn{\tau}. Select "truncatedCauchy" for full
#' Bayes with the Cauchy prior truncated to [1/p, 1], "halfCauchy" for full Bayes with
#' the half-Cauchy prior, or "fixed" to use a fixed value (an empirical Bayes estimate,
#' for example).
#'@param tau Use this argument to pass the (estimated) value of \eqn{\tau} in case "fixed"
#' is selected for method.tau. Not necessary when method.tau is equal to "halfCauchy" or
#' "truncatedCauchy". The default (tau = 1) is not suitable for most purposes and should be replaced.
#'@param burn Number of burn-in MCMC samples. Default is 1000.
#'@param nmc Number of posterior draws to be saved. Default is 5000.
#'@param thin Thinning parameter of the chain. Default is 1 (no thinning).
#'@param alpha Level for the credible intervals. For example, alpha = 0.05 results in
#'95\% credible intervals.
#'@param Xtest test design matrix.
#'
#'
#'@return 
#' \item{ProbHat}{Predictive probability}
#' \item{BetaHat}{Posterior mean of Beta, a \eqn{p} by 1 vector}
#' \item{LeftCI}{The left bounds of the credible intervals}
#' \item{RightCI}{The right bounds of the credible intervals}
#' \item{BetaMedian}{Posterior median of Beta, a \eqn{p} by 1 vector}
#' \item{LambdaHat}{Posterior samples of \eqn{\lambda}, a \eqn{p*1} vector}
#' \item{TauHat}{Posterior mean of global scale parameter tau, a positive scalar}
#' \item{BetaSamples}{Posterior samples of \eqn{\beta}}
#' \item{TauSamples}{Posterior samples of \eqn{\tau}}
#' \item{LikelihoodSamples}{Posterior samples of likelihood}
#' \item{DIC}{Devainace Information Criterion of the fitted model}
#' \item{WAIC}{Widely Applicable Information Criterion}
#'
#'
#'
#' @importFrom stats dnorm pnorm rbinom rnorm var dbinom
#' 
#' 
#' @examples
#'
#' burnin <- 100
#' nmc    <- 500
#' thin <- 1
#' 
#'  
#' p <- 100  # number of predictors
#' ntrain <- 250  # training size
#' ntest  <- 100   # test size
#' n <- ntest + ntrain  # sample size
#' q <- 10   # number of true predictos
#' 
#' beta.t <- c(sample(x = c(1, -1), size = q, replace = TRUE), rep(0, p - q))  
#' x <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(p))    
#' 
#' zmean <- x %*% beta.t
#' z <- rbinom(n, size = 1, prob = boot::inv.logit(zmean))
#' X <- scale(as.matrix(x))  # standarization
#' 
#' 
#' # Training set
#' ztrain <- z[1:ntrain]
#' Xtrain  <- X[1:ntrain, ]
#' 
#' # Test set
#' ztest <- z[(ntrain + 1):n]
#' Xtest  <- X[(ntrain + 1):n, ]
#' 
#' posterior.fit <- logiths(z = ztrain, X = Xtrain, method.tau = "halfCauchy",
#'                          burn = burnin, nmc = nmc, thin = 1,
#'                          Xtest = Xtest)
#'                              
#' posterior.fit$BetaHat
#'
#'
#' # Posterior processing to recover the true predictors
#' cluster <- kmeans(abs(posterior.fit$BetaHat), centers = 2)$cluster
#' cluster1 <- which(cluster == 1)
#' cluster2 <- which(cluster == 2)
#' min.cluster <- ifelse(length(cluster1) < length(cluster2), 1, 2)
#' which(cluster == min.cluster)  # this matches with the true variables
#'
#'
#'
#' @export





logiths <- function(z, X, method.tau = c("fixed", "truncatedCauchy","halfCauchy"), tau = 1,
                    burn = 1000, nmc = 5000, thin = 1, alpha = 0.05,
                    Xtest = NULL)
{
  
  method.tau = match.arg(method.tau)
  
  
  
  niter <- burn+nmc
  effsamp=(niter -burn)/thin
  
  
  n=nrow(X)  # sample size
  p <- ncol(X)  # number of variables
  if(is.null(Xtest))
  {
    Xtest <- X
    ntest <- n
  } else {
    ntest <- nrow(Xtest)
  }
  
  
  y <- z - 0.5        # for coding convenience
  
  
  
  ## parameters ##
  
  beta.b <- rep(0, p)
  
  lambda <- rep(1, p)
  
  
  ## output ##
  
  betaout          <- matrix(0, p, effsamp)
  
  lambdaout        <- matrix(0, p, effsamp)
  
  tauout           <- rep(0, effsamp)
  
  likelihoodout    <- matrix(0, n, effsamp)
  
  loglikelihoodout <- rep(0, effsamp)
  
  probout          <- matrix(0, ntest, effsamp)
  
  
  ## matrices ##
  
  I_n=diag(n)
  
  l0=rep(0,p)
  
  l1=rep(1,n)
  
  l2=rep(1,p)
  
  
  
  ## start Gibb's sampling ##
  
  message("Markov chain monte carlo is running")
  
  
  for(i in 1:niter)
  {
    
    ################################
    ######### binary part ##########
    ################################
    
    
    
    
    omega2 <- as.matrix(1/pgdraw::pgdraw(1, X %*% beta.b), n, 1)
    
    
    ## update beta ##
    if((p > n) || (p == n))
    {
      bs  = bayesreg.sample_beta(X, z = omega2 * y, mvnrue = FALSE, b0 = 0, sigma2 = 1, tau2 = tau^2, 
                                 lambda2 = lambda^2, omega2  = omega2, XtX = NA)
      beta.b <- bs$x
      
    } else {
      
      bs  = bayesreg.sample_beta(X, z = omega2 * y, mvnrue = TRUE, b0 = 0, sigma2 = 1, tau2 = tau^2, 
                                 lambda2 = lambda^2, omega2  = omega2, XtX = NA)
      beta.b <- bs$x
    }
    


    beta <- c(beta.b)  # all \beta's together
    Beta <- matrix(beta, ncol = p, byrow = TRUE)
    
    
    ## update lambda_j's in a block using slice sampling ##
    eta = 1/(lambda^2)
    upsi = stats::runif(p,0,1/(1+eta))
    tempps = apply(Beta^2, 2, sum)/(2*tau^2)
    ub = (1-upsi)/upsi
    # now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
    Fub = stats::pgamma(ub, (1 + 1)/2, scale = 1/tempps)
    Fub[Fub < (1e-4)] = 1e-4;  # for numerical stability
    up = stats::runif(p,0,Fub)
    # eta = -log(1-up)/tempps
    eta <- stats::qgamma(up, (1 + 1)/2, scale=1/tempps)
    lambda = 1/sqrt(eta);

    ## update tau ##
    ## Only if prior on tau is used
    if(method.tau == "halfCauchy"){
      tempt <- sum(apply(Beta^2, 2, sum)/(2*lambda^2))
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
      tempt <- sum(apply(Beta^2, 2, sum)/(2*lambda^2))
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
    
    
    probability   <- boot::inv.logit(Xtest %*% beta.b)
    likelihood    <- dbinom(z, size = 1, prob = boot::inv.logit(X %*% beta.b))
    loglikelihood <- sum(log(likelihood))
    
    
    if (i%%500 == 0)
    {
      message("iteration = ", i)
    }
    
    
    
    
    if(i > burn && i%%thin== 0)
    {
      betaout[ ,(i-burn)/thin]          <- Beta
      lambdaout[ ,(i-burn)/thin]        <- lambda
      tauout[(i - burn)/thin]           <- tau
      probout[, (i - burn)/thin]        <- probability
      likelihoodout[ ,(i - burn)/thin]  <- likelihood
      loglikelihoodout[(i - burn)/thin] <- loglikelihood
      }
  }
  
  
  
  pMean          <- apply(betaout,1,mean)
  pMedian        <- apply(betaout,1,stats::median)
  pLambda        <- apply(lambdaout, 1, mean)
  pTau           <- mean(tauout)
  pProb          <- apply(probout, 1, mean)
  pLikelihood    <- apply(likelihoodout, 1, mean)
  pLoglikelihood <- mean(loglikelihoodout)
  
  loglikelihood.posterior <- sum(dbinom(z, size = 1, prob = boot::inv.logit(X %*% pMean), log = TRUE))
  
  DIC  <- -4 * pLoglikelihood + 2 * loglikelihood.posterior
  lppd <- sum(log(pLikelihood))
  WAIC <- -2 * (lppd - 2 * (loglikelihood.posterior - pLoglikelihood))
  
  #construct credible sets
  left  <- floor(alpha*effsamp/2)
  right <- ceiling((1-alpha/2)*effsamp)
  
 
  
  betaSort     <- apply(betaout, 1, sort, decreasing = F)
  left.points  <- betaSort[left, ]
  right.points <- betaSort[right, ]
  
  
  result=list("ProbHat" = pProb, "BetaHat"= pMean, 
              "LeftCI" = left.points, "RightCI" = right.points,
              "BetaMedian" = pMedian,
              "LambdaHat" = pLambda, "TauHat"=pTau, "BetaSamples" = betaout,
              "TauSamples" = tauout, "LikelihoodSamples" = likelihoodout,
              "DIC" = DIC, "WAIC" = WAIC)
  return(result)
}




# ============================================================================================================================
# Sample the regression coefficients
bayesreg.sample_beta <- function(X, z, mvnrue, b0, sigma2, tau2, lambda2, omega2, XtX)
{
  alpha  = (z - b0)
  Lambda = sigma2 * tau2 * lambda2
  sigma  = sqrt(sigma2)
  
  # Use Rue's algorithm
  if (mvnrue)
  {
    # If XtX is not precomputed
    if (any(is.na(XtX)))
    {
      omega = sqrt(omega2)
      X0    = apply(X,2,function(x)(x/omega))
      bs    = bayesreg.fastmvg2_rue(X0/sigma, alpha/sigma/omega, Lambda)
    }
    
    # XtX is precomputed (Gaussian only)
    else {
      bs    = bayesreg.fastmvg2_rue(X/sigma, alpha/sigma, Lambda, XtX/sigma2)
    }
  }
  
  # Else use Bhat. algorithm
  else
  {
    omega = sqrt(omega2)
    X0    = apply(X,2,function(x)(x/omega))
    bs    = bayesreg.fastmvg_bhat(X0/sigma, alpha/sigma/omega, Lambda)
  }
  
  return(bs)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Rue's algorithm
bayesreg.fastmvg2_rue <- function(Phi, alpha, d, PtP = NA)
{
  Phi   = as.matrix(Phi)
  alpha = as.matrix(alpha)
  r     = list()
  
  # If PtP not precomputed
  if (any(is.na(PtP)))
  {
    PtP = t(Phi) %*% Phi
  }
  
  p     = ncol(Phi)
  if (length(d) > 1)
  {
    Dinv  = diag(as.vector(1/d))
  }
  else
  {
    Dinv   = 1/d
  }
  L     = t(chol(PtP + Dinv))
  v     = forwardsolve(L, t(Phi) %*% alpha)
  r$m   = backsolve(t(L), v)
  w     = backsolve(t(L), rnorm(p,0,1))
  
  r$x   = r$m + w
  return(r)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Bhat. algorithm
bayesreg.fastmvg_bhat <- function(Phi, alpha, d)
{
  d     = as.matrix(d)
  p     = ncol(Phi)
  n     = nrow(Phi)
  r     = list()
  
  u     = as.matrix(rnorm(p,0,1)) * sqrt(d)
  delta = as.matrix(rnorm(n,0,1))
  v     = Phi %*% u + delta
  Dpt   = (apply(Phi, 1, function(x)(x*d)))
  W     = Phi %*% Dpt + diag(1,n)
  w     = solve(W,(alpha-v))
  
  r$x   = u + Dpt %*% w
  r$m   = r$x
  
  return(r)
}
