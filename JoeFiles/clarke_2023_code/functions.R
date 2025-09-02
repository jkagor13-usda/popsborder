##########################################################################
# Functions to estimate statistics introduced in
# Sections 3.1 - 3.3 of journal paper
##########################################################################

library(dplyr)
library(actuar)
library(extraDistr)
library(sqldf)
library(kSamples)
library(splus2R)

fp <- function(p, alpha, beta, Nbar, ty, b) {
  fp =p^(alpha - 1)*(1 - (1 - p)^Nbar)^ty*(1 - p)^(beta - 1 + Nbar*(b - ty))
  return(fp)
}

pf <- function(p, alpha, beta, Nbar, ty, b) {
  #
  #marginal = integrate(function(x) {fp(x,alpha, beta, Nbar, ty, b)}, 0,1)
  pf = fp(p, alpha, beta, Nbar, ty, b)#/marginal$value
  return(pf)
}

Tx_given_ty_NC <-function(N,Nbar,b,ty,l,alpha,beta){
  val =0
  if (l >= ty){
    tmpVal =sapply(0:ty, function(r) (-1)^r *exp(lbeta(l+alpha,N+beta-l)+lchoose(ty,r)+lchoose(N-(b-ty+r)*Nbar,l)))
    tmpVal = sum(tmpVal)
    val = tmpVal
  }else{
    val =0
  }

  #print(cat(val,l))
  return(val)
}


Tx_tx_given_ty_NC <-function(N,Nbar,b,ty,l,alpha,beta){
  val =0
  tmpVal =sapply(0:ty, function(r) (-1)^r *exp(lbeta(l+alpha,N+Nbar*r-Nbar*ty+beta-l)+lchoose(ty,r)+lchoose(N-b*Nbar,l)))
  tmpVal = sum(tmpVal)
  val = tmpVal
  return(val)
}


# clustered functions
phi =function(p,theta,Nbar){
  phi =  1 - exp(lgamma(theta*(1/p - 1) + Nbar)+lgamma(theta/p)-(lgamma(theta/p + Nbar)+lgamma(theta/p - theta)))
# 1 - gamma(theta*(1/p - 1) + Nbar)*gamma(theta/p)/(gamma(theta/p + Nbar)*gamma(theta/p - theta))
  return(phi)
}

fpc = function(p, alpha, beta,theta, Nbar, ty, b){
  fpc =p^(alpha - 1)*(1 - p)^(beta - 1)*phi(theta, p, Nbar)^ty*(1 - phi(theta, p, Nbar))^(b - ty)
  return(fpc)
}

pfc <- function(p, alpha, beta, theta, Nbar, ty, b) {
  #
  #marginal = integrate(function(x) {fpc(x,alpha, beta, theta,Nbar, ty, b)}, 0,1)
  pfc = fpc(p, alpha, beta, theta, Nbar, ty, b)#/marginal$value
  return(pfc)
}

# zero truncated beta binomial distribution pdf
ztbb <- function(k,Nbar,alpha,beta) {
  #
  if (k>0){
    ztbb = dbbinom(k, Nbar, alpha = alpha, beta = beta, log = FALSE)/
      (1-dbbinom(0, Nbar, alpha = alpha, beta = beta, log = FALSE))
  }else{
    ztbb = 0
  }
  return(ztbb)
}

# simple sampling algorithm
sample_fun <- function(n,support, probs) sample(
  support, # the support of pdf
  n, # n draws
  TRUE,  # with replacement
  probs # using these probabilities
)
