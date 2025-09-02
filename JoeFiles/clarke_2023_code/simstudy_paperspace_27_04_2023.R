#########################################################################
# This script performs the simulations for the Supp Materials
# 27/4/2023
# to run from paperspace linux:
#  nohup Rscript simstudy_paperspace_27_04_2023.R >& stuff.txt &
#########################################################################

#########################################################################
# Define functions for calculating densities, likelihoods and other
# quantities
#########################################################################

library(parallel)

dty.unclust.oneval <- function(ty,b,Nbar,alpha,beta,R=1000){
  # returns P[ty] (where ty must be a single value)
  #  where ty|p ~ bin(b,1-(1-p)^Nbar) and p~Beta(alpha,beta)
  #if(ty==0) return(exp(lbeta(alpha,beta+Nbar*b)-lbeta(alpha,beta)))
  if(ty==0) return(beta.ratio(alpha=alpha,beta=beta,bplus=Nbar*b))
  if(ty==1) return(b*beta.diff.ratio(alpha=alpha,beta=beta,
                                     add1=Nbar*(b-1),add2=Nbar*b))
  if(ty>1){
    pvals <- qbeta(c(1:R)/(R+1),alpha,beta)
    return(
      choose(b,ty)*mean((1-(1-pvals)^Nbar)^ty*(1-pvals)^(Nbar*(b-ty)))
    )
  }
}

dty.unclust <- function(ty,b,Nbar,alpha,beta,R=1000){
  # returns a vector P[ty] (for each value in ty)
  #  where ty|p ~ bin(b,1-(1-p)^Nbar) and p~Beta(alpha,beta)
  # ty, b and Nbar may be vectors
  if(length(b)==1) b <- rep(b,length(ty))
  if(length(Nbar)==1) Nbar <- rep(Nbar,length(ty))
  out <- rep(NA,length(ty))
  for(j in c(1:length(ty))) out[j]<-dty.unclust.oneval(ty[j],b[j],Nbar[j],alpha,beta,R=R)
  out
}

dty.clust <- function(ty,b,Nbar,alpha,beta,theta,R=1000){
  # returns a vector P[ty] (for each value in ty)
  #  where tyi|pi ~ bin(b,1-(1-pi)^Nbar), pi|p~Beta(theta,theta*(1/p-1)) and
  #     p~Beta(alpha,beta)
  # returns P[ty] for each value in ty
  pvals <- qbeta(c(1:R)/(R+1),alpha,beta)
  log.one.minus.phi.vals <- log1p(-pvals)
  for(i in c(1:(Nbar-1))){
    # log.one.minus.phi.vals <- log.one.minus.phi.vals + log1p(-theta/(theta/pvals+i)) # old parameterisation
    log.one.minus.phi.vals <- log.one.minus.phi.vals + log1p(-pvals*theta/(theta+i))
  }
  log.phi.vals <- Rmpfr::log1mexp(-log.one.minus.phi.vals)
  prob.ty <- rep(NA,length(ty))
  phi.vals <- exp(log.phi.vals)
  one.minus.phi.vals <- exp(log.one.minus.phi.vals)
  for(k in c(1:length(ty))){
    prob.ty[k] <- mean(choose(b,ty[k])*phi.vals^ty[k]*one.minus.phi.vals^(b-ty[k]))
  }
  return(prob.ty)
}

dty <- function(ty,b,Nbar,alpha,beta,theta=Inf,R=1000){
  if(theta==Inf) return(dty.unclust(ty=ty,b=b,Nbar=Nbar,
                                    alpha=alpha,beta=beta,R=R))
  if(theta!=Inf) return(dty.clust(ty=ty,b=b,Nbar=Nbar,theta=theta,
                                  alpha=alpha,beta=beta,R=R))
}

negll <- function(par,ty,b,Nbar,freq,theta=Inf,R=1000,fix.logitmu,fix.logshape,fix.alpha){
  # returns the negative of the log-likelihood of a vector of values of ty
  #  where ty|p ~ bin(b,1-(1-p)^Nbar) and p~Beta(alpha,beta)
  # ty, b and Nbar may be vectors
  # par=(log(alpha),log(beta))
  # freq is the number of times each value of ty appears in dataset
  #  (if freq left blank, it is assumed that each value in ty represents a
  #    single observation)
  # (theta=Inf represents no clustering within groups,
  #   theta=0 represents perfect clustering within groups)
  # (note that alpha,beta control between-population variation i.e.
  #   clustering within consignments)
  #
  # par can be a matrix; in this case, each column should be a parameter vector
  #
  # if fix.logitmu is supplied, then mu is fixed at 1-1/(1+exp(logitmu)), and
  #   par is interpreted as log(alpha+beta)
  # if fix.logshape is supplied, then alpha+beta is fixed at exp(fix.logshape),
  #   and par is interpreted as logit(mu)
  # if fix.alpha is supplied, then alpha is fixed at this value,
  #   and par is interpreted as log(beta)
  #
  if(missing(freq)) freq <- rep(1,length(ty))
  alpha <- exp(par[1])
  beta <- exp(par[2])
  if(!missing(fix.logitmu)){
    alpha <- exp(par)*(1-1/(1+exp(fix.logitmu)))
    beta <- exp(par)*1/(1+exp(fix.logitmu))
  }
  if(!missing(fix.logshape)){
    alpha <- exp(fix.logshape)*(1-1/(1+exp(par)))
    beta <- exp(fix.logshape)*1/(1+exp(par))
  }
  if(!missing(fix.alpha)){
    alpha <- fix.alpha
    beta <- exp(par)
  }
  -sum(freq*log(dty(ty=ty,b=b,Nbar=Nbar,alpha=alpha,beta=beta,theta=theta)))
}

summary.Tx.clustered <- function(alpha,beta,theta,b,Nbar,B,R=1e5){
  # returns summary information on the prior distribution of Tx and leakage
  # (i.e. not conditional on ty)
  # population assumed to be rejected if ty>=1
  sim.p <- rbeta(R,alpha,beta)
  sim.p.group <- sim.Xi <- matrix(NA,nrow=R,ncol=B)
  for(r in c(1:R)){
    if((theta>0)&(theta<Inf)) sim.p.group[r,] <- rbeta(B,theta,theta*(1/sim.p[r]-1))
    if(theta==0) sim.p.group[r,] <- 1*(runif(B)<=sim.p[r])
    if(theta==Inf) sim.p.group[r,] <- sim.p[r]
    sim.Xi[r,] <- rbinom(B,Nbar,sim.p.group[r,])
  }
  tx.samp <- apply(sim.Xi[,1:b],1,sum)
  tx.nonsamp <- apply(sim.Xi[,(b+1):B],1,sum)
  Tx <- apply(sim.Xi,1,sum)
  ty.samp <- apply(sim.Xi[,1:b]>=1,1,sum)
  Tx.post <- Tx[ty.samp==0]
  L <- Tx*(tx.samp==0)
  c(mean(Tx),mean(tx.samp>0),mean(L),mean(L>0),mean(L[tx.samp==0]),mean(L[tx.samp==0]>0))
}

beta.ratio <- function(alpha,beta,bplus,mu,tot,log=FALSE){
  # calculates beta(alpha,beta+bplus)/beta(alpha,beta)
  # where bplus is a positive integer
  # works reasonably well even if alpha<<beta or alpha<<1
  #  if mu and tot are supplied rather than alpha and beta,
  #  then alpha=mu*tot and beta=mu*tot, and calculation still works if tot=0
  if(!missing(alpha)){
    tot<-alpha+beta
    mu<-alpha/(alpha+beta)
  }
  if(length(mu)==1) mu<-rep(mu,length(tot)) else{
    if(length(tot)==1) tot<-rep(tot,length(mu))
  }
  log.out<-rep(NA,length(mu))
  for(i in c(1:length(mu))){
    log.out[i]<-log1p(-mu[i])+sum(log1p(-mu[i]/(1+c(1:(bplus-1))/tot[i])))
  }
  if(log) return(log.out) else return(exp(log.out))
}

beta.diff.ratio <- function(alpha,beta,add1,add2,mu,tot,log=FALSE){
  # calculates (beta(alpha,beta+add1)-beta(alpha,beta+add2))/beta(alpha,beta)
  # where add1<add2 is a positive integer
  # works reasonably well even if alpha<<beta or alpha<<1
  #  if mu and tot are supplied rather than alpha and beta,
  #  then alpha=mu*tot and beta=mu*tot, and calculation still works if tot=0
  if(add1>=add2) stop("add1 must be strictly less than add2")
  if(!missing(alpha)){
    tot<-alpha+beta
    mu<-alpha/(alpha+beta)
  }
  if(length(mu)==1) mu<-rep(mu,length(tot)) else{
    if(length(tot)==1) tot<-rep(tot,length(mu))
  }
  log.out<-rep(NA,length(mu))
  for(i in c(1:length(mu))){
    log.out[i] <- log1p(-mu[i])+sum(log1p(-mu[i]/(1+c(1:(add1-1))/tot[i]))) + 
      Rmpfr::log1mexp(-sum(log1p(-mu[i]/(1+c(add1:(add2-1))/tot[i]))))
  }
  if(log) return(log.out) else return(exp(log.out))
}

# MLE

BB.group.model <- function(ty,b,B,Nbar,freq,theta=Inf,R=1000,startval,SE=FALSE){
  if(missing(startval)) startval<-c(0,0)
  if(all(ty==0)) return(list(optim.results=list(value=0),alpha=0,beta=0,mu=0,rho=0,D=1,E.leak=0,prob.leak=0,
                             se.alpha=0,se.beta=0,se.mu=0,se.rho=0,se.D=0,se.par=c(0,0),
                             log.prob.leak=-Inf,pty0=1))
  o <- optim(startval,negll,ty=ty,freq=freq,b=b,Nbar=Nbar,R=R,theta=theta,
             hessian=SE)
  N <- Nbar*B
  n <- Nbar*b
  alpha <- exp(o$par[1])
  beta <- exp(o$par[2])
  mu <- alpha/(alpha+beta)
  rho <- 1/(alpha+beta+1)
  D <- 1+(Nbar-1)*rho
  
  pty0 <- exp(lbeta(alpha,beta+n)-lbeta(alpha,beta))
  #E.leak <- (N-n)*alpha/(alpha+beta+n)*beta(alpha,beta+n)/beta(alpha,beta)
  E.leak <- (N-n)*alpha/(alpha+beta+n)*beta.ratio(alpha=alpha,beta=beta,bplus=n)
  #prob.leak <- beta(alpha,beta+n)/beta(alpha,beta)-
  #  beta(alpha,beta+N)/beta(alpha,beta)
  log.prob.leak <- sum(log1p(-alpha/(alpha+beta+c(0:(n-1))))) +
    Rmpfr::log1mexp(-sum(log1p(-alpha/(alpha+beta+c(n:(N-1))))))
  prob.leak <- exp(log.prob.leak)
  
  if(SE){
    par.vcov <- solve(o$hessian)
    se.par <- sqrt(diag(par.vcov))
    se.alpha <- alpha*se.par[1]
    se.beta <- beta*se.par[2]
    dmu.dpar <- c(1,-1)*alpha*beta/(alpha+beta)^2
    se.mu <- sqrt(as.numeric(t(dmu.dpar)%*%par.vcov%*%dmu.dpar))
    drho.dpar <- -c(alpha,beta)/(alpha+beta+1)^2
    se.rho <- sqrt(as.numeric(t(drho.dpar)%*%par.vcov%*%drho.dpar))
    se.D <- (Nbar-1)*se.rho
    return(list(optim.results=o,alpha=alpha,beta=beta,mu=mu,rho=rho,D=D,
                se.alpha=se.alpha,se.beta=se.beta,se.mu=se.mu,se.rho=se.rho,
                se.D=se.D,se.par=se.par,
                E.leak=E.leak,prob.leak=prob.leak,log.prob.leak=log.prob.leak,
                pty0=pty0))
  }
  return(list(optim.results=o,alpha=alpha,beta=beta,mu=mu,rho=rho,D=D,
              E.leak=E.leak,prob.leak=prob.leak,log.prob.leak=log.prob.leak,
              pty0=pty0))
}

# test:
BB.group.model(ty=c(0,4,4,7,21),freq=c(68,1,1,1,1),b=94,B=1000,Nbar=100,
               theta=Inf,SE=TRUE)

BB.group.model(ty=c(0,4,4,7,21),freq=c(68,1,1,1,1),b=94,B=1000,Nbar=100,
               theta=0,SE=TRUE)


#########################################################################
# Function to perform simulation for a given set of parameters
#########################################################################

simfn <- function(k,parvalues,R){
  # set seed, set up parameters
  set.seed(parvalues$seed[k])
  Nbar=parvalues$Nbar[k]
  b=parvalues$b[k] ; B=parvalues$B[k]
  alpha=parvalues$alpha[k] ; mu=parvalues$mu[k]
  beta=alpha*(1-mu)/mu
  rho=1/(alpha+beta+1)
  D=1+(Nbar-1)*rho
  theta=parvalues$theta[k]
  num.consignments=parvalues$num.consignments[k]
  N=Nbar*B ; n=Nbar*b
  if(theta==Inf){
    pty0 <- exp(lbeta(alpha,beta+n)-lbeta(alpha,beta))
    E.leak <- (N-n)*alpha/(alpha+beta+n)*beta(alpha,beta+n)/beta(alpha,beta)
    log.prob.leak <- sum(log1p(-alpha/(alpha+beta+c(0:(n-1))))) +
      Rmpfr::log1mexp(-sum(log1p(-alpha/(alpha+beta+c(n:(N-1))))))
    prob.leak <- exp(log.prob.leak)
  }
  if(theta==0){
    pty0 <- exp(lbeta(alpha,beta+b)-lbeta(alpha,beta))
    E.leak <- (N-n)*alpha/(alpha+beta+b)*beta(alpha,beta+b)/beta(alpha,beta)
    log.prob.leak <- sum(log1p(-alpha/(alpha+beta+c(0:(b-1))))) +
      Rmpfr::log1mexp(-sum(log1p(-alpha/(alpha+beta+c(n:(B-1))))))
    prob.leak <- exp(log.prob.leak)
  }
  if((theta>0)&(theta<Inf)){
    p.i.vals<-qbeta(c(1:1000)/1001,alpha,beta)
    ratio.vals <- beta.ratio(alpha=theta,beta=theta*(1/p.i.vals-1),bplus=Nbar)
    pty0<-mean(ratio.vals^b)
    E.leak<-mean((N-n)*p.i.vals*ratio.vals^b)
    prob.leak<-mean(ratio.vals^b*(1-ratio.vals^(B-b)))
  }
  # object to save results
  simres <- data.frame(a=rep(NA,R),profile.deviance.mu=NA,
                       profile.deviance.shape=NA,profile.deviance.alpha=NA)
  # loop
  for(r in c(1:R)){
    all.consignments<-data.frame(i=rep(c(1:num.consignments),each=B),
                                 j=rep(c(1:B),times=num.consignments),
                                 p.i=NA,p.ij=NA)
    all.consignments$p.i <- rep(rbeta(num.consignments,alpha,beta),
                                each=B)
    if((theta<Inf)&(theta>0)) all.consignments$p.ij <-
      rbeta(B*num.consignments,theta*all.consignments$p.i,theta*(1-all.consignments$p.i))
    if(theta==Inf) all.consignments$p.ij <- all.consignments$p.i
    if(theta==0) all.consignments$p.ij <- rbinom(B*num.consignments,1,
                                                 all.consignments$p.i)
    all.consignments$X.i<-rbinom(B*num.consignments,Nbar,all.consignments$p.ij)
    ty<-with(all.consignments[all.consignments$j<=b,],tapply(X.i>=1,i,FUN=sum))
    Tx <- with(all.consignments,tapply(X.i,i,FUN=sum))
    tab.ty <- table(ty)
    modfit <- BB.group.model(ty=as.numeric(names(tab.ty)),
                               freq=as.numeric(tab.ty),b=b,B=B,Nbar=Nbar,
                               theta=Inf,SE=TRUE)
    if(sum(ty)==0) simres[r,"profile.deviance.shape"] <- 0 else
    simres[r,"profile.deviance.shape"] <- 2*(optimise(f=negll,
                                                   interval=c(-20,-1),
                                                   fix.logshape=log(alpha+beta),
                                                   ty=as.numeric(names(tab.ty)),
                                                   freq=as.numeric(tab.ty),b=b,Nbar=Nbar,
                                                   R=R,theta=Inf)$objective - modfit$optim.results$value)
    simres[r,"profile.deviance.mu"] <- 2*(optimise(f=negll,
                                                   interval=c(-3,15),
                                                   fix.logitmu=pmax(-20,make.link("logit")$linkfun(mu)),
                                                   ty=as.numeric(names(tab.ty)),
                                                   freq=as.numeric(tab.ty),b=b,Nbar=Nbar,
                                                   R=R,theta=Inf)$objective - modfit$optim.results$value)
    if(sum(ty)==0) simres[r,"profile.deviance.alpha"] <- 0 else
    simres[r,"profile.deviance.alpha"] <- 2*(optimise(f=negll,
                                                   interval=c(-3,15),
                                                   fix.alpha=alpha,
                                                   ty=as.numeric(names(tab.ty)),
                                                   freq=as.numeric(tab.ty),b=b,Nbar=Nbar,
                                                   R=R,theta=Inf)$objective - modfit$optim.results$value)
    simres[r,c("a","b","mu","rho","D")] <- c(modfit$alpha,modfit$beta,modfit$mu,
                                         modfit$rho,modfit$D)
    simres[r,c("se.a","se.b","se.mu","se.rho","se.D")] <-
      c(modfit$se.alpha,modfit$se.beta,modfit$se.mu,modfit$se.rho,modfit$se.D)
    simres[r,c("se.log.a","se.log.b")] <- modfit$se.par
    simres[r,c("prob.leak","pty0","E.leak")] <- c(modfit$prob.leak,modfit$pty0,modfit$E.leak)
    simres[r,"P.leak"] <- mean((ty==0)*(Tx>0))
    simres[r,"mean.ty"] <- mean(ty)
    simres[r,"shipments.positive"] <- sum(ty>0)
    simres[r,"max.ty"] <- max(ty)
  }
  simres$se.a[is.na(simres$se.a)] <- 0
  simres$se.b[is.na(simres$se.b)] <- 0
  simres$se.mu[is.na(simres$se.mu)] <- 0
  simres$se.D[is.na(simres$se.D)] <- 0
  simres$se.rho[is.na(simres$se.rho)] <- 0
  simres$se.log.a[is.na(simres$se.log.a)] <- 0
  simres$se.log.b[is.na(simres$se.log.b)] <- 0
  simres$k <- k
  simres$scenario <- parvalues$scenario[k]
  cat(paste0("finished simulation ",k," \n"),file="update.txt",append=TRUE)
  return(simres)
}

#########################################################################
# Find value of theta corresponding to lambda=50 when Nbar=100
#########################################################################

fn.lambda <- function(logtheta){
  theta <- exp(logtheta)
  ( 100 / sum(theta/(theta+c(0:99))) - 50 )^2
}
(theta.mid.opt <- optimize(f=fn.lambda,interval=c(-10,10)))
exp(theta.mid.opt$minimum) # 0.2046949



#########################################################################
# Actually perform the simulation over all combinations of
# parameter settings
# Note that the parallelisation may not work on all operating systems.
# The simulation is time consuming. Progress is output to a file
# called "update.txt" in the working directory.
#########################################################################

theta.mid <- 0.2046949
g.mid <- 2 # note that g=Nbar/lambda where lambda defined as in paper
g.inf <- 100
g0 <- 1
pargrid <- rbind(expand.grid(Nbar=c(100),b=100,B=1000,alpha=c(0.01,0.015,0.025,0.1),
                          num.consignments=c(100,400),
                          theta=c(0,theta.mid,Inf)) )
pargrid$g[pargrid$theta==0] <- g0
pargrid$g[pargrid$theta==Inf] <- g.inf
pargrid$g[pargrid$theta==theta.mid] <- g.mid
pargrid$mu <- 5e-3/pargrid$g

numscen <- nrow(pargrid)
pargrid$scenario <- c(1:nrow(pargrid))
pargrid <- rbind(pargrid,pargrid,pargrid,pargrid,pargrid)
set.seed(81749174)
five.seeds <- sample(10000:1e8,5,replace=FALSE)
pargrid$seed <- rep(five.seeds,each=numscen)

t1 <- proc.time()[3]
K <- nrow(pargrid)
cat("Starting simulation study \n",file="update.txt")
cl<-makeCluster(8,type="FORK")
simresults <- parLapply(cl,c(1:K),simfn,parvalues=pargrid,R=200)
t2 <- proc.time()[3]
t2-t1
save.image("all_cucurbit_simulation_results_27_04_2023.rdata") # saves all results


