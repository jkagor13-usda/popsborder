# This is the same as cucurbits_06_09_2022.R, but
# updated 6/4/2023 to fix an error in the output for Table 1 of the paper (old numbering)
#  where the estimated alpha was stated in place of beta in the model with theta=0
# Also modified for new format of Tables 1 and 2
# Further updated on 19/4/2023 to tabulate lambda instead of theta, to use an approximate
#   beta method rather than approximate gamma, and to combine Tables 1 and 2.

loglikfn <- function(par,ty,freq,theta,mu,alpha,b,R=1000,Nbar,deviance=FALSE,
                     subtract=0,B,mushape=FALSE,alpha.mu=FALSE,leakage=FALSE){
  # if theta, mu and alpha are all missing, then
  #    par=c(log(alpha),log(beta),log(theta))
  # Otherwise:
  #   if theta is missing, it is the exponential of the last element of par
  #   if alpha is missing, it is the exp of the first element of par
  #   beta is the first element of par if alpha supplied, 
  #       otherwise the second element
  #   except that if mu is supplied, beta=(1-mu)/mu*alpha
  missing.alpha <- missing(alpha)
  if(missing(theta)) theta <- exp(par[length(par)])
  if(missing(alpha)) alpha <- exp(par[1])
  if(missing(mu)&missing.alpha) beta <- exp(par[2])
  if(missing(mu)&!missing.alpha) beta <- exp(par[1])
  if(!missing(mu)) beta <- (1-mu)/mu*alpha
  if(mushape){
    mu <- 1-1/(1+exp(par[1]))
    shape <- exp(par[2])
    alpha <- shape*mu
    beta <- shape*(1-mu)
  }
  if(alpha.mu){
    alpha <- exp(par[1])
    mu <- 1 / (1+exp(par[2]))
    beta <- alpha*(1-mu)/mu
  }
  if(theta>0){
    pvals <- qbeta(c(1:R)/(R+1),alpha,beta)
    log.pvals <- log(pvals)
    if(theta==Inf){
      #log.one.minus.phi.vals <- Nbar*log1p(-pvals) # i.e. Nbar*log(1-pvals)
      log.one.minus.phi.vals <- Nbar*Rmpfr::log1mexp(-log.pvals) # i.e. Nbar*log(1-pvals)
    }
    if(theta<Inf){
      log.one.minus.phi.vals <- lbeta(theta*pvals,theta*(1-pvals)+Nbar) -
        lbeta(theta*pvals,theta*(1-pvals))
      log.one.minus.phi.vals <- pmin(log.one.minus.phi.vals,0)
    }
    log.phi.vals <- Rmpfr::log1mexp(-log.one.minus.phi.vals)
    out <- 0
    for(j in c(1:length(ty))){
      if(ty[j]!=0) log.terms <- ty[j]*log.phi.vals + 
          (b-ty[j])*log.one.minus.phi.vals
      if(ty[j]==0) log.terms <- b*log.one.minus.phi.vals
      K <- max(log.terms)-10
      out <- out + freq[j]*(log(mean(exp(log.terms-K)))+K)
    }
  }
  if(theta==0) out <- sum(freq*(lbeta(alpha+ty,beta+b-ty)-lbeta(alpha,beta)))
  if(leakage){
    if(theta==Inf){
      P.L <- exp(lbeta(alpha,beta+b*Nbar)-lbeta(alpha,beta))-
        exp(lbeta(alpha,beta+B*Nbar)-lbeta(alpha,beta))
      E.L <- (B-b)*Nbar*exp(lbeta(alpha+1,beta+b*Nbar)-lbeta(alpha,beta))
    }
    if(theta==0){
      P.L <- exp(lbeta(alpha,beta+b)-lbeta(alpha,beta))-
        exp(lbeta(alpha,beta+B)-lbeta(alpha,beta))
      E.L <- (B-b)*Nbar*exp(lbeta(alpha+1,beta+b)-lbeta(alpha,beta))
    }
    if((theta<Inf)&(theta>0)){
      P.L <- mean(exp(b*log.one.minus.phi.vals)-exp(B*log.one.minus.phi.vals))
      E.L <- (B-b)*Nbar*mean(exp(b*log.one.minus.phi.vals+log.pvals))
    }
  }
  if((!leakage)&deviance) return(-2*out-subtract)
  if((!leakage)&(!deviance)) return(out-subtract)
  if(leakage) return(c(deviance=-2*out,P.L=P.L,E.L=E.L))
}

########################################################################
# Calculate parameter estimates based on theta=0 and theta=Inf
########################################################################

o.theta.inf <- optim(c(0,0),loglikfn,ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,
                     theta=Inf,Nbar=100,deviance=FALSE,control=list(reltol=1e-12,fnscale=-1),
                     R=1e4,hessian=TRUE)
o.theta.inf$mu <- exp(o.theta.inf$par[1])/sum(exp(o.theta.inf$par))
o.theta.inf$alpha <- exp(o.theta.inf$par[1])
o.theta.inf$beta <- exp(o.theta.inf$par[2])
temp <- loglikfn(o.theta.inf$par,ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,B=800,
    theta=Inf,Nbar=100,leakage=TRUE,R=1e4)
o.theta.inf$P.L <- temp["P.L"]
o.theta.inf$E.L <- temp["E.L"]
exp(o.theta.inf$par)
exp(o.theta.inf$par[1])/sum(exp(o.theta.inf$par))

cucurbit.data <- data.frame(ty=c(rep(0,68),4,4,7,21))
library(aod)
(approx.BB.mod <- betabin(cbind(ty,94-ty)~1,~1,data=cucurbit.data))

approx.BB.pars <- approx.BB.mod@param
approx.mean <- 1-1/(1+exp(approx.BB.pars[1]))
approx.phi <- approx.BB.pars[2]
approx.shape1 <- approx.mean*(1/approx.phi-1)
approx.shape2 <- (1-approx.mean)*(1/approx.phi-1)
approx.alpha <- approx.shape1
approx.beta <- approx.shape2 * 100
approx.E.p <- approx.alpha/(approx.alpha+approx.beta)
approx.P.L <- (beta(approx.alpha,approx.beta+94*100)-beta(approx.alpha,approx.beta+800*100))/
  beta(approx.alpha,approx.beta)
approx.E.L <- (800-94)*100*beta(approx.alpha+1,approx.beta+94*100)/
  beta(approx.alpha,approx.beta)


o.theta0 <- optim(c(0,0),loglikfn,ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,R=1e4,
                  theta=0,Nbar=100,deviance=TRUE,control=list(reltol=1e-12))
o.theta0$mu <- exp(o.theta0$par[1])/sum(exp(o.theta0$par))
temp <- loglikfn(o.theta0$par,ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,B=800,
                 theta=0,Nbar=100,leakage=TRUE,R=1e4)
o.theta0$P.L <- temp["P.L"]
o.theta0$E.L <- temp["E.L"]

########################################################################
# Produce Results for Table 1 for paper
########################################################################

parvals.theta.inf <- expand.grid(logalpha=o.theta.inf$par[1]+seq(-4.6,4.6,0.02),
                                 logitmu=o.theta.inf$par[1]-o.theta.inf$par[2]+seq(-4.6,4.6,0.02))
for(k in c(1:nrow(parvals.theta.inf))){
  if(k%%1e3==0) cat("finished row ",k," of ",nrow(parvals.theta.inf)," \n")
  A <- exp(parvals.theta.inf$logalpha[k])
  mu <- 1-1/(1+exp(parvals.theta.inf$logitmu[k]))
  B <- A*(1-mu)/mu
  stuff.k <- loglikfn(par=c(log(A),log(B)),theta=Inf,
                      ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,B=800,
                      Nbar=100,deviance=TRUE,R=1e3,leakage=TRUE)
  parvals.theta.inf[k,c("deviance","P.L","E.L")] <-
    stuff.k[c("deviance","P.L","E.L")]
  stuff.k2 <- loglikfn2.theta.inf(par=c(log(A),log(B)),theta=Inf,
                      ty=c(0,4,7,21),freq=c(68,2,1,1),b=94,B=800,
                      Nbar=100,deviance=TRUE,R=1e3,leakage=FALSE)
  parvals.theta.inf[k,"deviance2"] <- stuff.k2
}
parvals.theta.inf$alpha <- exp(parvals.theta.inf$logalpha)
parvals.theta.inf$mu <- 1-1/(1+exp(parvals.theta.inf$logitmu))
parvals.theta.inf$beta <- parvals.theta.inf$alpha*(1-parvals.theta.inf$mu)/parvals.theta.inf$mu
parvals.theta.inf$deviance <- parvals.theta.inf$deviance-min(parvals.theta.inf$deviance,na.rm=TRUE)
parvals.theta.inf$deviance2 <- parvals.theta.inf$deviance2-min(parvals.theta.inf$deviance2,na.rm=TRUE)
parvals.theta.inf$deviance[is.na(parvals.theta.inf$deviance)] <- Inf
save(parvals.theta.inf,file="parvals.theta.inf_06_04_2023.rdata")
  # load("parvals.theta.inf_06_04_2023.rdata")
parvals.theta.inf$loglik.approx <- with(parvals.theta.inf,
                                        68*extraDistr::dbbinom(0,size=94,alpha=alpha,beta=beta/100,log=TRUE)+
                                          2*extraDistr::dbbinom(4,size=94,alpha=alpha,beta=beta/100,log=TRUE)+
                                          extraDistr::dbbinom(7,size=94,alpha=alpha,beta=beta/100,log=TRUE)+
                                          extraDistr::dbbinom(21,size=94,alpha=alpha,beta=beta/100,log=TRUE))
parvals.theta.inf$deviance.approx <- 2*(max(parvals.theta.inf$loglik.approx)-parvals.theta.inf$loglik.approx)
parvals.theta.inf[which.min(parvals.theta.inf$deviance),]
parvals.theta.inf[which.min(parvals.theta.inf$deviance.approx),]
with(parvals.theta.inf,range(alpha[deviance<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(beta[deviance<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(mu[deviance<=qchisq(0.95,1)]))*1e3
with(parvals.theta.inf,range(E.L[deviance<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(P.L[deviance<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(alpha[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(beta[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(mu[deviance.approx<=qchisq(0.95,1)]))*1e3
with(parvals.theta.inf,range(E.L[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta.inf,range(P.L[deviance.approx<=qchisq(0.95,1)]))

# profile for theta=0 can be done more efficiently, because in this case,
#  p.i ~ Beta(alpha,beta)
#  ty.i | p.i ~ Binomial(94,p.i)
# So, ty.i ~ BetaBinomial(94,alpha,beta)
parvals.theta0 <- expand.grid(logalpha=o.theta0$par[1]+seq(-4.6,4.6,0.02),
                                 logitmu=o.theta0$par[1]-o.theta0$par[2]+seq(-4.6,4.6,0.02))
parvals.theta0$alpha <- exp(parvals.theta0$logalpha)
parvals.theta0$mu <- 1-1/(1+exp(parvals.theta0$logitmu))
parvals.theta0$beta <- parvals.theta0$alpha*(1-parvals.theta0$mu)/parvals.theta0$mu
parvals.theta0$loglik <- with(parvals.theta0,68*(lchoose(94,0)+lbeta(0+alpha,94-0+beta)-lbeta(alpha,beta))+
                                2*(lchoose(94,4)+lbeta(4+alpha,94-4+beta)-lbeta(alpha,beta))+
                                1*(lchoose(94,7)+lbeta(7+alpha,94-7+beta)-lbeta(alpha,beta))+
                                1*(lchoose(94,21)+lbeta(21+alpha,94-21+beta)-lbeta(alpha,beta)) )
parvals.theta0$deviance <- 2*(max(parvals.theta0$loglik)-parvals.theta0$loglik)
parvals.theta0$P.L <- with(parvals.theta0,exp(lbeta(alpha,beta+94)-lbeta(alpha,beta))-exp(lbeta(alpha,beta+1000)-lbeta(alpha,beta)))
parvals.theta0$E.L <- with(parvals.theta0,(1000-94)*100*exp(lbeta(alpha+1,beta+94)-lbeta(alpha,beta)))
parvals.theta0[which.max(parvals.theta0$loglik),]
parvals.theta0$loglik.approx <- with(parvals.theta0,68*dnbinom(0,mu=alpha/beta*94,size=alpha,log=TRUE) +
                                          2*dnbinom(4,mu=alpha/beta*94,size=alpha,log=TRUE) +
                                          dnbinom(7,mu=alpha/beta*94,size=alpha,log=TRUE) +
                                          dnbinom(21,mu=alpha/beta*94,size=alpha,log=TRUE) )
parvals.theta0$deviance.approx <- 2*(max(parvals.theta0$loglik.approx)-parvals.theta0$loglik.approx)
with(parvals.theta0,range(alpha[deviance<=qchisq(0.95,1)]))
with(parvals.theta0,range(beta[deviance<=qchisq(0.95,1)]))
with(parvals.theta0,range(mu[deviance<=qchisq(0.95,1)]))*1e3
with(parvals.theta0,range(E.L[deviance<=qchisq(0.95,1)]))
with(parvals.theta0,range(P.L[deviance<=qchisq(0.95,1)]))
with(parvals.theta0,range(alpha[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta0,range(beta[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta0,range(mu[deviance.approx<=qchisq(0.95,1)]))*1e3
with(parvals.theta0,range(E.L[deviance.approx<=qchisq(0.95,1)]))
with(parvals.theta0,range(P.L[deviance.approx<=qchisq(0.95,1)]))

###########################################################################
# Produce Results for Cucurbits Table 1 for paper
############################################################################

# Output first row of Cucurbits Table 1 for paper
cat("$\\lambda=1$ &",
    format(round(exp(o.theta.inf$par[1]),digits=4),nsmall=4)," (",
    format(round(min(parvals.theta.inf$alpha[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta.inf$alpha[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),") & ",
    format(round(exp(o.theta.inf$par[2]),digits=1),nsmall=1)," (",
    format(round(min(parvals.theta.inf$beta[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),",",
    format(round(max(parvals.theta.inf$beta[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),") & ",
    format(round(o.theta.inf$mu*1e3,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta.inf$mu[parvals.theta.inf$deviance<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),",",
    "\\phantom{0}",format(round(max(parvals.theta.inf$mu[parvals.theta.inf$deviance<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),") & ",
    format(round(o.theta.inf$E.L,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta.inf$E.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3),",",
    "\\phantom{0}",format(round(max(parvals.theta.inf$E.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3),") & ",
    format(round(o.theta.inf$P.L,digits=4),nsmall=4)," (",
    format(round(min(parvals.theta.inf$P.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta.inf$P.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),") \\\\ \n  ",
    file="../bulk sampling biometrics MS/table1_21_04_2023.txt",append=FALSE,sep="")
# Now append second row of Cucurbits Table 1 for paper (approx theta=infinity)
cat("$\\lambda=1$ (approx) &",
    format(round(approx.alpha,digits=4),nsmall=4)," (",
    format(round(min(parvals.theta.inf$alpha[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta.inf$alpha[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),") & ",
    format(round(approx.beta,digits=1),nsmall=1)," (",
    format(round(min(parvals.theta.inf$beta[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=1),nsmall=1),",",
    format(round(max(parvals.theta.inf$beta[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=1),nsmall=1),") & ",
    format(round(approx.E.p*1e3,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta.inf$mu[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),",",
    "\\phantom{0}",format(round(max(parvals.theta.inf$mu[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),") & ",
    format(round(approx.E.L,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta.inf$E.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=3),nsmall=3),",",
    "\\phantom{0}",format(round(max(parvals.theta.inf$E.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=3),nsmall=3),") & ",
    format(round(approx.P.L,digits=4),nsmall=4)," (",
    format(round(min(parvals.theta.inf$P.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta.inf$P.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),") \\\\ \n  ",
    file="../bulk sampling biometrics MS/table1_21_04_2023.txt",append=TRUE,sep="")
# Now append third row of Cucurbits Table 1 for paper (theta=0)
cat("$\\lambda=\\bar{N}=100$ &",
    format(round(exp(o.theta0$par[1]),digits=4),nsmall=4)," (",
    format(round(min(parvals.theta0$alpha[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta0$alpha[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),") & ",
    format(round(exp(o.theta0$par[2]),digits=1),nsmall=1)," (",
    format(round(min(parvals.theta0$beta[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),",",
    "\\phantom{0}\\phantom{0}",
    format(round(max(parvals.theta0$beta[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),") & ",
    format(round(o.theta0$mu*1e3,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta0$mu[parvals.theta0$deviance<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),",",
    format(round(max(parvals.theta0$mu[parvals.theta0$deviance<=qchisq(0.95,1)]*1e3),digits=3),nsmall=3),") & ",
    format(round(o.theta0$E.L,digits=3),nsmall=3)," (",
    format(round(min(parvals.theta0$E.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3),",",
    format(round(max(parvals.theta0$E.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3),") & ",
    format(round(o.theta0$P.L,digits=4),nsmall=4)," (",
    format(round(min(parvals.theta0$P.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),",",
    format(round(max(parvals.theta0$P.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),") \\\\ \n  ",
    file="../bulk sampling biometrics MS/table1_21_04_2023.txt",append=TRUE,sep="")

tab1a <- cbind(
  c("$\\lambda=1$","$\\lambda=1$ (approx)","$\\lambda=\\bar{N}=100$"),"&",
  c(format(round(exp(o.theta.inf$par[1]),digits=4),nsmall=4),
    format(round(approx.alpha,digits=4),nsmall=4),
    format(round(exp(o.theta0$par[1]),digits=4),nsmall=4)),
  " & (",
  c(format(round(min(parvals.theta.inf$alpha[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(min(parvals.theta.inf$alpha[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(min(parvals.theta0$alpha[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4)),
  ",",
  c(format(round(max(parvals.theta.inf$alpha[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(max(parvals.theta.inf$alpha[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(max(parvals.theta0$alpha[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4)),
  ")&",
  c(format(round(exp(o.theta.inf$par[2]),digits=1),nsmall=1),
    format(round(approx.beta,digits=1),nsmall=1),
    format(round(exp(o.theta0$par[2]),digits=1),nsmall=1)),
  " & (",
  c(format(round(min(parvals.theta.inf$beta[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),
    format(round(min(parvals.theta.inf$beta[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=1),nsmall=1),
    format(round(min(parvals.theta0$beta[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1)),
  ",",
  c(format(round(max(parvals.theta.inf$beta[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1),
    format(round(max(parvals.theta.inf$beta[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=1),nsmall=1),
    paste0("\\phantom{0}\\phantom{0}",format(round(max(parvals.theta0$beta[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=1),nsmall=1))),
  ")&",
  c(format(round(o.theta.inf$mu*1000,digits=3),nsmall=3),
    format(round(approx.E.p*1000,digits=3),nsmall=3),
    format(round(o.theta0$mu*1000,digits=3),nsmall=3)),
  " & (",
  c(format(round(min(parvals.theta.inf$mu[parvals.theta.inf$deviance<=qchisq(0.95,1)])*1000,digits=3),nsmall=3),
    format(round(min(parvals.theta.inf$mu[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)])*1000,digits=3),nsmall=3),
    format(round(min(parvals.theta0$mu[parvals.theta0$deviance<=qchisq(0.95,1)])*1000,digits=3),nsmall=3)),
  ",",
  c(paste0("\\phantom{0}",format(round(max(parvals.theta.inf$mu[parvals.theta.inf$deviance<=qchisq(0.95,1)])*1000,digits=3),nsmall=3)),
    paste0("\\phantom{0}",format(round(max(parvals.theta.inf$mu[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)])*1000,digits=3),nsmall=3)),
    format(round(max(parvals.theta0$mu[parvals.theta0$deviance<=qchisq(0.95,1)])*1000,digits=3),nsmall=3)),
  " ) \\\\ \n  "
)

write.table(tab1a,file="../bulk sampling biometrics MS/tab1a_24_04_2023.txt",
            sep="",quote=FALSE,row.names=FALSE,col.names=FALSE)

tab1b <- cbind(
  c("$\\lambda=1$","$\\lambda=1$ (approx)","$\\lambda=\\bar{N}=100$"),"&",
  c(format(round(o.theta.inf$E.L,digits=3),nsmall=3),
    format(round(approx.E.L,digits=3),nsmall=3),
    format(round(o.theta0$E.L[1],digits=3),nsmall=3)),
  " & (",
  c(format(round(min(parvals.theta.inf$E.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3),
    format(round(min(parvals.theta.inf$E.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=3),nsmall=3),
    format(round(min(parvals.theta0$E.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3)),
  ",",
  c(paste0("\\phantom{0}",format(round(max(parvals.theta.inf$E.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3)),
    paste0("\\phantom{0}",format(round(max(parvals.theta.inf$E.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=3),nsmall=3)),
    format(round(max(parvals.theta0$E.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=3),nsmall=3)),
  ")&",
  c(format(round(o.theta.inf$P.L,digits=4),nsmall=4),
    format(round(approx.P.L,digits=4),nsmall=4),
    format(round(o.theta0$P.L,digits=4),nsmall=4)),
  " & (",
  c(format(round(min(parvals.theta.inf$P.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(min(parvals.theta.inf$P.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(min(parvals.theta0$P.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4)),
  ",",
  c(format(round(max(parvals.theta.inf$P.L[parvals.theta.inf$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(max(parvals.theta.inf$P.L[parvals.theta.inf$deviance.approx<=qchisq(0.95,1)]),digits=4),nsmall=4),
    format(round(max(parvals.theta0$P.L[parvals.theta0$deviance<=qchisq(0.95,1)]),digits=4),nsmall=4)),
  " ) \\\\ \n  "
)

write.table(tab1b,file="../bulk sampling biometrics MS/tab1b_24_04_2023.txt",
            sep="",quote=FALSE,row.names=FALSE,col.names=FALSE)





profile.exact <- with(parvals.theta.inf,tapply(deviance,mu,FUN=min))
profile.approx <- with(parvals.theta.inf,tapply(deviance.approx,mu,FUN=min))
plot(as.numeric(names(profile.exact)),profile.exact,type="l",col="blue",xlim=c(0,0.002),ylim=c(0,6))
lines(as.numeric(names(profile.approx)),profile.approx,type="l",col="red")
abline(h=qchisq(0.95,1),lty=2)

with(o.theta.inf,alpha/(alpha+beta/10/100))
with(o.theta.inf,1-beta(alpha,beta/10+100)/beta(alpha,beta/10))

########################################################################
# Produce Fig 1 for paper
########################################################################

# let theta range from 0.001 to 2000 on evenly spaced log scale
#   ok, make log(theta) range from -7.5 to 7.5 by 0.1

theta.grid <- expand.grid(logtheta=c(-Inf,seq(from=-7.5,to=7.5,by=0.1),Inf),
                          alpha=NA,beta=NA,deviance=NA,P.L=NA,E.L=NA)
theta.grid$theta <- exp(theta.grid$logtheta)
for(k in c(1:nrow(theta.grid))){
  cat("k=",k,"\n")
  optim.k <- optim(c(0,0),loglikfn,ty=c(0,4,4,7,21),freq=c(68,1,1,1,1),b=94,
                   theta=theta.grid$theta[k],Nbar=100,deviance=TRUE,
                   control=list(reltol=1e-12),R=1e4)
  leakage.info.k <- loglikfn(optim.k$par,ty=c(0,4,4,7,21),freq=c(68,1,1,1,1),
                             b=94,B=800,theta=theta.grid$theta[k],
                             Nbar=100,R=1e4,leakage=TRUE)
  theta.grid[k,c("alpha","beta")] <- exp(optim.k$par)
  theta.grid[k,c("deviance","P.L","E.L")] <- leakage.info.k
}
theta.grid$mu <- theta.grid$alpha/(theta.grid$alpha+theta.grid$beta)
# calculate lambda assuming Nbar=100
theta.grid$g.theta <- 1
for(k in c(1:99)){
  theta.grid$g.theta<-theta.grid$g.theta + theta.grid$theta/(theta.grid$theta+k)
}
theta.grid$lambda <- 100 / theta.grid$g.theta
postscript("../bulk sampling biometrics MS/theta_plots_26_04_2023.eps",width=6,height=4)
par(mfrow=c(2,3))
with(theta.grid,plot(lambda,alpha,cex=0.4,ylim=c(0.0162,0.0164),yaxp=c(0.0162,0.0164,2),
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="",las=1,cex.axis=0.7,cex.main=0.7,
                     main=latex2exp::TeX("(a): $\\hat{\\alpha}$ assuming various values of $\\lambda$")))
mtext(side=2,text=latex2exp::TeX("$\\hat{\\alpha}$"),las=1,line=3.2,cex=0.7)
with(theta.grid,plot(lambda,beta,cex=0.4,
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="",las=1,cex.axis=0.7,cex.main=0.7,
                     main=latex2exp::TeX("(b): $\\hat{\\beta}$ assuming various values of $\\lambda$")))
mtext(side=2,text=latex2exp::TeX("$\\hat{\\beta}$"),las=1,line=3.2,cex=0.7)
with(theta.grid,plot(lambda,mu,cex=0.4,
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="",las=1,cex.axis=0.7,cex.main=0.7,
                     main=latex2exp::TeX("(c): $\\hat{\\mu}$ assuming various values of $\\lambda$")))
mtext(side=2,text=latex2exp::TeX("$\\hat{\\mu}$"),las=1,line=3.2,cex=0.7)
with(theta.grid,plot(lambda,P.L,cex=0.4,cex.main=0.7,
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="",las=1,cex.axis=0.7,yaxp=c(0.0319,0.0321,2),#ylim=c(0.0315,0.0325),
                     main=latex2exp::TeX("(d): $\\hat{P}(L>0)$ assuming various values of $\\lambda$")))
mtext(side=2,text=latex2exp::TeX("$\\hat{P}(L>0)$"),line=2.9,cex=0.5)
with(theta.grid,plot(lambda,E.L,cex=0.4,cex.main=0.7,
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="",las=1,cex.axis=0.7,
                     main=latex2exp::TeX("(e): $\\hat{E}(L)$ assuming various values of $\\lambda$")))
mtext(side=2,text=latex2exp::TeX("$\\hat{E}(L)$"),line=2.5,cex=0.5)
with(theta.grid,plot(lambda,deviance-min(deviance),cex=0.4,cex.main=0.7,
                     xlab=latex2exp::TeX("$\\lambda$"),
                     ylab="deviance",las=1,cex.axis=0.7,#ylim=c(0,0.06),
                     main=latex2exp::TeX("(f): Deviance assuming various values of $\\theta$")))
dev.off()
par(mfrow=c(1,1))


########################################################################
# Produce Fig 2 for paper
# (distribution of # seeds, unconditional, and conditional on ty=0)
########################################################################

B <- 800
b <- 94
Nbar <- 100
nn <- b*Nbar
N <- B*Nbar
Txvals <- c(0:N)
log.probs.uncond <- lchoose(N,Txvals)+
  lbeta(Txvals+o.theta.inf$alpha,N-Txvals+o.theta.inf$beta)-
  lbeta(o.theta.inf$alpha,o.theta.inf$beta)
log.probs.cond0 <- lchoose(N-nn,Txvals)+
  lbeta(Txvals+o.theta.inf$alpha,N-nn-Txvals+nn+o.theta.inf$beta)-
  lbeta(o.theta.inf$alpha,o.theta.inf$beta+nn)
probs.uncond <- exp(log.probs.uncond)
probs.cond0 <- exp(log.probs.cond0)
sum(Txvals*probs.uncond)
sum(Txvals*probs.cond0)
sum(probs.cond0)

sum(probs.uncond[1:100])
plot(Txvals[2:100],probs.uncond[2:100])

probs.uncond.trunc <- c(probs.uncond[1:15],1-sum(probs.uncond[1:15]))
probs.cond0.trunc <- c(probs.cond0[1:15],1-sum(probs.cond0[1:15]))

postscript("../bulk sampling biometrics MS/Tx_barplot_06_04_2023.eps",width=5,height=7)
(bp <- barplot(rbind(rev(probs.uncond.trunc),rev(probs.cond0.trunc)),log="x",beside=TRUE,las=2,
               space=c(0.2,0.6),xlim=c(1e-4,2),horiz=TRUE,
               names.arg=rev(c(as.character(0:14),"15+")),cex.axis=0.7,cex.names=0.7) )
legend("right",col=c("black","grey"),cex=0.7,
       legend=c("unconditional",latex2exp::TeX("conditional on $\\t_{yi}=0$")),
       pch=15)
text(y=bp[1,],x=rev(probs.uncond.trunc)*1.5,srt=0,cex=0.5,
     labels=format(round(rev(probs.uncond.trunc),3),nsmall=3))
text(y=bp[2,],x=rev(probs.cond0.trunc)*1.5,srt=0,cex=0.5,
     labels=format(round(rev(probs.cond0.trunc),3),nsmall=3))
dev.off()


