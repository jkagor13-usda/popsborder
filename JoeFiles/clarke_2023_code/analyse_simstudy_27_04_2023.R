#########################################################################
# This script analyses simulation results for Section 5.5 of the paper
# 27/4/2023
#########################################################################

load("paperspace_sep22/all_cucurbit_simulation_results_27_04_2023.rdata")
pargrid$beta <- pargrid$alpha*(1-pargrid$mu)/pargrid$mu
pargrid$rho <- 1/(pargrid$alpha+pargrid$beta+1)
pargrid$D <- 1+(pargrid$Nbar-1)*pargrid$rho
pargrid$sigma <- sqrt(pargrid$mu^2*(1-pargrid$mu)/(pargrid$mu+pargrid$alpha))

simresultsDF <- simresults[[1]]
for(k in c(2:length(simresults))){
  simresultsDF <- rbind(simresultsDF,simresults[[k]])
}

K <- max(pargrid$scenario)
J <- nrow(pargrid)
simresults2 <- vector("list",K)
for(k in c(1:K)){
  for(j in c(1:J)){
    if(simresults[[j]]$scenario[1]==k)
      simresults2[[k]] <- rbind(simresults2[[k]],simresults[[j]])
  }
  simresults2[[k]]$alpha0 <- pargrid$alpha[k]
  simresults2[[k]]$beta0 <- pargrid$beta[k]
  simresults2[[k]]$mu0 <- pargrid$mu[k]
  simresults2[[k]]$rho0 <- pargrid$rho[k]
  simresults2[[k]]$D0 <- pargrid$D[k]
  simresults2[[k]]$sigma0 <- pargrid$sigma[k]
  n <- pargrid$b[k]*pargrid$Nbar[k]
  N <- pargrid$B[k]*pargrid$Nbar[k]
  if(pargrid$theta[k]==Inf){
    simresults2[[k]]$pty0.true <- exp(lbeta(pargrid$alpha[k],pargrid$beta[k]+n)-lbeta(pargrid$alpha[k],pargrid$beta[k]))
    simresults2[[k]]$E.leak.true <- (N-n)*pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+n)*beta(pargrid$alpha[k],pargrid$beta[k]+n)/beta(pargrid$alpha[k],pargrid$beta[k])
    simresults2[[k]]$log.prob.leak.true <- sum(log1p(-pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+c(0:(n-1))))) +
      Rmpfr::log1mexp(-sum(log1p(-pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+c(n:(N-1))))))
    simresults2[[k]]$prob.leak.true <- exp(simresults2[[k]]$log.prob.leak.true)
  }
  if(pargrid$theta[k]==0){
    simresults2[[k]]$pty0.true <- exp(lbeta(pargrid$alpha[k],pargrid$beta[k]+pargrid$b[k])-lbeta(pargrid$alpha[k],pargrid$beta[k]))
    simresults2[[k]]$E.leak.true <- (N-n)*pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+pargrid$b[k])*beta(pargrid$alpha[k],pargrid$beta[k]+pargrid$b[k])/beta(pargrid$alpha[k],pargrid$beta[k])
    simresults2[[k]]$log.prob.leak.true <- sum(log1p(-pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+c(0:(pargrid$b[k]-1))))) +
      Rmpfr::log1mexp(-sum(log1p(-pargrid$alpha[k]/(pargrid$alpha[k]+pargrid$beta[k]+c(n:(pargrid$B[k]-1))))))
    simresults2[[k]]$prob.leak.true <- exp(simresults2[[k]]$log.prob.leak.true)
  }
  if((pargrid$theta[k]>0)&(pargrid$theta[k]<Inf)){
    p.i.vals<-qbeta(c(1:1000)/1001,pargrid$alpha[k],pargrid$beta[k])
    ratio.vals <- beta.ratio(alpha=pargrid$theta[k],beta=pargrid$theta[k]*(1/p.i.vals-1),bplus=pargrid$Nbar[k])
    simresults2[[k]]$pty0.true<-mean(ratio.vals^pargrid$b[k])
    simresults2[[k]]$E.leak.true<-mean((N-n)*p.i.vals*ratio.vals^pargrid$b[k])
    simresults2[[k]]$prob.leak.true<-mean(ratio.vals^pargrid$b[k]*(1-ratio.vals^(pargrid$B[k]-pargrid$b[k])))
  }
}

sim.summary.fn <- function(X){
  X$a[X$max.ty==0] <- Inf
  X$b[X$max.ty==0] <- Inf
  X$mu[X$max.ty==0] <- 0
  X$sigma <- sqrt(X$mu^2*(1-X$mu)/(X$a+X$mu))
  c(mean.ty=mean(X$mean.ty),
    prop.zero.shipments.pathogen=mean(X$shipments.positive==0),
    prop.one.shipments.pathogen=mean(X$shipments.positive==1),
    prop.ty01=mean(X$max.ty<=1),
    pty0.true=mean(X$pty0),
    E.leak.true=mean(X$E.leak.true),
    prob.leak.true=mean(X$prob.leak.true),
    rmse.alpha=sqrt(mean((X$a-X$alpha0)^2)),
    rmse.pct.alpha=sqrt(mean((X$a-X$alpha0)^2))/mean(X$alpha0)*100,
    rmse.alpha.fn=sqrt(mean(((1-1/(X$a+1))-(1-1/(X$alpha0+1)))^2)),
    rmse.alpha2=sqrt(mean(((X$a-X$alpha0)[X$max.ty>=2])^2)),
    rmse.mu=sqrt(mean((X$mu-X$mu0)^2)),
    rmse.pct.mu=sqrt(mean((X$mu-X$mu0)^2))/mean(X$mu0)*100,
    rmse.mu2=sqrt(mean(((X$mu-X$mu0)[X$max.ty>=2])^2)),
    rmse.sigma=sqrt(mean((X$sigma-X$sigma0)^2)),
    rmse.rho=sqrt(mean((X$rho-X$rho0)^2)),
    rmse.rho2=sqrt(mean(((X$rho-X$rho0)[X$max.ty>=2])^2)),
    mad.mu=1.4826*median(abs(X$mu-X$mu0)),
    mad.pct.mu=1.4826*median(abs(X$mu-X$mu0))/mean(X$mu0)*100,
    mad.pct.mu2=1.4826*median(abs(X$mu-X$mu0)[X$max.ty>=2])/mean(X$mu0)*100,
    mad.rho=1.4826*median(abs(X$rho-X$rho0)),
    mad.alpha=1.4826*median(abs(X$a-X$alpha0)),
    mad.pct.alpha=1.4826*median(abs(X$a-X$alpha0))/mean(X$alpha0)*100,
    mad.pct.alpha2=1.4826*median(abs(X$a-X$alpha0)[X$max.ty>=2])/mean(X$alpha0)*100,
    mad.mu2=1.4826*median(abs((X$mu-X$mu0)[X$max.ty>=2])),
    mad.alpha2=1.4826*median(abs((X$a-X$alpha0)[X$max.ty>=2])),
    medbias.mu=median(X$mu-X$mu0),
    medbias.pct.mu=median(X$mu-X$mu0)/mean(X$mu0)*100,
    medbias.pct.mu2=median((X$mu-X$mu0)[X$max.ty>=2])/mean(X$mu0)*100,
    medbias.alpha=median(X$a-X$alpha0),
    medbias.pct.alpha=median(X$a-X$alpha0)/mean(X$alpha0)*100,
    medbias.pct.alpha2=median((X$alph-X$alpha0)[X$max.ty>=2])/mean(X$alpha0)*100,
    medbias.mu2=median((X$mu-X$mu0)[X$max.ty>=2]),
    medbias.alpha2=median((X$a-X$alpha0)[X$max.ty>=2]),
    rmse.pty0=sqrt(mean((X$pty0-X$pty0.true)^2)),
    rmse.E.leak=sqrt(mean((X$E.leak-X$E.leak.true)^2)),
    rmse.prob.leak=sqrt(mean((X$prob.leak-X$prob.leak.true)^2)),
    bias.alpha=mean(X$a-X$alpha0),
    bias.pct.alpha=mean(X$a-X$alpha0)/mean(X$alpha0)*100,
    bias.alpha.fn=mean((1-1/(X$a+1))-(1-1/(X$alpha0+1))),
    bias.alpha2=mean((X$a-X$alpha0)[X$max.ty>=2]),
    bias.mu=mean(X$mu-X$mu0),
    bias.pct.mu=mean(X$mu-X$mu0)/mean(X$mu0)*100,
    bias.mu2=mean((X$mu-X$mu0)[X$max.ty>=2]),
    bias.sigma=mean(X$sigma-X$sigma0),
    bias.rho=mean(X$rho-X$rho0),
    bias.rho2=mean((X$rho-X$rho0)[X$max.ty>=2]),
    bias.mu.median=median(X$mu-X$mu0),
    bias.rho.median=median(X$rho-X$rho0),
    bias.pty0=mean(X$pty0-X$pty0.true),
    bias.E.leak=mean(X$E.leak-X$E.leak.true),
    bias.prob.leak=mean(X$prob.leak-X$prob.leak.true),
    noncover.alpha=100*mean(abs((X$a-X$alpha0)/X$se.a)>qnorm(0.975)),
    noncover.alpha2=100*mean(abs((X$a-X$alpha0)/X$se.a)[X$max.ty>=2]>qnorm(0.975)),
    noncover.mu=100*mean(abs((X$mu-X$mu0)/X$se.mu)>qnorm(0.975)),
    noncover.mu2=100*mean(abs((X$mu-X$mu0)/X$se.mu)[X$max.ty>=2]>qnorm(0.975)),
    noncover.rho=100*mean(abs((X$rho-X$rho0)/X$se.rho)>qnorm(0.975)),
    noncover.rho2=100*mean(abs((X$rho-X$rho0)/X$se.rho)[X$max.ty>=2]>qnorm(0.975)),
    prop.leak=mean(X$P.leak),
    noncover.mu.profile=100*mean(X$profile.deviance.mu>qchisq(0.95,1)),
    noncover.shape.profile=100*mean(X$profile.deviance.shape>qchisq(0.95,1)),
    noncover.mu.profile2=100*mean(X$profile.deviance.mu[X$max.ty>=2]>qchisq(0.95,1)),
    noncover.shape.profile2=100*mean(X$profile.deviance.shape[X$max.ty>=2]>qchisq(0.95,1)),
    noncover.alpha.profile=100*mean(X$profile.deviance.alpha>qchisq(0.95,1))
  )
}

K <- length(unique(pargrid$scenario))
short.results <- as.data.frame(t(sapply(simresults2,sim.summary.fn)))
short.results <- cbind(pargrid[1:K,-10],short.results)
short.results$lambda <- 100 / short.results$g


short.results2 <- short.results[,
  c("scenario","num.consignments","alpha","mu","lambda","mad.pct.alpha","mad.pct.mu",
    "medbias.pct.alpha","medbias.pct.mu","noncover.alpha","noncover.mu",
    "noncover.alpha.profile","noncover.mu.profile")]
short.results2[,4] <- sapply(strsplit(format(signif(short.results2[,4],3), scientific=TRUE), split="e"),
                             function(x) paste0("$", x[1], " \\times 10^{", x[2], "}$"))
#short.results2[,5] <- format(round(short.results2[,5],digits=3),nsmall=3)
short.results2[,c(6:13)] <- format(round(short.results2[,c(6:13)],digits=1),nsmall=1)
#short.results2$theta[short.results2$theta=="  Inf"] <- "$\\infty$"
(tab1 <- short.results2[c(17:20,9:12,1:4),
              c("scenario","num.consignments","alpha","mu","lambda","mad.pct.alpha","mad.pct.mu",
                "medbias.pct.alpha","medbias.pct.mu","noncover.alpha","noncover.mu",
                "noncover.alpha.profile","noncover.mu.profile")])
tab1[c(4,8),ncol(tab1)] <- paste0(tab1[c(4,8),ncol(tab1)],"\\vspace{2mm}")
write.table(tab1[,c(3,5,4,6:13)],file=paste0("../bulk sampling biometrics MS/simtab1_27_04_2023.txt"),
            sep=" & ",eol=" \\\\ \n" , row.names=FALSE , col.names=FALSE,quote=F)


(tab2 <- short.results2[c(17:20,9:12,1:4)+4,
                        c("scenario","num.consignments","alpha","mu","lambda","mad.pct.alpha","mad.pct.mu",
                          "medbias.pct.alpha","medbias.pct.mu","noncover.alpha","noncover.mu",
                          "noncover.alpha.profile","noncover.mu.profile")])
tab2[c(4,8),ncol(tab2)] <- paste0(tab2[c(4,8),ncol(tab2)],"\\vspace{2mm}")
write.table(tab2[,c(3,5,4,6:13)],file=paste0("../bulk sampling biometrics MS/simtab2_27_04_2023.txt"),
            sep=" & ",eol=" \\\\ \n" , row.names=FALSE , col.names=FALSE,quote=F)

