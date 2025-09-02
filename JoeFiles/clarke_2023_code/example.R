##########################################################################
# Perform calculations and produce figures for statistics introduced
# Sections 3.1 - 3.3 of journal paper
##########################################################################
## import functions

source('R/functions.R')
library(ggplot2)

#################################
#    setting the model parameters
#################################

# number of simulation
nsimVal = 100000
# number of units in the population
N= 100000
# number of units in the sample tested
n = 9400
# beta dist parameters
alpha = .0168
beta = 248
# number of units in in a single bulk sample
Nbar =100
ty = 0
# number of bulk samples tested bNbar = n
b=floor(n/Nbar)
# Number of bulk samples in the population
B= floor(N/Nbar)


#############################################
#        inference without clustering
#############################################
# sample size
nsim = nsimVal

# prior for prevalence
prior = rbeta(nsim, alpha, beta)
prior = as.data.frame(prior)

#mean
mean(prior$prior)
sd(prior$prior)
median(prior$prior)

mean = alpha/(alpha+beta)#mean(prior$prior,na.rm = T)
median = median(prior$prior, na.rm = T)
sd = sd(prior$prior, na.rm = T)
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)
if (mean >median){
  trs = mean+sd
}else{
  trs = median+sd
}

#plot beta prior
ggplot(data=subset(prior, prior<=trs), aes(x=prior,y=..count../sum(..count..)))+
  geom_histogram(alpha=.5, position="identity")+
  labs(x=paste("prevalence (p)\n", "Mean= ", round(mean,digits = 8),", ", "Median= ", round(median,digits = 8),", ", "sd= ",
               round(sd,digits = 8),collapse = ", "), y="pdf")+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Prior: (p)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())


# posterior p|ty
support <- seq(0.0, 1, 0.0001)
post_size = length(support)
#post_size
posterior <- data.frame(xp = rep(NA, post_size), yp = rep(NA, post_size))
for (i in c(1:post_size)) {
  posterior[i, 1] <- fp(support[i], alpha, beta, Nbar, ty, b)
  posterior[i, 2] <- support[i]
}

posterior = subset(posterior, !is.na(xp) &!is.inf(xp))
posterior$xp = posterior$xp/sum(posterior$xp)

mean = sum(posterior$xp*posterior$yp)

sd =sqrt(sum(posterior$xp* (posterior$yp-mean)^2))

i<-1
tmpMed =0
while (i < nrow(posterior)) {
  tmpMed = tmpMed+posterior$xp[i]
  if (tmpMed>0.5){
    median = posterior$yp[i]
    i <- nrow(posterior)+1
  } else{
    i =i+1
  }
}
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)
if (mean >median){
  trs = mean+5*sd
}else{
  trs = median+5*sd
}
ggplot(data=subset(posterior, yp<=trs), aes(x=yp,y=xp))+
  geom_col(fill = "grey")+
  labs(x=paste("prevalence (p|ty)\n", "Mean= ", round(mean,digits = 6), ", ",
               "Median= ",round(median,digits = 6),", ",
               "sd= ",round(sd,digits = 6)),y="pdf")+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Posterior: (p|ty)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())




# posterior for Tx|ty
Tx = data.frame(x= rep(NA,N),y= rep(NA,N))

if (ty>0){
  #marginal = integrate(function(x) {fp(x,alpha, beta, Nbar, ty, b)}, 0,1)
  for (i in c(1:N)) {
    Tx[i,1]= Tx_given_ty_NC(N,Nbar,b,ty,i,alpha,beta)#/marginal$value
    Tx[i,2] =i
  }
  Tx$x = Tx$x/sum(Tx$x)
}else{
  i<-0
  tmpCnt =0
  marginal = lbeta(alpha, Nbar*b + beta)
  while (i < N) {
    Tx[i+1,1]= exp(lbeta(alpha + i, N + beta - i) -marginal+lchoose(N-b*Nbar,i))
    Tx[i+1,2] =i
    tmpCnt = tmpCnt+Tx[i+1,1]
    if (tmpCnt>0.9999){
      i <- N+1
    } else{
      i =i+1
    }
  }
}
# Tx|ty
Tx = subset(Tx, !is.na(x))

# plot Tx|ty
mean = sum(Tx$x*Tx$y)
#median = NA #median(Tx$x, na.rm = T)
sd =sqrt(sum(Tx$x* (Tx$y-mean)^2))

i<-1
tmpMed =0

while (i < N) {
  tmpMed = tmpMed+Tx$x[i]
  if (tmpMed>=0.5){
    median = Tx$y[i]
    i <- N+1
  } else{
    i =i+1
  }
}

# summary stats
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)

if (mean >median){
  trs = mean+2*sd
}else{
  trs = median+2*sd
}

ggplot(data=subset(Tx, y< trs), aes(x=y,y=x))+
  geom_col(fill = "grey")+
  labs(x=paste("Nbr contaminated units (Tx|ty)\n", "Mean= ", round(mean,digits = 6), ", ",
               "Median= ",round(median,digits = 6),", ",
               "sd= ",round(sd,digits = 6)),y="pdf")+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Posterior:  (Tx|ty)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())



# Tx-tx|ty
Tx_tx = data.frame(xtx= rep(NA,N),ytx= rep(NA,N))

if (ty>0){
  #marginal = integrate(function(x) {fp(x,alpha, beta, Nbar, ty, b)}, 0,1)
  for (i in c(1:N)) {
    Tx_tx[i,1]= Tx_tx_given_ty_NC(N,Nbar,b,ty,i,alpha,beta)#/marginal$value
    Tx_tx[i,2] =i
  }
  Tx_tx$xtx <- Tx_tx$xtx/sum(Tx_tx$xtx)
}else{
  Tx_tx = Tx
  colnames(Tx_tx) <- c('xtx','ytx')
}

Tx_tx = subset(Tx_tx, !is.na(xtx))

# plot Tx-tx|ty
mean = sum(Tx_tx$xtx*Tx_tx$ytx)
sd =sqrt(sum(Tx_tx$xtx* (Tx_tx$ytx-mean)^2))

i<-1
tmpMed =0

while (i < N) {
  tmpMed = tmpMed+Tx_tx$xtx[i]
  if (tmpMed>0.5){
    median = Tx_tx$ytx[i]
    i <- N+1
  } else{
    i =i+1
  }
}

# summary stats
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)

if (mean >median){
  trs = mean+2*sd
}else{
  trs = median+2*sd
}

ggplot(data=subset(Tx_tx, ytx< trs), aes(x=ytx,y=xtx))+
  geom_col(fill = "grey")+
  labs(x=paste("Nbr contaminated units (Tx-tx|ty)\n", "Mean= ", round(mean,digits = 6), ", ",
               "Median= ",round(median,digits = 6),", ",
               "sd= ",round(sd,digits = 6)),y="pdf")+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Posterior: (Tx-tx|ty)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())


#############################################
#        inference with clustering
#############################################
# sample size
# number of simulation
nsimVal = 100000
# number of units in the population
N= 100000
# number of units in the sample tested
n = 9400
# beta dist parameters
alpha = 1
beta = 100
# number of units in in a single bulk sample
Nbar =100
ty = 0
theta = .5
# number of bulk samples tested bNbar = n
b=floor(n/Nbar)
# Number of bulk samples in the population
B= floor(N/Nbar)

#

# set support of pdf
support <- seq(0.000001, 1, 0.000001)

# Find the probability densities over the support
probs <- pfc(support, alpha, beta, theta, Nbar, ty, b)
probs<-probs[!is.na(probs)]
probs <- probs/sum(probs) # normalize

# # number of sample nsim
nsim = 10000

# prior p
prior = rbeta(nsim, alpha, beta)
prior = as.data.frame(prior)

mean = mean(prior$prior,na.rm = T)
median = median(prior$prior, na.rm = T)
sd = sd(prior$prior, na.rm = T)
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)
##  plot beta prior
ggplot(data=subset(prior, prior<= max(mean,median)+2*sd), aes(x=prior,y=..count../sum(..count..)))+
  geom_histogram(alpha=.5, position="identity")+
  labs(x=paste("prevalence (p)\n", "Mean= ", round(mean,digits = 6),", ", "Median= ",round(median,digits = 6),", ",
               "sd= ",round(sd,digits = 6)),
       y="pdf", show.legend = TRUE)+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Prior: (p)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())

# posterior p|ty
simulated <- sample_fun(nsim,support,probs)
posterior = simulated
posterior = as.data.frame(posterior)

mean = mean(posterior$posterior,na.rm = T)
median = median(posterior$posterior, na.rm = T)
sd = sd(posterior$posterior, na.rm = T)
stats <- data.frame(Ref = c("Mean", "Median"),
                    vals = c(mean, median),
                    stringsAsFactors = FALSE)
#       plot beta prior
ggplot(data=subset(posterior,posterior<= max(mean,median)+2*sd), aes(x=posterior,y=..count../sum(..count..)))+
  geom_histogram(alpha=.5, position="identity")+
  labs(x=paste("prevalence (p|ty)\n", "Mean= ", round(mean,digits = 6),", ", "Median= ",round(median,digits = 6),", ",
               "sd= ",round(sd,digits = 6)), y="pdf")+
  geom_vline(aes(xintercept=median,
                 color="median"), linetype="solid",
             size=1) +
  geom_vline(aes(xintercept=mean,
                 color="mean"), linetype="solid",
             size=1) +
  scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
  theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggtitle("Posterior: (p|ty)") +
  theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                      legend.justification=c(1,0),
                                                      legend.position=c(.8, .75),
                                                      legend.background = element_blank(),
                                                      legend.key = element_blank())



# Tx|ty
y = c(rep(1, ty), rep(0, b-ty), rep(1, B-b))
Txc = data.frame(x= rep(0,length(nsim))) # Tx in the paper
txcl = data.frame(tx= rep(0,length(nsim))) # tx in the paper

supportZTBB <- seq(0, Nbar, 1)

# Find the probability densities over the support
for (cnt in c(1:nsim)){
  p = simulated[cnt]
  Xis <- 0
  if (ty>0){
    probsTBB = data.frame(x= rep(NA,Nbar),y= rep(NA,Nbar))
    for (i in c(0:Nbar)){
      probsTBB[i+1,1] = ztbb(i, Nbar,theta,theta*(1/p-1))
      probsTBB[i+1,2] = i
    }
    probsTBB<-subset(probsTBB, !is.na(x))
    probsTBB <- probsTBB$x/sum(probsTBB$x) # normalize

    tmpVal = sum(sample_fun(ty,supportZTBB,probsTBB))
    Txc[cnt,1] = sum(rbbinom(B-b, Nbar,theta,theta*(1/p-1)))+
      tmpVal
    txcl[cnt,1] = tmpVal
  }else{
    Txc[cnt,1] = sum(rbbinom(B-b, Nbar,theta,theta*(1/p-1)))
    txcl[cnt,1] = 0
  }
}

# plot
if (ty ==0){
  mean = mean(Txc$x,na.rm = T)
  median = median(Txc$x, na.rm = T)
  sd = sd(Txc$x, na.rm = T)
  stats <- data.frame(Ref = c("Mean", "Median"),
                      vals = c(mean, median),
                      stringsAsFactors = FALSE)

  ggplot(data=subset(Txc, x<= mean+2*sd), aes(x=x,y=..count../sum(..count..)))+
    geom_histogram(binwidth = 1,alpha=.5, position="identity")+
    labs(x=paste("Nbr contaminated units (Tx-tx|ty)\n", "Mean= ", round(mean,digits = 6),", ",
                 "Median= ",round(median,digits = 6), ", ", "sd= ",round(sd,digits = 6)), y="pdf")+
    geom_vline(aes(xintercept=median,
                   color="median"), linetype="solid",
               size=1) +
    geom_vline(aes(xintercept=mean,
                   color="mean"), linetype="solid",
               size=1) +
    scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
    theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
    ggtitle("Posterior: (Tx-tx|ty)") +
    theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                        legend.justification=c(1,0),
                                                        legend.position=c(.8, .75),
                                                        legend.background = element_blank(),
                                                        legend.key = element_blank())
}else{
  txcl =subset(tmp, !is.na(tx))
  tt = sqldf("select *, count(x) as feq from Txc group by
           x")
  tt$feq = tt$feq/sum(tt$feq)

  tt1 = sqldf("select *, count(tx) as feq1 from txcl group by
           tx")
  tt1$feq1 = tt1$feq1/sum(tt1$feq1)

  x1 <- tt$x
  p1 <- tt$feq
  x2 <- -tt1$tx
  p2 <- tt1$feq1

  Tx_txc = conv(x1,p1,x2,p2)
  Tx_txc = as.data.frame(Tx_txc)
  Tx_txc = subset(Tx_txc,V1>0)
  Tx_txc$V2 = Tx_txc$V2/sum(Tx_txc$V2)

  mean = sum(Tx_txc$V1*Tx_txc$V2)
  sd =sqrt(sum(Tx_txc$V2* (Tx_txc$V1-mean)^2))

  i<-1
  tmpMed =0
  while (i < N) {
    tmpMed = tmpMed+Tx_txc$V2[i]
    if (tmpMed>0.5){
      median = Tx_txc$V1[i]
      i <- N+1
    } else{
      i =i+1
    }
  }

  # summary stats
  stats <- data.frame(Ref = c("Mean", "Median"),
                      vals = c(mean, median),
                      stringsAsFactors = FALSE)

  if (mean >median){
    trs = ifelse(mean>1,mean+2*sd,mean+5*sd)
  }else{
    trs = ifelse(median>1,median+2*sd,median+5*sd)
  }

  ggplot(data=subset(Tx_txc, V1< trs), aes(x=V1,y=V2))+
    geom_col(fill = "grey")+
    labs(x=paste("Nbr contaminated units (Tx-tx|ty)\n", "Mean= ", round(mean,digits = 6), ", ",
                 "Median= ",round(median,digits = 6),", ",
                 "sd= ",round(sd,digits = 6)),y="pdf")+
    geom_vline(aes(xintercept=median,
                   color="median"), linetype="solid",
               size=1) +
    geom_vline(aes(xintercept=mean,
                   color="mean"), linetype="solid",
               size=1) +
    scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))+
    theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
    ggtitle("Posterior: (Tx-tx|ty)") +
    theme(plot.title = element_text(hjust = 0.5))+theme(legend.title = element_text(size=12, color = "salmon", face="bold"),
                                                        legend.justification=c(1,0),
                                                        legend.position=c(.8, .75),
                                                        legend.background = element_blank(),
                                                        legend.key = element_blank())
}



