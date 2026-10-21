library(DMCfun)
P <- read.csv("benchmarks/r3_params.csv")
out <- list()
for (i in seq_len(nrow(P))) {
  p <- P[i,]
  s <- dmcSim(amp=p$amp, tau=p$tau, drc=p$drc, bnds=p$bnds, resMean=p$resMean, resSD=0, aaShape=p$aaShape,
              spBias=p$spBias, spDist=0, sigm=4, nTrl=400000, tmax=3000, rtMax=5000,
              pDelta=c(10,30,50,70,90), tDelta=1, printInputArgs=FALSE, printResults=FALSE, setSeed=TRUE, seedValue=i)
  sm <- s$summary; d <- s$delta
  out[[i]] <- data.frame(set=p$set, comp=c("comp","incomp"), rtCor=sm$rtCor, perErr=sm$perErr, rtErr=sm$rtErr,
     q10=c(d$meanComp[1],d$meanIncomp[1]), q30=c(d$meanComp[2],d$meanIncomp[2]), q50=c(d$meanComp[3],d$meanIncomp[3]),
     q70=c(d$meanComp[4],d$meanIncomp[4]), q90=c(d$meanComp[5],d$meanIncomp[5]))
}
write.csv(do.call(rbind,out), "benchmarks/r3_dmcfun_out.csv", row.names=FALSE)
cat("ok\n")
