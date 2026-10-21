import numpy as np
def gp(t, shape, ct, A):
    t=np.maximum(t,1e-6); return A*np.exp(-t/ct)*((t*np.e)/((shape-1)*ct))**(shape-1)*(((shape-1)/t)-1/ct)
def euler_sim(drift_fn, alpha, beta, n, dt=1e-4, s=0.1, tmax=3.0, seed=0):
    """Independent Euler-Maruyama simulator: drift_fn(t) gives drift at decision time t."""
    rng=np.random.default_rng(seed)
    x=np.full(n,(beta-0.5)*alpha); done=np.zeros(n,bool); rt=np.full(n,np.nan); ch=np.zeros(n,int)
    B=alpha/2; t=0.0; sq=s*np.sqrt(dt)
    while t<tmax and not done.all():
        idx=~done
        x[idx]+= drift_fn(t)*dt + sq*rng.standard_normal(idx.sum())
        t+=dt
        up=idx&(x>=B); lo=idx&(x<=-B)
        rt[up]=t; ch[up]=1; rt[lo]=t; ch[lo]=0; done|=up|lo
    return rt, ch
