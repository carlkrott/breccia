# Lloyd-Max codebooks for N(0,1); self-check vs Octave's int8 LUT3/LUT4.
import math
phi=lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi); Phi=lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
def lm(bits, it=20000):
    n=1<<bits; lv=[-3+6*(i+.5)/n for i in range(n)]
    for _ in range(it):
        t=[-math.inf]+[(lv[i]+lv[i+1])/2 for i in range(n-1)]+[math.inf]
        lv=[(phi(t[i])-phi(t[i+1]))/(Phi(t[i+1])-Phi(t[i])) for i in range(n)]
    t=[-math.inf]+[(lv[i]+lv[i+1])/2 for i in range(n-1)]+[math.inf]
    # MSE = 1 - sum p_i * c_i^2 for the centroid condition
    mse=1-sum((Phi(t[i+1])-Phi(t[i]))*lv[i]**2 for i in range(n))
    return lv,mse
L3=[-127,-79,-45,-14,14,45,79,127]; L4=[-126,-95,-74,-58,-43,-30,-18,-6,6,18,30,43,58,74,95,126]
for b,ref,m in ((3,L3,59),(4,L4,46)):
    lv,mse=lm(b,3000); err=max(abs(a*m-r) for a,r in zip(lv,ref)); print(b,"mse",round(mse,5),"max int8 err",round(err,2)); assert err<1.0
for b in (5,6):
    lv,mse=lm(b); print(b,"mse",round(mse,6))
    print(f"const float LUT{b}[{1<<b}] = float[{1<<b}](" + ", ".join(f"{x:.6f}" for x in lv) + ");")
