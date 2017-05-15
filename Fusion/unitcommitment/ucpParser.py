'''
File ucpPar.py
Copyright: Mosek ApS
Content: Parser of the Unit Commitment problem descriptions. Thermal instances only. Demonstration purpose only.
'''
import numpy as np

def loadModel(filename, m=1):
    with open(filename) as f:
         f.readline()
         T = int(f.readline().split()[1])
         N = int(f.readline().split()[1])
         demand = []
         a, b, c = [], [], []
         pmin, pmax = [], []
         p0, u0, l0 = [], [], []
         rup, rdown = [], []
         sc, fx = [], []
         mup, mdown = [], []
         for _ in range(6): f.readline()
         for x1 in range(int(f.readline().split()[1])):
            demand += [float(x) for x in f.readline().split()]
         for _ in range(3): f.readline()
         for _ in range(N):
            desc = f.readline().split()
            ramp = f.readline().split()
            a.append(float(desc[1]))
            b.append(float(desc[2]))
            c.append(float(desc[3]))
            pmin.append(float(desc[4]))
            pmax.append(float(desc[5]))
            uptime = int(desc[6])
            l0.append(abs(uptime))
            u0.append(1 if uptime>0 else 0)
            p0.append(float(desc[15]))
            mup.append(int(desc[7]))
            mdown.append(int(desc[8]))
            rup.append(max(float(ramp[1]), float(desc[4])))
            rdown.append(max(float(ramp[2]), float(desc[4])))
            fx.append(float(desc[14]))
            sc.append(float(desc[13])*int(desc[12]))

         return T*m, N, demand*m, pmin, pmax, rdown, rup, mup, mdown, a, b, c, sc, fx, p0, u0, l0
