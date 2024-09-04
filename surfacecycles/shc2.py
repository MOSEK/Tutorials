# Optimization of cycles on surfaces

import mosek                        
from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np 
import sys, io, math, plyfile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#A model of a surface (or a more general 2D complex in 3D)
class Surface:
    #Initialize from plydata using edge weights given by metric
    def __init__(self, plydata, metric):

        ### Initialize basic data
        self.v = plydata.elements[0].count                              # number of vertices
        self.t = plydata.elements[1].count                              # number of triangles
        self.coo = plydata['vertex']                                    # coordinates of vertices in 3D
        self.xc, self.yc, self.zc = zip(*self.coo)
        self.tri = [tuple(sorted(f[0])) for f in plydata['face']]       # list of triangles, as sorted triples of vertices
        self.edg = list(set([tuple(sorted([f[i],f[j]]))                 # list of edges, as sorted pairs of vertices
            for f in self.tri for i,j in [(0,1),(0,2),(1,2)]]))
        self.e = len(self.edg)                                          # number of edges
        self.wgh = [metric(self.coo[f[0]],self.coo[f[1]])               # weights of edges
            for f in self.edg]
        
        ### Homological algebra
        # Construct the sparse matrix D1
        D1 = [ (self.edg[i][k], i, sgn) for i in range(self.e) for k,sgn in [(1,+1),(0,-1)] ]
        s1, s2, val = zip(*D1)
        self.D1 = Matrix.sparse(self.v, self.e, list(s1), list(s2), list(val))
        # Construct the sparse matrix D2
        D2 = [ (self.edg.index((self.tri[i][k],self.tri[i][l])), i, sgn) 
            for i in range(self.t) for k,l,sgn in [(1,2,+1),(0,2,-1),(0,1,+1)] ]
        s1, s2, val = zip(*D2)
        self.D2 = Matrix.sparse(self.e, self.t, list(s1), list(s2), list(val))
        
    #Plot the surface and a number of chains
    def plot(self, chains=[], colors=[]):
        fig = plt.figure()
        ax = plt.axes(projection='3d',computed_zorder=False)
        ax.plot_trisurf(self.xc, self.yc, self.zc, triangles=self.tri, linewidth=0.5, alpha=1, edgecolor='green')
        i = 0
        for C in chains:
            for j in range(self.e):
                if abs(C[j])>0.0001:
                    # Extract the edge and plot with intensity depending on its coefficient in C
                    xc,yc,zc = zip(*[ self.coo[self.edg[j][0]], self.coo[self.edg[j][1]] ])
                    ax.plot(xc, yc, zc, color=colors[i], alpha=min(1,abs(C[j])), zorder=1)
            i += 1
        plt.axis('off')
        plt.show()

    #Find the shortest chain homologous to a given chain C 
    def short(self, C):
        ### Set up a MOSEK Fusion model for solving shortest homologous cycles
        with Model("short") as M:
            x    = M.variable(self.e, Domain.unbounded())                # the chain we are looking for
            xabs = M.variable(self.e, Domain.unbounded())                # absolute value of x
            y    = M.variable(self.t, Domain.unbounded())                # a 2-chain whose boundary is x-C

            # -xabs <= x <= xabs
            M.constraint(xabs >= x)
            M.constraint(xabs >= -x)

            # x - c = D2*y, so x-c is a boundary
            M.constraint(x - C == self.D2 @ y)

            # min weighted 1-norm of x
            M.objective(ObjectiveSense.Minimize, xabs.T @ self.wgh)

            # solve
            M.solve()
            return x.level()

#Two ways of assigning edge weights
def euclideanM(x, y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(3)]))
def graphM(x,y):
    return 1

#Normalize a chain so that l_\infty norm is 1
def normalize(C):
    return C/max(abs(C))

#Fetching examples predefined for this presentation
def getExample(name):
    return plyfile.PlyData.read(open(name, 'rb'))

#Find the basis for the nullspace of a matrix
def nullspace(A):
    u, s, vh = np.linalg.svd(A)
    nnz = (s >= 1e-13).sum()
    ns = vh[nnz:].conj().T
    return ns, ns.shape[1]

def homology(S):
    return nullspace(np.reshape(
            np.concatenate((S.D1.getDataAsArray(), S.D2.transpose().getDataAsArray())),
            [S.v+S.t, S.e]))

s = Surface(getExample('torus.ply'), graphM)

#Find the first Betti number and homology generators
generators, betti1 = homology(s)
print("B_1 = {0}".format(betti1))

G = [ normalize(generators[:,i]) for i in range(betti1) ]
GS= [ normalize(s.short(g)) for g in G ]
for i in range(betti1):
    s.plot([G[i], GS[i]], ['yellow', 'red'])

