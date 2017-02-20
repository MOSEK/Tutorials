# Optimization of cycles on surfaces

import mosek                             # Mosek version >= 8.0.0.55
from mosek.fusion import *
import numpy as np 
import sys, io, math, urllib2, plyfile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#A model of a surface (or a more general 2D complex in 3D)
class Surface(mosek.fusion.Model):
	#Initialize from plydata using edge weights given by metric
	def __init__(self, plydata, metric):
		super(Surface, self).__init__()

		### Initialize basic data
		self.v = plydata.elements[0].count								# number of vertices
		self.t = plydata.elements[1].count								# number of triangles
		self.coo = plydata['vertex']									# coordinates of vertices in 3D
		self.xc, self.yc, self.zc = zip(*self.coo)
		self.tri = [tuple(sorted(f[0])) for f in plydata['face']]		# list of triangles, as sorted triples of vertices
		self.edg = list(set([tuple(sorted([f[i],f[j]])) 				# list of edges, as sorted pairs of vertices
			for f in self.tri for i,j in [(0,1),(0,2),(1,2)]]))
		self.e = len(self.edg)											# number of edges
		self.wgh = [metric(self.coo[f[0]],self.coo[f[1]]) 				# weights of edges
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
		
		### Set up a MOSEK Fusion model for solving shortest homologous cycles
		x    = self.variable(self.e, Domain.unbounded())				# the chain we are looking for
		xabs = self.variable(self.e, Domain.unbounded())				# absolute value of x
		y    = self.variable(self.t, Domain.unbounded())				# a 2-chain whose boundary is x-C
		C    = self.variable(self.e, Domain.unbounded())				# the input chain

		# -xabs <= x <= xabs
		self.constraint(Expr.add(xabs, x), Domain.greaterThan(0))
		self.constraint(Expr.sub(xabs, x), Domain.greaterThan(0))

		# x - c = D2*y, so x-c is a boundary
		self.constraint(Expr.sub(Expr.sub(x, C), Expr.mul(self.D2, y)), Domain.equalsTo(0))
	
		# min weighted 1-norm of x
		self.objective(ObjectiveSense.Minimize, Expr.dot(self.wgh, xabs))

		# Save the model for later cloning
		self.x, self.c, self.C = x, self.constraint(C, Domain.equalsTo(0)), 0

	#Plot the surface and a number of chains
	def plot(self, chains=[], colors=[]):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot_trisurf(self.xc, self.yc, self.zc, triangles=self.tri, linewidth=0.5, alpha=1.0, edgecolor='green')
		i = 0
		for C in chains:
			for j in range(self.e):
				if abs(C[j])>0.0001:
					# Extract the edge and plot with intensity depending on its coefficient in C
					xc,yc,zc = zip(*[ self.coo[self.edg[j][0]], self.coo[self.edg[j][1]] ])
					ax.plot(xc, yc, zc, color=colors[i], alpha=min(1,abs(C[j])))
			i += 1
		plt.axis('off')
		plt.show()

	#Find the shortest chain homologous to a given chain C 
	def short(self, C):
		# We add a constraint c=C to the model
		# This is a little hack, because we must remember the old C
		self.c.add(self.C-C)
		self.C = C
		self.solve()
		return self.x.level()

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
	if name=='mug.ply':
		return plyfile.PlyData.read(urllib2.build_opener().open(urllib2.Request(
			'http://people.sc.fsu.edu/~jburkardt/data/ply/mug.ply')))
	else:
		return plyfile.PlyData.read(io.open('/home/aszek/CODE/ply/'+name, 'r'))
def getCycleExample(name):
	return np.load(np.DataSource().open('/home/aszek/CODE/ply/'+name))

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

#First example - a cycle on a mug and its shortest homologous one
s = Surface(getExample('surface.ply'), graphM)
s.plot([getCycleExample('surface-C1.npy'), getCycleExample('surface-C2.npy'), getCycleExample('surface-C3.npy')], ['red','yellow','violet'])

#Find the first Betti number and homology generators
generators, betti1 = homology(s)
print("B_1 = ", betti1)

G = [ normalize(generators[:,i]) for i in range(betti1) ]
GS= [ normalize(s.short(g)) for g in G ]
for i in range(betti1):
	s.plot([G[i], GS[i]], ['yellow', 'red'])

