import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Optimization of cycles on surfaces""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this notebook we use MOSEK Fusion to solve some geometric optimization problems on surfaces. The code demonstrates setting up a linear optimization problem, extending the `Model` class and shows the practical difference between interior-point and basic solution.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Requirements""")
    return


@app.cell
def _():
    # '%matplotlib notebook' command supported automatically in marimo
    import mosek                             
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
    import mosek.fusion.pythonic
    import numpy as np 
    import sys, io, math, plyfile
    import matplotlib.pyplot as plt

    #Fetching examples predefined for this presentation
    def getExample(name):
        return plyfile.PlyData.read(open(name, 'rb'))
    return Domain, Matrix, Model, ObjectiveSense, getExample, math, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### References
    <a href="https://pypi.python.org/pypi/plyfile">`plyfile`</a> is a package for reading the descriptions of surfaces in <a href="https://en.wikipedia.org/wiki/PLY_(file_format)">PLY format</a>. It can be installed with `pip install plyfile`. You will also need some data files with examples used in this presentation. The complete code and additional files are available from <a href="https://github.com/MOSEK/Tutorials/tree/master/surfacecycles">GitHub</a>.

    One of the models used in this tutorial comes from <a href="http://people.sc.fsu.edu/~jburkardt/data/ply/ply.html">John Burkardt's</a> library of PLY files and the other ones were generated using <a href="http://www.cgal.org/">CGAL</a>. A comprehensive survey of topological optimization problems can be found in the <a href="http://jeffe.cs.illinois.edu/pubs/pdf/optcycles.pdf">survey paper</a> by Jeff Erickson. The model used in this tutorial comes from the paper <a href="https://arxiv.org/abs/1001.0338">Optimal Homologous Cycles, Total Unimodularity, and Linear Programming</a> by Dey, Hirani and Krishnamoorthy. The experts in the field are kindly asked to fill in some vague statements in the following text with the precise algebro-topological language on their own.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A crash-course in homotopy
    A triangulated surface is a mesh of 2D triangles in 3D that fit together to form a surface. The vertices and edges of a triangulation form a graph. The *cycles* in that graph can be used to detect interesting *topological features*. For an example of what that means, see the three loops on the surface below:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<img src="https://raw.githubusercontent.com/MOSEK/Tutorials/master/surfacecycles/surface1.png" alt="surface1" width="400">""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now imagine that each loop is a rubber band that can be stretched in a continuous way and slide on the surface. Then the violet loop is not very interesting - it lies in a small patch of the surface and if we start shrinking it, it will collapse to a point (in other words, it is *contractible*). The yellow loop is different - it circles around a hole in the surface, and no matter how much we deform it, it will not disappear. We say this loop *detects a feature* - in this case the feature is one of the holes. The red loop has the same property. Moreover, the yellow and red loops are *homotopic* - each of them can be continuosly moulded into the second one - i.e. they detect the same feature (hole).

    In some applications it is useful to have *small* representations of topological features, in this case the shortest possible loops in a given homotopy class. There are at least two reasonable ways of measuring the length of a path on a surface: with the Euclidean metric on the edges, or just counting the number of edges.
    """
    )
    return


@app.cell
def _(math):
    def euclideanM(x, y):
        return math.sqrt(sum([(x[_i] - y[_i]) ** 2 for _i in range(3)]))

    def graphM(x, y):
        return 1
    return (euclideanM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## First problem: shortest equivalent representation of a cycle
    Suppose the surface has $v$ vertices, $e$ edges and $t$ triangles. It is convenient to represent it by *boundary matrices*, which are just variants of the <a href="https://en.wikipedia.org/wiki/Incidence_matrix">incidence matrix of a graph</a>. Specifically, let $V(i)$, $E(ij)$ and $T(ijk)$ denote the vertex, edge and triangle with the given vertex set. Then we have a linear map $D_1:\mathbb{R}^e\to\mathbb{R}^v$ given on the basis vectors by:

    $$D_1(\mathbf{e}_{E(ij)})=\mathbf{e}_{V(j)}-\mathbf{e}_{V(i)}$$

    and we identify $D_1\in \mathbb{R}^{v\times e}$ with the matrix of this map. Similarly, we can record the incidence between triangles and edges by $D_2:\mathbb{R}^t\to\mathbb{R}^e$:

    $$D_2(\mathbf{e}_{T(ijk)})=\mathbf{e}_{E(jk)}-\mathbf{e}_{E(ik)}+\mathbf{e}_{E(ij)}$$

    and let $D_2\in \mathbb{R}^{e\times t}$ be its matrix. We refer to <a href="https://en.wikipedia.org/wiki/Simplicial_homology">other sources</a> for more details. Note that matrices $D_1,D_2$ are extremely sparse - they have just two, resp. three, nonzeros in each column.

    ### Shortest homologous cycle problem
    A *chain* (more precisely, *1-chain*) is just a vector $c\in \mathbb{R}^e$, that is an assignment of a real coefficient to every edge. If $c$ is an actual path or cycle in the graph, then these coefficients are $\pm1$ for edges in the cycle, and $0$ otherwise, but more general chains are allowed. We will mostly concentrate on cycles.

    We can now formulate the following problem: given a cycle $c\in\mathbb{R}^e$, find the shortest cycle that detects the same topological features as $c$ (strictly speaking, the shortest cycle *homologous* to $c$). Suppose $w_1,\ldots,w_e$ are the weights (lengths) of the edges.

    $$
    \begin{array}{ll}
    \textrm{minimize}   & \sum_{i=1}^e w_i|x_i| \\
    \textrm{subject to} & x-c=D_2y \\
                        & x\in\mathbb{R}^e, y\in\mathbb{R}^t
    \end{array}
    $$

    ### Implementation
    This optimization problem can easily be formulated as a linear program. Below is the implementation of this program in a class `Surface`, which extends a Mosek Fusion `Model`.
    """
    )
    return


@app.cell
def _(Domain, Matrix, Model, ObjectiveSense, plt):
    class Surface:

        def __init__(self, plydata, metric):
            self.v = plydata.elements[0].count
            self.t = plydata.elements[1].count
            self.coo = plydata['vertex']
            self.xc, self.yc, self.zc = zip(*self.coo)
            self.tri = [tuple(sorted(f[0])) for f in plydata['face']]
            self.edg = list(set([tuple(sorted([f[_i], f[j]])) for f in self.tri for _i, j in [(0, 1), (0, 2), (1, 2)]]))
            self.e = len(self.edg)
            self.wgh = [metric(self.coo[f[0]], self.coo[f[1]]) for f in self.edg]
            d1 = [(self.edg[_i][k], _i, sgn) for _i in range(self.e) for k, sgn in [(1, +1), (0, -1)]]
            s1, s2, val = zip(*d1)
            self.D1 = Matrix.sparse(self.v, self.e, list(s1), list(s2), list(val))
            d2 = [(self.edg.index((self.tri[_i][k], self.tri[_i][l])), _i, sgn) for _i in range(self.t) for k, l, sgn in [(1, 2, +1), (0, 2, -1), (0, 1, +1)]]
            s1, s2, val = zip(*d2)
            self.D2 = Matrix.sparse(self.e, self.t, list(s1), list(s2), list(val))

        def plot(self, chains=[], colors=[]):
            fig = plt.figure()
            ax = plt.axes(projection='3d', computed_zorder=False)
            ax.plot_trisurf(self.xc, self.yc, self.zc, triangles=self.tri, linewidth=0.5, alpha=0.5, edgecolor='green')
            _i = 0
            for C in chains:
                for j in range(self.e):
                    if abs(C[j]) > 0.0001:
                        xc, yc, zc = zip(*[self.coo[self.edg[j][0]], self.coo[self.edg[j][1]]])
                        ax.plot(xc, yc, zc, color=colors[_i], alpha=min(1, abs(C[j])), zorder=1)
                _i += 1
            plt.axis('off')
            plt.show()

        def short(self, C):
            with Model('short') as M:
                x = M.variable(self.e, Domain.unbounded())
                xabs = M.variable(self.e, Domain.unbounded())
                y = M.variable(self.t, Domain.unbounded())
                M.constraint(xabs >= x)
                M.constraint(xabs >= -x)
                M.constraint(x - C == self.D2 @ y)
                M.objective(ObjectiveSense.Minimize, xabs.T @ self.wgh)
                M.solve()
                return x.level()
    return (Surface,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The initialization part processes the PLY description of the surface and initializes the sparse boundary matrices. Every time we optimize for the shortest cycle a new Fusion model is constructed and solved from the input data.

    ### Example
    We predefined the yellow cycle on the mug, and Mosek finds the red cycle - its shortest topological equivalent.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<img src="https://raw.githubusercontent.com/MOSEK/Tutorials/master/surfacecycles/mug.png" alt="mug" width="400">""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Indeed, if we again imagine the yellow loop is made out of rubber, then once it starts contracting it will reach its minimum length when it sits tightly around the mug's handle.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Detecting features
    OK, but what if we don't have any predefined cycle, and we would simply like to visualize all topological features on a given input surface using shortest possible loops? In general, this is a hard problem, for which some algorithms are known on surfaces. However, we can use our model to try a more pedestrian approach, which is not perfect, but often good enough for visualization.

    Without going into too much algebra, the space of features on the surface can be identified with the nullspace of the matrix $$A=\left[\begin{array}{l} D_1 \\ D_2^T\end{array}\right]$$ (for the experts, we are now computing first homology with real coefficients). This can be easily found using singular value decomposition:
    """
    )
    return


@app.cell
def _(np):
    def nullspace(A):
        u, s, vh = np.linalg.svd(A)
        nnz = (s >= 1e-13).sum()
        ns = vh[nnz:].conj().T
        return (ns, ns.shape[1])

    def homology(S):
        return nullspace(np.reshape(np.concatenate((S.D1.getDataAsArray(), S.D2.transpose().getDataAsArray())), [S.v + S.t, S.e]))
    return (homology,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For an orientable surface the dimension of the nullspace of $A$ equals $2g$, where $g$ is the <a href="https://en.wikipedia.org/wiki/Genus_(mathematics)">genus</a>, or the number of holes. For example, the torus has one hole and two essential features: the parallel and the meridian cycle.

    Since the SVD decomposition is oblivious to the underlying geometry, the chains it returns are typically dense vectors with real coordinates, not very useful for visualization. However, we can replace them with their shortest equivalents and see if they are any better. It turns out in practice they often are. To make this easier to see we also normalize so that the maximal coefficient of each chain is 1.
    """
    )
    return


@app.cell
def _(Surface, euclideanM, getExample, homology):
    def normalize(C):
        return C / max(abs(C))
    S = Surface(getExample('torus.ply'), euclideanM)
    _generators, _betti1 = homology(S)
    _G = [normalize(_generators[:, _i]) for _i in range(_betti1)]
    GS = [normalize(S.short(g)) for g in _G]
    for _i in range(_betti1):
        S.plot([_G[_i], GS[_i]], ['yellow', 'red'])
    return (normalize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The two projections of the torus do not reveal where the holes are. In yellow we see the dense cycles found by SVD.  Their shortest homologous red versions make this a bit more clear. For example, the coil-shaped part of the first cycle clearly detects the hole in the middle of the torus.

    Here is an example of this on a more complicated surface.
    """
    )
    return


@app.cell
def _(Surface, euclideanM, getExample, homology, normalize):
    def _():
        S = Surface(getExample('surface.ply'), euclideanM)
        _generators, _betti1 = homology(S)
        print('Number of features: {0}'.format(_betti1))
        _G = [normalize(_generators[:, _i]) for _i in range(_betti1)]
        for _i in [2, 5]:
            S.plot([_G[_i], normalize(S.short(_G[_i]))], ['yellow', 'red'])
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. The **MOSEK** logo and name are trademarks of <a href="http://mosek.com">Mosek ApS</a>. The code is provided as-is. Compatibility with future release of **MOSEK** or the `Fusion API` are not guaranteed. For more information contact our [support](mailto:support@mosek.com).""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
