import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Equilibrium of a system of weights connected by strings/springs

    In this notebook we show how to solve the following problem: Find the equlibrium of a system of masses connected by a system of strings, with some masses being assigned fixed coordinates (attached to the wall, say). See the next picture.

    ![](basic.png)

    Suppose we have $n$ masses with weights $w_1,\ldots,w_n$, and the length of the string between $i$ and $j$ is $\ell_{ij}$ for some set $L$ of pairs of indices $(i,j)$ (we assume $\ell_{ij}$ is not defined if there is no connection). The strings themselves have no mass. We also have a set $F$ of indices such that the $i$-th point is fixed to have coordinates $f_i$ if $i\in F$. The equilibrium of the system is a configuration which minimizes potential energy. With this setup we can write our problem as:

    \begin{equation}
    \begin{array}{ll}
    minimize & g\cdot \sum_i w_ix_i^{(2)} \\
    s.t.     & \|x_i-x_j\|\leq \ell_{ij},\ ij\in L \\
             & x_i = f_i,\ i\in F
    \end{array}
    \end{equation}

    where $x\in (\mathbf{R}^n)^2$, $x_i^{(2)}$ denotes the second (vertical) coordinate of $x_i$ and $g$ is the gravitational constant.

    Here is a sample problem description.
    """
    )
    return


@app.cell
def _():
    w = [0.0, 1.1, 2.2, 0.0, 2.1, 2.2, 0.2]
    l = {(0,1): 1.0, (1,2): 1.0, (2,3): 1.0, (1,4): 1.0, (4,5): 0.3, (5,2): 1.0, (5,6): 0.5, (1,3): 8.0}
    f = {0: (0.0,1.0), 3: (2.0,1.0)}
    g = 9.81
    return f, g, l, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we can formulate the problem using Mosek Fusion:""")
    return


@app.cell
def _():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus, SolutionType, ProblemStatus
    import mosek.fusion.pythonic

    def stringModel(w, l, f, g):
        _n, m = (len(w), len(l))
        starts = [lKey[0] for lKey in l.keys()]
        ends = [lKey[1] for lKey in l.keys()]
        M = Model('strings')
        x = M.variable('x', [_n, 2])
        A = Matrix.sparse(m, _n, list(range(m)) + list(range(m)), starts + ends, [1.0] * m + [-1.0] * m)
        c = M.constraint('c', Expr.hstack(Expr.constTerm(list(l.values())), A @ x), Domain.inQCone())
        for i in f:
            M.constraint(Var.flatten(x[i, :]) == list(f[i]))
        M.objective(ObjectiveSense.Minimize, g * (x[:, 1].T @ w))
        M.solve()
        if M.getProblemStatus(SolutionType.Interior) == ProblemStatus.PrimalAndDualFeasible:
            return (x.level().reshape([_n, 2]), c.dual().reshape([m, 3]))
        else:
            return (None, None)
    return (
        Domain,
        Expr,
        Matrix,
        Model,
        ObjectiveSense,
        ProblemStatus,
        SolutionType,
        Var,
        stringModel,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here is a quick description of how we use vectorization to deal with all the conic constraints in one go. The matrix $A$ is the incidence matrix between the masses and the strings, with coefficients $+1, -1$ for the two endpoints of each string. It is chosen so that the product $Ax$ has rows of the form

    $$
    (x_i^{(1)} - x_j^{(1)}, x_i^{(2)} - x_j^{(2)})
    $$

    for all pairs $i,j$ for which $\ell_{ij}$ is bounded. Stacking the values of $\ell$ in the left column produces a matrix with each row of the form

    $$
    (\ell_{ij}, x_i^{(1)} - x_j^{(1)}, x_i^{(2)} - x_j^{(2)})
    $$

    and a conic constraint is imposed on all the rows, as required.

    The objective and linear constraints show examples of slicing the variable $x$.

    The function returns the coordinates of the masses and the values of the dual conic variables. A zero dual value indicates that a particular string is hanging loose, and a nonzero value means it is fully stretched. 

    All we need now is to define a display function and we can look at some plots.
    """
    )
    return


@app.function
def display(x, c, d):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], color='r')
    for i in range(len(c)):
        col = 'b' if c[i][0] > 0.0001 else 'b--'
        ax.plot([x[d[i][0]][0], x[d[i][1]][0]], [x[d[i][0]][1], x[d[i][1]][1]], col)
    ax.axis('equal')
    plt.show()


@app.cell
def _(f, g, l, stringModel, w):
    def _():
        x, c = stringModel(w, l, f, g)
        if x is not None:
            return display(x, c, list(l.keys()))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How about we find a discrete approximation to the [catenary](https://en.wikipedia.org/wiki/Catenary):""")
    return


@app.cell
def _(stringModel):
    def _():
        _n = 1000
        w_1 = [1.0] * _n
        l_1 = {(i, i + 1): 1.0 / _n for i in range(_n - 1)}
        f_1 = {0: (0.0, 1.0), _n - 1: (0.7, 1.0)}
        g_1 = 9.81
        x, c = stringModel(w_1, l_1, f_1, g_1)
        if x is not None:
            return display(x, c, list(l_1.keys()))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also have more suspension points and more complicated shapes:""")
    return


@app.cell
def _(stringModel):
    def _():
        _n = 20
        w_2 = [1.0] * _n
        l_2 = {(i, i + 1): 0.09 for i in range(_n - 1)}
        l_2.update({(5, 14): 0.3})
        f_2 = {0: (0.0, 1.0), 13: (0.5, 0.9), 17: (0.7, 1.1)}
        g_2 = 9.81
        x, c = stringModel(w_2, l_2, f_2, g_2)
        if x is not None:
            return display(x, c, list(l_2.keys()))


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Duality and feasibility

    The dual problem is as follows:

    \begin{equation}
    \begin{array}{ll}
    maximize & -\sum_{ij\in L}\ell_{ij}y_{ij} - \sum_{i\in F}f_i\circ z_i\\
    s.t.     & y_{ij}\geq \|v_{ij}\|,\ ij\in L \\
             & \sum_{j~:~ij\in L} v_{ij}\mathrm{sgn}_{ij} + \left(\begin{array}{c}0\\ gw_i\end{array}\right) +z_i = 0, \ i=1,\ldots,n
    \end{array}
    \end{equation}

    where $\mathrm{sgn}_{ij}=+1$ if $i>j$ and $-1$ otherwise and $\circ$ is the dot product. The variables are $(y_{ij},v_{ij})\in \mathbf{R}\times\mathbf{R}^2$ for $ij\in L$ and $z_i\in\mathbf{R}^2$ for $i\in F$ (we assume $z_i=0$ for $i\not\in F$).

    Obviously (!) the linear constraints describe the equilibrium of forces at every mass. The ingredients are: the vectors of forces applied through adjacent strings ($v_{ij}$), gravity, and the attaching force holding a fixed point in its position. By proper use of vectorization this is much easier to express in Fusion than it looks:
    """
    )
    return


@app.cell
def _(Domain, Expr, Matrix, Model, ObjectiveSense):
    def dualStringModel(w, l, f, g):
        _n, m = (len(w), len(l))
        starts = [lKey[0] for lKey in l.keys()]
        ends = [lKey[1] for lKey in l.keys()]
        M = Model('dual strings')
        x = M.variable(Domain.inQCone(m, 3))
        y = x[0:m, 0]
        v = x[0:m, 1:3]
        z = M.variable([_n, 2])
        for i in range(_n):
            if i not in f:
                M.constraint(z[i, :] == 0)
        B = Matrix.sparse(m, _n, list(range(m)) + list(range(m)), starts + ends, [1.0] * m + [-1.0] * m).transpose()
        w2 = Matrix.sparse(_n, 2, range(_n), [1] * _n, [-wT * g for wT in w])
        M.constraint(B @ v + z == w2)
        fM = Matrix.sparse(_n, 2, list(f.keys()) + list(f.keys()), [0] * len(f) + [1] * len(f), [pt[0] for pt in f.values()] + [pt[1] for pt in f.values()])
        M.objective(ObjectiveSense.Maximize, -Expr.dot(list(l.values()), y) - Expr.dot(fM, z))
        M.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let us quickly discuss the possible situations regarding feasibility:

    * The system has an equilibrium — the problem is **primal feasible** and **dual feasible**.
    * The strings are too short and it is impossible to stretch the required distance between fixed points — the problem is **primal infeasible**.
    * The system has a component that is not connected to any fixed point, hence some masses can keep falling down indefinitely, causing the problem **primal unbounded**. Clearly the forces within such component cannot be balanced, so the problem is **dual infeasible**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Springs

    We can extend this to consider infinitely strechable springs instead of fixed-length strings connecting the masses. The next model appears in [Applications of SOCP](http://stanford.edu/~boyd/papers/pdf/socp.pdf) by Lobo, Boyd, Vandenberghe, Lebret. We will now interpret $\ell_{ij}$ as the base length of the spring and assume that the elastic potential energy stored in the spring at length $x$ is 

    $$
    E_{ij}=\left\{\begin{array}{ll}0 & x\leq \ell_{ij}\\ \frac{k}{2}(x-\ell_{ij})^2 & x>\ell_{ij}\end{array}\right.
    $$

    That leads us to consider the following second order cone program minimizing the total potential energy:

    \begin{equation}
    \begin{array}{ll}
    minimize & g\cdot \sum_i w_ix_i^{(2)} + \frac{k}{2}\sum_{ij\in L} t_{ij}^2 \\
    s.t.     & \|x_i-x_j\|\leq \ell_{ij}+t_{ij},\ ij\in L \\
             & 0\leq t_{ij},\ ij\in L \\
             & x_i = f_i,\ i\in F
    \end{array}
    \end{equation}

    If $t$ denotes the vector of $t_{ij}$ then using a rotated quadratic cone for $(1,T,t)$:

    $$
    2\cdot 1\cdot T\geq \|t\|^2
    $$

    will place a bound on $\frac12\sum t_{ij}^2$. We now have a simple extension of the first model.
    """
    )
    return


@app.cell
def _(
    Domain,
    Expr,
    Matrix,
    Model,
    ObjectiveSense,
    ProblemStatus,
    SolutionType,
    Var,
):
    def elasticModel(w, l, f, g, k):
        _n, m = (len(w), len(l))
        starts = [lKey[0] for lKey in l.keys()]
        ends = [lKey[1] for lKey in l.keys()]
        M = Model('strings')
        x = M.variable('x', [_n, 2])
        t = M.variable(m, Domain.greaterThan(0.0))
        T = M.variable(1)
        M.constraint(Expr.vstack(T, Expr.constTerm(1.0), t), Domain.inRotatedQCone())
        A = Matrix.sparse(m, _n, list(range(m)) + list(range(m)), starts + ends, [1.0] * m + [-1.0] * m)
        c = M.constraint('c', Expr.hstack(t + Expr.constTerm(list(l.values())), A @ x), Domain.inQCone())
        for i in f:
            M.constraint(Var.flatten(x[i, :]) == list(f[i]))
        M.objective(ObjectiveSense.Minimize, k * T + g * (x[:, 1].T @ w))
        M.solve()
        if M.getProblemStatus(SolutionType.Interior) == ProblemStatus.PrimalAndDualFeasible:
            return (x.level().reshape([_n, 2]), c.dual().reshape([m, 3]))
        else:
            return (None, None)
    return (elasticModel,)


@app.cell
def _(elasticModel):
    def _():
        _n = 20
        w_3 = [1.0] * _n
        l_3 = {(i, i + 1): 0.09 for i in range(_n - 1)}
        l_3.update({(5, 14): 0.3})
        f_3 = {0: (0.0, 1.0), 13: (0.5, 0.9), 17: (0.7, 1.1)}
        g_3 = 9.81
        k = 800
        x, c = elasticModel(w_3, l_3, f_3, g_3, k)
        if x is not None:
            return display(x, c, list(l_3.keys()))
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
