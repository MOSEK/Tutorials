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
    Smallest sphere enclosing a set of points.
    ===========================

    The aim of this tutorial is two-fold

    1. Demostrate how to write a conic quadratic model in `Fusion` in a very simple and compact way.
    2. Show how and way the dual formulation may solved more efficiently.


    Our problem is the defined as:

    **Find the smallest sphere that encloses a set of** $k$ **points** $p_i \in \mathbb{R}^n$.
    """
    )
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import random

    def plot_points(p, p0=[], r0=0.):
        n,k= len(p0), len(p)

        plt.rc('savefig',dpi=120)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot([ p[i][0] for i in range(k)], [ p[i][1] for i in range(k)], 'b*')

        if len(p0)>0:
            ax.plot(  p0[0],p0[1], 'r.')
            ax.add_patch( mpatches.Circle( p0,  r0 ,  fc="w", ec="r", lw=1.5) )
        plt.grid()
        plt.show()

    n = 2
    k = 500

    p=  [ [random.gauss(0.,10.) for nn in range(n)] for kk in range(k)]

    plot_points(p)
    return p, plot_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The problem boils down to determine the sphere center $p_0\in \mathbb{R}^n$ and its radius $r_0\geq 0$, i.e.


    \begin{equation}
      \begin{aligned}
    \min \max_{i=1,\dots,k} \| p_0 - p_i\|_2 \\
      \end{aligned}
    \end{equation}

    The maximum in the objective function can be easily, i.e.

    \begin{equation}
      \begin{aligned}
    \min r_0 & & &\\
    s.t.& & &\\
    & r_0 \geq \| p_0 - p_i\|_2 ,& \quad & i=1,\ldots,k\\
    \end{aligned}
    \end{equation}

    The SOCP formulation reads

    \begin{equation}
      \begin{aligned}
    \min r_0 & & &\\
    s.t.& & &\\
    & \left[r_0,p_0 - p_i\right] \in Q^{(n+1)}, & \quad & i=1,\ldots,k.
    \end{aligned}
    \end{equation}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Before defining the constraints, we note that we can write


    \begin{equation}
    R_0 = \left( \begin{array}{c} r_0   \\ \vdots \\ r_0   \end{array} \right) \in \mathbb{R}^k          , \quad
    P_0 = \left( \begin{array}{c} p_0^T \\ \vdots \\ p_0^T \end{array} \right) \in \mathbb{R}^{k\times n}, \quad
    P   = \left( \begin{array}{c} p_1^T \\ \vdots \\ p_k^T \end{array} \right) \in \mathbb{R}^{k\times n}.
    \end{equation}

    so that 

    \begin{equation}
    \left[r_0,p_i - p_0\right] \in Q^{(n+1)},  \quad  i=1,\ldots,k.
    \end{equation}

    can be compactly expressed as 

    \begin{equation}
    \left[ R_0,P_0-P\right] \in \Pi Q^{(n+1)},
    \end{equation}

    that means, with a little abuse of notation, that each rows belongs to a quadratic cone of dimension $n+1$.


    Now we are ready for a compact implementation in `Fusion`:
    """
    )
    return


@app.cell
def _():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
    import mosek.fusion.pythonic
    import mosek as msk
    import sys


    def primal_problem(P):

        print(Model.getVersion())

        k= len(P)
        if k==0: return -1,[]

        n= len(P[0])

        with Model("minimal sphere enclosing a set of points - primal") as M:

            r0 = M.variable(1    , Domain.greaterThan(0.))
            p0 = M.variable([1,n], Domain.unbounded())

            R0 = Var.repeat(r0,k)
            P0 = Var.repeat(p0,k)

            M.constraint( Expr.hstack( R0, P0 - P ), Domain.inQCone())

            M.objective(ObjectiveSense.Minimize, r0)
            M.setLogHandler(sys.stdout)

            M.solve()

            return r0.level()[0], p0.level()
    return (
        Domain,
        Expr,
        Matrix,
        Model,
        ObjectiveSense,
        Var,
        primal_problem,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will also store the solver output in a file to use it later on. And then just solve the problem.""")
    return


@app.cell
def _(p, primal_problem):
    r0,p0 = primal_problem(p)

    print ("r0^* = ", r0)
    print ("p0^* = ", p0)
    return p0, r0


@app.cell
def _(p, p0, plot_points, r0):
    plot_points(p,p0,r0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dual Formulation 
    -----------------------

    The dual problem can be determined in few steps following basic rules. Introducing dual variables

    \begin{equation}
     Y = \left( \begin{array}{c} y_1^T\\ \vdots \\y_k  \end{array}\right), \quad z = \left( \begin{array}{c} z_1\\ \vdots \\z_k  \end{array}\right), 
    \end{equation}

    the dual is:

    \begin{aligned}
        \max & \left\langle P, Y \right\rangle \\
        & e_k^T z = 1\\
        & Y^T e_k = \mathbf{0}_n \\
        & \left[z_i , y_i\right] \in \mathcal{Q}^{n+1}\\
        & z_i\in \mathbb{R}, y_i\in \mathbb{R}^n,
    \end{aligned}

    where $e_k\in \mathbb{R}^k$ is a vector of all ones.

    The ``Fusion`` code is the following:
    """
    )
    return


@app.cell
def _(Domain, Expr, Matrix, Model, ObjectiveSense, Var, p, sys):
    def dual_problem(P):

        k= len(P)
        if k==0: return -1,[]

        n= len(P[0])

        with Model("minimal sphere enclosing a set of points - dual") as M:

            Y= M.variable([k,n], Domain.unbounded())
            z= M.variable(k    , Domain.unbounded())

            M.constraint(Expr.sum(z) == 1.0)

            e= [1.0 for i in range(k)]

            M.constraint(Y.T @ Matrix.ones(k,1) == 0)

            M.constraint( Var.hstack(z,Y), Domain.inQCone())

            M.objective( ObjectiveSense.Maximize, Expr.dot( P, Y )) 

            M.setLogHandler(sys.stdout)

            M.solve()

            return 

    dual_problem(p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can track and compare the progression of both the **primal** and **dual** problems from the log outputs. When we take a close look into the **primal objective (POBJ)** and **dual objective (DOBJ)** progression, we see that:  

    **The solver has got the same result doing exactly the same steps!**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In fact, **MOSEK** solves the same problem, thanks to the automatic dualizer that decides whether it is more convenient to solve the **primal** or the **dual** of a given problem.  

    The primal problem log output reports the solved problem as *dual*:  
    `Optimizer  - solved problem         : the dual`

    While the dual formulation reports solving the *primal*:  
    `Optimizer  - solved problem         : the primal`

    This proves the claim. Therefore, in both cases, we actually solve the **very same formulation**.
    """
    )
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
