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
    # *Fusion:* Object-Oriented API for Conic Optimization

    *Fusion* is an Object-Oriented API specifically designed for Conic Optimization with **MOSEK**. In version 9 of **MOSEK** *Fusion* is available for Python, C#, Java and C++.

    *Fusion* makes it easy to assemble optimization models from conic blocks without going through the nitty-gritty of converting the optimization problem into matrix form - *Fusion* takes care of that part. It makes it easy to add and remove constraints and experiment with the model, making prototyping of complex models very quick. It provides  linear expressions, linear algebra operations and cones.

    This is a quick demonstration of the main capabilities of *Fusion*. More details may be found in the documentation for the respective APIs. In particular section 6 of each Fusion API manual contains a lot more modeling techniques.
    """
    )
    return


@app.cell
def _():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var
    import mosek.fusion.pythonic      # Provides operators +, -, @, .T, slicing etc.
    import numpy as np
    import sys
    return Domain, Expr, Model, ObjectiveSense, np, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Problem formulation in *Fusion*""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fusion solves optimization problems of the form

    $$
    \begin{array}{rll}
    \text{minimize/maximize}    & c^T x       & \\
    \text{subject to}           & A^i x + b^i & \in & K^i, & \forall i, \\
    \end{array}
    $$

    where $K^i$ are convex cones. The possible cones $K^i$ are  

    * $\{0\}$ - the zero cone. This expresses simply a linear equality constraint $Ax+b=0$.
    * $\mathbb{R}_+$ - positive orthant. This expresses simply a linear inequality constraint $Ax+b\geq 0$.
    * $\mathcal{Q}$ - quadratic cone, $x_1\geq \sqrt{x_2^2+\cdots+x_n^2}$ where $n$ is the length of the cone.
    * $\mathcal{Q_r}$ - rotated quadratic cone, $2x_1x_2\geq x_3^2+\cdots+x_n^2$, $x_1,x_2\geq 0$.
    * $K_\mathrm{exp}$ - the exponential cone $x_1\geq x_2\exp(x_3/x_2)$, useful in particular in entropy ptoblems.
    * $\mathcal{P}_\alpha$ - the three-dimensional power cone $x_1^\alpha x_2^{1-\alpha}\geq |x_3|$, where $0<\alpha<1$.
    * $\mathbb{S}_+$ - the cone of positive semidefinite matrices.

    That allows for expressing linear, conic quadratic (SOCP), semidefinite, relative entropy, $p$-norm and many other types of problems.

    # Linear expressions

    Linear expressions are represented by the class ``Expression``, of which ``Variable`` (that is an optimization variable in the model) is a special case. Linear expressions are constructed in an intuitive way. For example if $A,b$ are constant data matrices then we can form $Ax+b$ as follows:
    """
    )
    return


@app.cell
def _(Model, np):
    _m, n = (10, 6)
    _A = np.random.uniform(-1.0, 1.0, [_m, n])
    b = np.random.uniform(-1.0, 1.0, [_m])
    M = Model('example model')
    x = M.variable(n)
    e = _A @ x + b
    print(e.getShape())
    return M, e, n, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Conic constraints

    We can now solve the unconstrained linear regression problem, that is minimize $\|Ax+b\|_2$. This will require a new variable $t$ such that the compound vector $(t,Ax+b)$ belongs to the quadratic cone (i.e. $t\geq \|Ax+b\|_2$). The compound vector is created as a stack from existing expressions of compatible shapes.
    """
    )
    return


@app.cell
def _(Domain, Expr, M, ObjectiveSense, e, x):
    # Add scalar variable and conic quadratic constraint t >= \|Ax+b\|_2
    t = M.variable()
    M.constraint(Expr.vstack(t, e), Domain.inQCone())

    # Let t be the objective we minimize
    M.objective(ObjectiveSense.Minimize, t)

    # Solve and print solution
    M.solve()
    print(x.level())
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Suppose we want to further restrict $x$, say $f^Tx\geq 1$ where $f$ is a given vector.""")
    return


@app.cell
def _(Expr, M, n, np, x):
    f = np.random.uniform(0.0, 1.0, [n])

    # f^T dot x >= 1
    M.constraint(Expr.dot(f,x) >= 1)

    M.solve()
    print(x.level())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Matrix notation

    Now suppose we want change the objective to

    $$
    \text{minimize}\quad \|Ax + b\|_2 + \sum_{j=1}^n \lambda_i e^{x_i}
    $$

    where $\lambda$ are positive coefficients. This can be rewritten as

    $$
    \begin{array}
    {lll}
    \text{minimize}    & t + \lambda^T w   & \\
    \text{subject to}  & (t, Ax+b) \in \mathcal{Q}, & \\
                       & (w_i, 1, x_i) \in K_\mathrm{exp}.
    \end{array}
    $$

    (Indeed, the last set of constraints is just $w_i\geq e^{x_i}$). We only need to define the last set of constraints and this can be achieved in one go by stacking them in a matrix as follows:
    """
    )
    return


@app.cell
def _(Domain, Expr, M, n, x):
    w = M.variable(n)

    M.constraint(Expr.hstack(w, Expr.constTerm(n, 1.0), x), Domain.inPExpCone());
    return (w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The horizontal stack above creates an expression of shape $n \times 3$ which looks as follows

    $$
    \left[
    \begin{array}{cc}
    w_1 & 1 & x_1 \\
    w_2 & 1 & x_2 \\
    . & . & .  \\
    w_n & 1 & x_n \\
    \end{array}
    \right]
    $$

    and the conic constraint is just a short way of writing that *every row of that matrix belongs to the exponential cone*, which is exactly what we need.

    We can now solve it, this time with log on screen.
    """
    )
    return


@app.cell
def _(M, ObjectiveSense, n, np, sys, t, w, x):
    lamb = np.random.uniform(0.0, 1.0, [n])

    M.setLogHandler(sys.stdout)

    # Objective = t + lambda dot w
    M.objective(ObjectiveSense.Minimize, t + w.T @ lamb)
    M.solve()
    print(x.level())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Semidefinite variables

    An $n\times n$ semidefinite variable can be defined with:
    """
    )
    return


@app.cell
def _(Domain, Model):
    n_1 = 5
    M_1 = Model('semidefinite model')
    X = M_1.variable(Domain.inPSDCone(n_1))
    print(X.getShape())
    return M_1, X, n_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""and it can be used like any variable of this shape. For example we can solve the very simple illustrative problem of maximizing the sum of elements of $X$ subject to a fixed diagonal.""")
    return


@app.cell
def _(Expr, M_1, ObjectiveSense, X, n_1, np):
    diag = np.random.uniform(1.0, 2.0, n_1)
    print(diag, '\n')
    M_1.constraint(X.diag() == diag)
    M_1.objective(ObjectiveSense.Maximize, Expr.sum(X))
    M_1.solve()
    print(np.reshape(X.level(), [5, 5]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# A cheatsheet to building expressions""")
    return


@app.cell
def _(Domain, Expr, Model, np):
    # -----------------------------
    # Problem dimensions and data
    # -----------------------------
    n_2 = 6
    _m = 10
    a = np.random.uniform(0.0, 1.0, n_2)         # Vector a
    _A = np.random.uniform(0.0, 1.0, (_m, n_2))  # Matrix A

    # -----------------------------
    # Define the model and variables
    # -----------------------------
    M_2 = Model('demo_model')

    x_1 = M_2.variable('x', n_2, Domain.unbounded())
    y   = M_2.variable('y', n_2, Domain.greaterThan(0.0))
    z   = M_2.variable('z', n_2, Domain.inRange(-1.0, 1.0))
    X_1 = M_2.variable('X', [n_2, n_2])

    # -----------------------------
    # Simple expressions
    # -----------------------------
    e0 = x_1 + 1.0
    e1 = x_1 + y
    e2 = a + y
    e3 = x_1 - y
    e4 = x_1 + y + z
    e5 = Expr.add([x_1, y, z])
    e6 = 7.0 * x_1
    e7 = Expr.mulElm(a, x_1)
    e8 = Expr.dot(a, x_1)           # scalar
    print("Shape of e8:", e8.getShape())
    e9 = Expr.outer(a, x_1)         # matrix
    print("Shape of e9:", e9.getShape())
    e10 = Expr.sum(x_1)
    e11 = _A @ X_1                   # matrix multiplication
    print("Shape of e11:", e11.getShape())

    # -----------------------------
    # Explicit expressions using Expr methods
    # -----------------------------
    e0_expl  = Expr.add(x_1, 1.0)
    e1_expl  = Expr.add(x_1, y)
    e2_expl  = Expr.add(a, y)
    e3_expl  = Expr.sub(x_1, y)
    e4_expl  = Expr.add(Expr.add(x_1, y), z)
    e6_expl  = Expr.mul(7.0, x_1)
    e11_expl = Expr.mul(_A, X_1)
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
