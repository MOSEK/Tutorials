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
    # Optimization problem over nonnegative trigonometric polynomials. #

    We consider the trigonometric polynomial

    $$H(\omega) = x_0 + 2 \sum_{k=1}^n [ \Re(x_k) \cos(\omega k) + \Im(x_k) \sin(\omega k) ],$$

    where $H(\omega)$ is a real valued function paramtrized by the complex vector $x\in {\bf C}^{n+1}$, and where $\Im(x_0)=0$.

    The example shows how to construct a *non-negative* polynomial $H(\omega)\geq 0, \: \forall \omega$ that satisfies,

    $$1 - \delta \leq  H(\omega) \leq 1 + \delta, \quad  \forall \omega \in [0, \omega_p]$$

    while minimizing $\sup_{\omega\in [\omega_s,\pi]} H(\omega)$ over the variables $x$.

    In the signal processing literature such a trigonometric polynomial is known as (the squared amplitude response of) a Chebyshev lowpass filter. 

    A squared amplitude response $H(\omega)$ is always symmetric around $0$, so $\Im(x_k)=0$, and we consider only

    $$H(\omega) = x_0 + 2 \sum_{k=1}^n x_k \cos(\omega k) $$

    over the real vector $x\in {\bf R}^{n+1}$. However, the concepts in the example are readily applied to the case with $x\in {\bf C}^{n+1}$.

    **References:**

      1. "Squared Functional Systems and Optimization Problems",  Y. Nesterov, 2004.

      2. "Convex Optimization of Non-negative Polynomials: Structured Algorithms and Applications", Ph.D thesis, Y. Hachez, 2003.
    """
    )
    return


@app.cell
def _():
    import mosek
    from   mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var
    import mosek.fusion.pythonic
    from   math import cos, pi
    import numpy as np
    import sys
    return Domain, Expr, Matrix, Model, ObjectiveSense, Var, cos, np, pi, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Nonnegativity everywhere ###

    Nesterov proved in [1] that $H(\omega) \geq 0, \: \forall \omega$ if and only if 
    $$x_i = \langle T_i^{n+1}, X \rangle, \quad X \in {\mathcal H}^{n+1}_+,$$
    where ${\mathcal H}_+$ is the cone of Hermitian semidefinite matrices and $T_i$ is a Toeplitz matrix

    $$[T_i]_{kl} = \left \{ \begin{array}{ll}1, & k-l=i\\0 & \text{otherwise}.\end{array} \right .$$

    For example, for $n=2$ we have

    $$
       T_0 = \left[\begin{array}{ccc}
       1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1
       \end{array}
       \right], \quad
       T_1 = \left[\begin{array}{ccc}
       0 & 0 & 0\\1 & 0 & 0\\0 & 1 & 0
       \end{array}
       \right],
       \quad
       T_2 = \left[\begin{array}{ccc}
       0 & 0 & 0\\0 & 0 & 0\\1 & 0 & 0
       \end{array}
       \right].
    $$

    In our case we have $\Im(x_i)=0$, i.e., $X\in {\mathcal S}^{n+1}_+$ is a real symmetric semidefinite matrix.

    We define the *cone on nonnegative trigonometric polynomials* as

    $$
       K^n_{0,\pi} = \{ x\in \mathbf{R} \times \mathbf{C}^n \mid x_i = \langle X, T_i\rangle, \: i=0,\dots,n, \: X\in\mathcal{H}_+^{n+1} \}.
    $$
    """
    )
    return


@app.cell
def _(Expr, Matrix):
    def T_dot_X(n, i, X, a=1.0):
        if i>=n or i<=-n: return Expr.constTerm(0.0)
        else: return Expr.dot(Matrix.diag(n, a, -i), X)
    return (T_dot_X,)


@app.cell
def _(Domain, T_dot_X):
    def trigpoly_0_pi(M, x):
        '''x[i] == <T(n+1,i),X>'''
        n = int(x.getSize()-1)
        X = M.variable("X", Domain.inPSDCone(n+1))

        for i in range(n+1):
            M.constraint(T_dot_X(n+1,i,X) == x[i])
    return (trigpoly_0_pi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that we have dropped the imaginary part of $X$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Nonnegativity on $[0,a]$ ###

    Similarly, $H(\omega)$ is nonnegative on $[0,a]$ if and only if

    $$x_i =
    \langle X_1, T_i^{n+1} \rangle + 
    \langle X_2, T_{i+1}^n \rangle +
    \langle X_2, T_{i-1}^n \rangle -
    2 \cos(a)\langle X_2, T_{i}^n \rangle, \quad 
    X_1 \in {\mathcal H}^{n+1}_+, \:
    X_2 \in {\mathcal H}^n_+,
    $$

    or equivalently

    $$
      K^n_{0,a} = \{ x\in \mathbf{R}\times \mathbf{C}^n \mid
       x_i = \langle X_1, T_i^{n+1} \rangle +
       \langle X_2 , T_{i+1}^n \rangle +
       \langle X_2 , T_{i-1}^n \rangle -
       2\cos(a)\langle X_2 , T_i^n\rangle, \: X_1\in \mathcal{H}_+^{n+1}, X_2\in\mathcal{H}_+^n \}.
    $$
    """
    )
    return


@app.cell
def _(Domain, T_dot_X, cos):
    def trigpoly_0_a(M, x, a):
        '''x[i] == <T(n+1,i),X1> + <T(n,i+1),X2> + <T(n,i-1),X2> - 2*cos(a)*<T(n,i),X2>'''
        n = int(x.getSize()-1)
        X1 = M.variable(Domain.inPSDCone(n+1))
        X2 = M.variable(Domain.inPSDCone(n))

        for i in range(n+1):
            M.constraint(T_dot_X(n+1,i,X1) + T_dot_X(n,i+1,X2) + \
                         T_dot_X(n,i-1,X2) + T_dot_X(n,i,X2,-2*cos(a)) == x[i])
    return (trigpoly_0_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that we have dropped the imaginary part of $X_1$ and $X_2$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Nonnegativity on $[a,\pi]$ ###

    Finally, $H(\omega)$ is nonnegative on $[a,\pi]$ if and only if

    $$x_i = 
    \langle X_1, T_i^{n+1} \rangle -
    \langle X_2, T_{i+1}^n \rangle -
    \langle X_2, T_{i-1}^n \rangle +
    2 \cos(a)\langle X_2, T_{i}^n \rangle, \quad 
    X_1 \in {\mathcal S}^{n+1}_+, \:
    X_2 \in {\mathcal S}^n_+,
    $$

    or equivalently

    $$
      K^n_{a,\pi} = \{ x\in \mathbf{R}\times \mathbf{C}^n \mid
       x_i = \langle X_1, T_i^{n+1} \rangle -
       \langle X_2 , T_{i+1}^n \rangle -
       \langle X_2 , T_{i-1}^n \rangle +
       2\cos(a)\langle X_2 , T_i^n\rangle, \: X_1\in \mathcal{H}_+^{n+1}, X_2\in\mathcal{H}_+^n \}.
    $$
    """
    )
    return


@app.cell
def _(Domain, T_dot_X, cos):
    def trigpoly_a_pi(M, x, a):
        '''x[i] == <T(n+1,i),X1> - <T(n,i+1),X2> - <T(n,i-1),X2> + 2*cos(a)*<T(n,i),X2>'''
        n = int(x.getSize()-1)
        X1 = M.variable(Domain.inPSDCone(n+1))
        X2 = M.variable(Domain.inPSDCone(n))

        for i in range(n+1):
            M.constraint(T_dot_X(n+1,i,X1) + T_dot_X(n,i+1,X2,-1.0) + \
                         T_dot_X(n,i-1,X2,-1.0) + T_dot_X(n,i,X2,2*cos(a)) == x[i])
    return (trigpoly_a_pi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that we have dropped the imaginary part of $X_1$ and $X_2$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## An epigraph formulation ##
    The *epigraph* $H(\omega) \leq t$ can now be characterized simply as 
    $-u \in K^n_{[a,b]}, \: u=(x_0-t, \, x_{1:n}).$
    """
    )
    return


@app.cell
def _(Domain, pi, trigpoly_0_a, trigpoly_0_pi, trigpoly_a_pi):
    def epigraph(M, x, t, a, b):
        '''Models 0 <= H(w) <= t, for all w in [a, b], where
             H(w) = x0 + 2*x1*cos(w) + 2*x2*cos(2*w) + ... + 2*xn*cos(n*w)'''
        n  = int(x.getSize()-1)    
        u = M.variable(n+1, Domain.unbounded())
        M.constraint(t == x[0] + u[0]) 
        M.constraint(x[1:] + u[1:] == 0)

        if a==0.0 and b==pi:
            trigpoly_0_pi(M, u)
        elif a==0.0 and b<pi:
            trigpoly_0_a(M, u, b)
        elif a<pi and b==pi:
            trigpoly_a_pi(M, u, a)
        else:
            raise ValueError("invalid interval.")
    return (epigraph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A hypograph formulation ##
    Similarly, the *hypograph* $H(\omega) \geq t$ can be characterized as 
    $u \in K^n_{[a,b]}, \: u=(x_0-t, \, x_{1:n}).$
    """
    )
    return


@app.cell
def _(Domain, Var, pi, trigpoly_0_a, trigpoly_0_pi, trigpoly_a_pi):
    def hypograph(M, x, t, a, b):
        '''Models 0 <= t <= H(w), for all w in [a, b], where
             H(w) = x0 + 2*x1*cos(w) + 2*x2*cos(2*w) + ... + 2*xn*cos(n*w)'''

        n  = int(x.getSize()-1)    
        u0 = M.variable(1, Domain.unbounded())    
        M.constraint(t == x[0] - u0)
        u = Var.vstack( u0, x[1:] )

        if a==0.0 and b==pi:
            trigpoly_0_pi(M, u)
        elif a==0.0 and b<pi:
            trigpoly_0_a(M, u,  b)
        elif a<pi and b==pi:
            trigpoly_a_pi(M, u, a)
        else:
            raise ValueError("invalid interval.")
    return (hypograph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Putting it together ##""")
    return


@app.cell
def _(Domain, Model):
    n = 10
    M = Model("trigpoly")
    x = M.variable("x", n+1, Domain.unbounded())
    return M, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Global nonnegativity ###""")
    return


@app.cell
def _(M, trigpoly_0_pi, x):
    # H(w) >= 0
    trigpoly_0_pi(M, x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Passband specifications ###""")
    return


@app.cell
def _(M, epigraph, hypograph, pi, x):
    wp = pi/4
    delta = 0.05

    # H(w) <= (1+delta), w in [0, wp]    
    epigraph(M, x, 1.0+delta, 0.0, wp)

    # (1-delta) <= H(w), w in [0, wp]
    hypograph(M, x, 1.0-delta, 0.0, wp)
    return (wp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Stopband specifications ###""")
    return


@app.cell
def _(Domain, M, epigraph, pi, wp, x):
    ws = wp + pi/8

    # H(w) < t, w in [ws, pi]
    t = M.variable("t", 1, Domain.greaterThan(0.0))
    epigraph(M, x, t, ws, pi)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting the objective ###""")
    return


@app.cell
def _(M, ObjectiveSense, t):
    M.objective(ObjectiveSense.Minimize, t)
    return


@app.cell
def _(M, sys):
    M.setLogHandler(sys.stdout)
    return


@app.cell
def _(M):
    M.solve()
    return


@app.cell
def _(x):
    x.level()
    return


@app.cell
def _(t):
    t.level()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plotting the amplitude response ###""")
    return


@app.cell
def _(cos, np, pi, x):
    xx = x.level()
    def H(w): return xx[0] + 2*sum([xx[i]*cos(i*w) for i in range(1,len(xx))])
    w  = np.linspace(0, pi, 100)
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    plt.plot(w, [H(wi) for wi in w], 'k')
    plt.show()
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
