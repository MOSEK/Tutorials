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
    # Binary quadratic problems

    We discuss a simple solver for unconstrained binary quadratic problems, that is optimization problems of the form

    \begin{equation}
    \begin{array}{ll}
    \textrm{minimize}  &   x^TQx+P^Tx+R \\
    \textrm{subject to}&   x\in\{0,1\}^n,
    \end{array}
    \end{equation}

    where $Q$ is symmetric. This framework can encode many hard problems, such as max-cut or maximum clique. Note that the continuous version with $x\in[0,1]^n$ need not be convex.

    We present a simple SDP-based branch-and-bound solver. This implementation is not intended to compete with specialized tools such as [BiqCrunch](http://lipn.univ-paris13.fr/BiqCrunch/) or other sophisticated algorithms for this problem. The aim is rather to show that one can achieve pretty decent results with under 100 lines of quickly prototyped Python code using the MOSEK Fusion API.

    For complete source code and examples see the folder at [GitHub](https://github.com/MOSEK/Tutorials/tree/master/Fusion/BinaryQuadratic-SDP).

    # Algorithm

    We use the standard semidefinite relaxation of the problem, namely:

    \begin{equation}
    \begin{array}{ll}
    \textrm{minimize}  &   \sum_{ij}Q_{ij}X_{ij} + \sum_i P_ix_i + R \\
    \textrm{subject to}&   \left[\begin{array}{cc}X & x\\ x^T & 1\end{array}\right]\succeq 0,\\
                       &   \mathrm{diag}(X) = x.
    \end{array}
    \end{equation}

    The value of the relaxation is a lower bound for the original problem as we see by setting $X=xx^T$ for $x\in\{0,1\}^n$.

    The algorithm maintains a lower bound ``lb``, upper bound ``ub`` (objective value of the best integer solution found) and a queue of active nodes. We follow these rules:

    * In each iteration we pick the node with smallest objective value of the relaxation.
    * Each time a relaxation is solved we round the solution to nearest integers and try to use the result to improve the upper bound.
    * We branch on the variable whose value in the relaxation is closest to $\frac12$. Both child nodes, obtained by fixing that variable to $0$ or $1$, are smaller instances of the same problem with easy to compute $Q',P',R'$.

    The main node-processing loop follows below.
    """
    )
    return


@app.cell
def _(heapq, np):
    # The node-processing loop
    def solveBB(self):
        while self.active:
            # Pop the node with minimum objective from the queue
            obj, _, Q, P, R, curr, idxmap, pivot = heapq.heappop(self.active)

            self.lb = obj

            # Optimality is proved
            if self.lb >= self.ub - self.gaptol:
                self.active = []
                return

            self.stats()

            if len(P) <= 1: continue

            pidx = idxmap[pivot]

            # Construct and solve relaxations of two child nodes
            # where the pivot variable is assigned 0 or 1
            for val in [0, 1]:
                idxmap0     = np.delete(idxmap, pivot)
                curr0       = np.copy(curr)
                curr0[pidx] = val

                QT = np.delete(Q, pivot, axis=0)
                Q0 = np.delete(QT, pivot, axis=1)
                QZ = QT[:,pivot]
                PT = np.delete(P, pivot)
                P0 = PT + 2*val*QZ
                R0 = val*Q[pivot,pivot] + val*P[pivot] + R

                obj0, pivot0 = self.solveRelaxation(Q0, P0, R0, curr0, idxmap0)

                heapq.heappush(self.active, (obj0, self.relSolved, Q0, P0, R0, curr0, idxmap0, pivot0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The methods which solve the relaxation, find the pivot (branching) index and update the best solution are straightforward. We demonstrate only the fragment which defines the semidefinite relaxation in Fusion API:""")
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense):
    # SDP relaxation 
    # min QX + Px + R
    # st  Z = [X, x; x^T, 1] >> 0
    def fusionSDP(Q, P, R):
        n = len(P)
        M = Model("fusionSDP")
        Z = M.variable("Z", Domain.inPSDCone(n+1))
        X = Z[0:n,0:n]
        x = Z[0:n,n]
        M.constraint(X.diag() == x)
        M.constraint(Z[n,n] == 1)

        M.objective(ObjectiveSense.Minimize, Expr.constTerm(R) + Expr.dot(P,x) + Expr.dot(Q,X))

        return M, x
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The solver is straightforward to use. It suffices to write:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```
    solver = BB(Q, P, R)
    solver.solve()
    solver.summary()
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```
    REL_SOLVED ACTIVE_NDS       OBJ_BOUND        BEST_OBJ
             1          0  -10000000000.0  -34.1829220672
             1          0  -40.9578610377  -34.1829220672
             2          0  -40.9578610377  -34.5354934947
             3          1  -40.0967961795  -34.5354934947
             5          2  -40.0967961795  -37.6612921792
             5          2  -39.9359433001  -37.6612921792
             7          3  -38.9245605048  -37.6612921792
             9          4  -38.4153688863  -37.6612921792
            11          5  -38.1908232037  -37.6612921792
            13          6  -38.1309765476  -37.6612921792
            15          7  -37.9917627647  -37.6612921792
            17          8  -37.8758057302  -37.6612921792
            19          9  -37.6906063328  -37.6612921792
            21         10  -37.6747901176  -37.6612921792
            23         11  -37.6613714044  -37.6612921792
            25          0  -37.6612921492  -37.6612921792
    val = -37.6612921792
    sol = [0 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0]
    relaxations   = 25
    intpntIter    = 243
    optimizerTime = 0.194850206375
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Example 1. Random data

    We consider a random problem

    \begin{equation}
    \mathrm{minimize}_{x\in\{0,1\}^n}\ x^TQx
    \end{equation}

    where $Q_{i,j}\sim \mathrm{Normal}(0,1)$ for $i\geq j$. We generate instances with $30\leq n\leq 100$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _src = "https://raw.githubusercontent.com/MOSEK/Tutorials/master/binary-quadratic/stats/random.png"

    mo.image(src=_src, rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Horizontal scale: $n$. Each dot corresponds to an instance. Left: number of relaxations solved. Right: total time spent in the MOSEK optimizer (sec.). Log scale.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Example 2: BiqMac

    We next consider problems of the same form from the library [BiqMac](http://biqmac.aau.at/biqmaclib.html) of binary quadratic problems. We take all instances with $n\leq 100$. Since all the $Q$ matrices in BiqMac are integral, we can use a stronger termination criterion:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```
    if np.ceil(self.lb) >= self.ub:
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _src = "https://raw.githubusercontent.com/MOSEK/Tutorials/master/binary-quadratic/stats/biq.png"

    mo.image(src=_src, rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Horizontal scale: $n$. Each dot corresponds to an instance. Left: number of relaxations solved. Right: total time spent in the MOSEK optimizer (sec.). Log scale.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Example 3: Binary least squares

    Binary least squares is the problem

    \begin{equation}
    \mathrm{minimize}_{x\in\{0,1\}^n}\ \|Ax-b\|_2^2 \quad (=x^T(A^TA)x-(2A^Tb)^Tx+b^Tb)
    \end{equation}

    where $A\in\mathbb{R}^{m\times n}, b\in \mathbb{R}^m$. This is a mixed-integer SOCP, so we can compare our solver with the solutions obtained directly with MOSEK. As suggested in [Park, Boyd 2017](https://web.stanford.edu/~boyd/papers/pdf/int_least_squares.pdf) we generate reasonably hard random data by taking

    \begin{equation}
    \begin{array}{l}
    A\in\mathbb{R}^{m\times n},\quad A_{i,j}\sim \mathrm{Normal}(0,1)\\
    b=Ac,\ c\in\mathbb{R}^n,\ c_i\sim\mathrm{Uniform}(0,1)
    \end{array}
    \end{equation}

    For this test we fix $n=50$ and vary $40\leq m\leq 150$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _src = "https://raw.githubusercontent.com/MOSEK/Tutorials/master/binary-quadratic/stats/bls.png"

    mo.image(src=_src, rounded=True)
    return


@app.cell
def _(mo):
    mo.md(r"""Horizontal scale: $m$. Each dot corresponds to an instance. Blue: this algorithm. Red: mixed-integer SOCP in MOSEK. Left: number of relaxations solved. Right: total time spent in the MOSEK optimizer (single-thread, sec.). Log scale.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Small executable example""")
    return


@app.cell
def _():
    import branchbound
    from branchbound import BB 
    import numpy as np

    n=25
    Q1 = np.random.normal(0.0, 1.0, (n,n))
    solver = BB((Q1+Q1.transpose())/2, np.random.uniform(-1.0, 3.0, n), 0.0)
    solver.solve()
    solver.summary()
    return (np,)


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
