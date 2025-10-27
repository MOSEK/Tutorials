import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Truss Topology Design with Multiple Load Scenarios

    This notebook demonstrates how to model and solve a **truss topology optimization problem** using the MOSEK Fusion API inspired by [Lectures on Modern Convex Optimization](https://www2.isye.gatech.edu/~nemirovs/LMCOBookSIAM.pdf) by Ben-Tal and Nemirovski. Truss topology design (TTD) is a classical structural optimization problem that arises in civil, mechanical, and aerospace engineering. The task is to determine the optimal configuration of a truss, a structure formed by straight bars connected at nodes, that is both lightweight and stiff under a given set of external loads. The problem is based on equilibrium constraints and second-order cone reformulations.



    ---


    ## Problem Definition


    We consider a truss consisting of:

    - A set of **bars** \(i \in I = \{1,\dots,n\}\)
    - A set of **nodes** \(j \in J = \{1,\dots,m\}\)
    - A set of **load scenarios** \(k \in K = \{1,\dots,K\}\)


    ### Parameters

    - \( f(j,k) \): External force applied at node \(j\) in scenario \(k\).
    - \( b(j,i) \): Stiffness contribution of bar \(i\) at node \(j\).
    - \( V_{\max} \): Maximum allowed total volume of the truss.


    ### Decision Variables

    - \( t(i) \geq 0 \): Volume of bar \(i\).
    - \( s(i,k) \): Stress/elongation of bar \(i\) under load scenario \(k\).
    - \( \sigma(i,k) \geq 0 \): Required cross-sectional area of bar \(i\) under load \(k\).
    - \( \tau \geq 0 \): Upper bound on compliance, which is minimized.


    ---


    ## Constraints


    The optimization model can be written as:


    1. **Rotated Cone Constraints (Stress-Volume Relation):**



    \[
    s(i,k)^2 \leq 2 \, t(i) \, \sigma(i,k), \quad \forall i \in I, \, k \in K
    \]

    This enforces the geometric relation between stress, bar volume, and cross-sectional area. It is expressed as a **rotated second-order cone**.


    2. **Resource Constraints (Scenario-wise Compliance):**



    \[
    \sum_{i \in I} \sigma(i,k) \leq \tau, \quad \forall k \in K
    \]

    Ensures that under each load scenario, the required cross-sections do not exceed \(\tau\).


    3. **Total Volume Constraint:**



    \[
    \sum_{i \in I} t(i) \leq V_{\max}
    \]

    Limits the total material used in the truss.

    4. **Stiffness Equilibrium Constraints:**

    \[
    \sum_{i \in I} b(j,i) \, s(i,k) = f(j,k), \quad \forall j \in J, \, k \in K
    \]

    Ensures that stresses balance the external nodal forces for each scenario.


    ---


    ## Objective


    We minimize the upper bound on compliance \(\tau\):

    \[
    \min \tau
    \]


    This ensures the stiffest possible truss design under all load scenarios.


    ---


    ## Cone Reformulation


    Constraint (1) is written in conic form using the **rotated quadratic cone**:


    \[
    (s(i,k), \, t(i), \, \sigma(i,k)) \in \mathcal{Q}^r = \{ (u,v,w) : 2uv \geq w^2, u,v \geq 0 \}
    \]


    This makes the problem a **Second Order Cone Program (SOCP)**, which can be solved efficiently with MOSEK.


    ---

    The data is taken from the example provided in [this implementation by GAMS](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_trussm.html) and is provided below
    .
    """
    )
    return


@app.cell
def _():
    import numpy as np

    # Index sets
    bars = ["i1","i2","i3","i4","i5"]
    nodes = ["j1","j2","j3","j4"]
    scen = ["k1","k2","k3"]

    n = 5
    m = 4
    K = 3

    # External Force Matrix
    f = np.array([
        [ 0.0008,  1.0668,  0.2944],   # j1
        [ 0.0003,  0.0593, -1.3362],   # j2
        [-0.0006, -0.0956,  0.7143],   # j3
        [-1.0003, -0.8323,  1.6236],   # j4
    ])  # shape (m, K)

    # b(j,i) stiffness matrix (rows = nodes, cols = bars)
    b = np.array([
        [ 1.0,  0.0,  0.5,  0.0,  0.0],   # j1
        [ 0.0,  0.0, -0.5, -1.0,  0.0],   # j2
        [ 0.0,  0.5,  0.0,  0.0,  1.0],   # j3
        [ 0.0,  0.5,  0.0,  1.0,  0.0],   # j4
    ])  # shape (m, n)

    maxvolume = 10.0
    return K, b, f, m, maxvolume, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we can implement this model using MOSEK Fusion in Python.""")
    return


@app.cell
def _(K, b, f, m, maxvolume, n):
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var
    import mosek.fusion.pythonic

    # Build Fusion model
    M = Model("truss")

    # Variables
    tau   = M.variable("tau", 1, Domain.greaterThan(0.0))
    t     = M.variable("t", n, Domain.greaterThan(0.0))       # bar volumes
    sigma = M.variable("sigma", [n, K], Domain.greaterThan(0.0))
    s     = M.variable("s", [n, K], Domain.unbounded())       # stresses

    # Rotated cone constraints: (t[i], sigma[i,k], s[i,k]) in RQCone
    for k in range(K):
        M.constraint(Expr.hstack(t,sigma[:,k],s[:,k]), Domain.inRotatedQCone())

    # Resource constraints: sum_i sigma[i,k] <= tau
    for k in range(K):
        M.constraint(
            Expr.sum(sigma[:,k]) <= (tau)
        )

    # Volume constraint: sum_i t[i] <= maxvolume
    M.constraint(Expr.sum(t), Domain.lessThan(maxvolume))

    # Stiffness equations: sum_i s[i,k] * b[j,i] = f[j,k]
    for j in range(m):
        for k in range(K):
            lhs = Expr.dot(b[j, :], s.slice([0, k], [n, k+1]))
            M.constraint(lhs, Domain.equalsTo(f[j, k]))

    # Objective: minimize tau
    M.objective(ObjectiveSense.Minimize, tau)

    # Solve
    M.solve()

    # Print results
    print("Problem Status:", M.getProblemStatus())
    print("tau (objective):", f"{tau.level()[0]:.4f}")
    return


if __name__ == "__main__":
    app.run()