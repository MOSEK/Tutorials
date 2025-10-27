import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var
    import mosek.fusion.pythonic
    import sys
    import requests, zipfile, io, os, glob
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import time
    return (
        Domain,
        Expr,
        Model,
        ObjectiveSense,
        StandardScaler,
        glob,
        io,
        mo,
        np,
        os,
        pd,
        requests,
        time,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Rank-One Sparse Regression Model

    This notebook is based on [Rank-one Convexification for Sparse Regression](https://arxiv.org/abs/1901.10334) by Atamtürk and Gómez. 

    In the paper a model is designed to solve regression problems when the number of predictors is large and we expect only a small subset of them to be truly relevant. This model combines **quadratic loss minimization** with a **sparsity-inducing penalty** and a **conic reformulation** that makes the optimization problem tractable. Some properties this model offers to users:

    - **Feature selection**: It automatically selects only the most relevant predictors.  
    - **Interpretability**: Produces simpler, more interpretable models.  
    - **Regularization**: Prevents overfitting by discouraging unnecessarily large coefficients.  
    - **High-dimensional settings**: Particularly useful when the number of predictors \(p\) is large compared to the number of samples \(n\).  

    In short, this model balances **accuracy** and **sparsity**, giving us predictive models that are both powerful and easy to interpret.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We consider the following optimization problem:

    \[
    \min_{\beta \in \mathbb{R}^p} \;\; 
    \|y - X\beta\|_2^2 
    \;+\; \lambda \|\beta\|_2^2 
    \;+\; \mu \|\beta\|_1 
    \]

    \[
    \quad \text{s.t.} \quad \|\beta\|_0 \leq k
    \]

    where: 

    - \(y \in \mathbb{R}^n\) : response vector (observed outcomes).  
    - \(X \in \mathbb{R}^{n \times p}\) : design (feature) matrix with \(n\) samples and \(p\) predictors.   
    - \(\lambda \geq 0\) : regularization parameter controlling the ridge penalty.  
    - \(\mu \geq 0\) : regularization parameter controlling the lasso penalty.  
    - \(k\) : sparsity level, i.e., maximum number of predictors allowed to be nonzero.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This optimization problem extends least squares regression by adding two regularization terms and a sparsity constraint:

    - Squared error term $\|y - X\beta\|_2^2$: Penalizes discrepancies between predictions $X\beta$ and the observed responses $y$.

    - Ridge penalty $\lambda \|\beta\|_2^2$: Shrinks coefficient values to improve numerical stability and reduce overfitting (especially when predictors are correlated or $p$ is large).

    - Lasso penalty $\mu \|\beta\|_1$: Encourages sparsity by shrinking some coefficients to zero.

    - Cardinality constraint $\|\beta\|_0 \leq k$: Limits the number of nonzero entries in $\beta$ to at most $k$, directly enforcing sparsity and interpretability.


    In the model the cardinality constraint is usually problematic in application. This is a hard sparsity constraint and problems involving this count constraint are usually *NP-Hard*. Therefore instead of solving this model, the following transformation is applied. 

    In the model the cardinality constraint is usually problematic in practice. This is a hard sparsity constraint, which forces the solution to include at most 𝑘 nonzero coefficients. However, optimization problems with such a count constraint are generally NP-hard, since solving them requires searching through all possible subsets of predictors. Therefore, instead of solving this formulation directly, the model is often transformed or relaxed into an alternative form that can be solved efficiently.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Alternative Reformulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \min_{\beta \in \mathbb{R}^p}\;
    \|y - X\beta\|_2^2 \;+\; \lambda \|\beta\|_2^2 \;+\; \mu \|\beta\|_1
    \quad \text{s.t.}\quad \|\beta\|_0 \le k .
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As the first step, expand the first two terms in the objective function. Through expansion one can see that we easily reach the general quadratic form. 

    $$
    \|y - X\beta\|_2^2
    = y^\top y - 2\,y^\top X\beta + \beta^\top X^\top X \beta .
    $$

    Then the partial objective becomes: 

    $$
    \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2
    = y^\top y - 2\,y^\top X\beta + \beta^\top (X^\top X + \lambda I)\beta .
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The $\ell_1$-norm $\|\beta\|_1 = \sum_{i=1}^p |\beta_i|$ is replaced by auxiliary variables $u_i \ge 0$ with constraints: 

    \[
    -u_i \le \beta_i \le u_i .
    \]

    These linear inequalities ensure $u_i \ge |\beta_i|$. Since the objective 
    minimizes $\sum_{i=1}^p u_i$, at optimality we have $u_i = |\beta_i|$. 
    Thus, 

    \[
    \|\beta\|_1 = \sum_{i=1}^p u_i
    \]

    is exactly represented in linear form.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To model the cardinality constraint $\|\beta\|_0 \le k$, we introduce binary 
    indicator variables $z_i \in \{0,1\}$ for each coefficient. Formally, 
    $z_i = 1$ if $\beta_i$ is allowed to be nonzero, and $z_i = 0$ if 
    $\beta_i$ must be zero. Thus, every nonzero $\beta_i$ is *counted* 
    through its corresponding indicator $z_i$.

    The constraint below ensures that at most $k$ coefficients can be nonzero. 

    \[
    \sum_{i=1}^p z_i \le k
    \]


    To enforce the connection between $\beta_i$ and its indicator $z_i$, one can impose: 

    \[\beta_i (1 - z_i) = 0 \quad \text{for } i=1,\dots,p\]

    This guarantees the *either–or* relationship: if $z_i=0$, then $\beta_i = 0$, while if $z_i=1$, then $\beta_i$ is free to take a 
    nonzero value. In this way, the challenge of the count operation can be worked around.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At this point, to model the constraint  

    $$
    \beta_i (1 - z_i) = 0, \quad i=1,\dots,p,
    $$  

    the paper suggests introducing a *big-$M$* formulation. In that approach, large constants are used to relax the constraint and enforce the link between $\beta_i$ and $z_i$.  

    After the big-$M$ value is fixed, the calculation becomes relatively easy to implement. However, **determining a suitable big-$M$ value is itself a challenge**. If $M$ is chosen too small, the model may cut off feasible solutions. If $M$ is chosen too large, the relaxation becomes weak and the solver may perform poorly. Therefore, based on the specific problem and the data at hand, an **appropriate big-$M$ value must be determined each time**.  

    There is no universal choice:  
    - The bound should be large enough so that all feasible $\beta_i$ can be represented.  
    - But it should not be excessively large, since that would weaken the relaxation and slow down the solver.  


    By contrast, **MOSEK** has native support for *disjunctive constraints*. This means we do not need to approximate with big-$M$. Instead, we can directly model the logical condition:  

    - Either $z_i = 1$ (the variable $\beta_i$ is allowed to be nonzero),  
    - Or $\beta_i = 0$ (the variable is forced to zero when $z_i = 0$).  

    This way, the constraint is expressed exactly as an **either–or** condition, and MOSEK will internally handle the disjunction:

    \[
    \beta_i = 0 \;\;\;\; \lor \;\;\;\; z_i = 1, 
    \quad i = 1,\dots,p .
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then the final model becomes:

    $$
    \begin{aligned}
    \min_{\beta,u,z}\quad
    & -\,2\,y^\top X\beta \;+\; \beta^\top (X^\top X + \lambda I)\beta \;+\; \mu \sum_{i=1}^p u_i \\
    \text{s.t.}\quad
    & -u_i \le \beta_i \le u_i,\;\; u_i \ge 0, \quad i=1,\dots,p, \\
    & \beta_i = 0 \;\;\;\; \lor \;\;\;\; z_i = 1, 
    \quad i = 1,\dots,p \\
    & \sum_{i=1}^p z_i \le k, \quad z_i \in \{0,1\}, \;\; i=1,\dots,p, \\
    & \beta \in \mathbb{R}^p,\;\; z \in \{0,1\}^p,\;\; u \in \mathbb{R}_+^p.
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Implementation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The objective we are minimizing has three types of terms:

    \[
    -2y^\top X \beta \;+\; \beta^\top (X^\top X + \lambda I)\beta \;+\; \mu \sum_i u_i \;+\; \text{const.}
    \]

    ---

    ## 1. Linear terms:

    - The scalar term \(-2y^\top X\beta\) is a simple inner product, i.e. linear in the decision variables \(\beta\).
    - The penalty \(\mu \sum_i u_i\) is also linear in the auxiliary variables \(u_i\).  
    These can be added directly to the optimization model without special reformulation.

    ---

    ## 2. Quadratic Term

    The expression

    \[
    \beta^\top (X^\top X + \lambda I)\beta
    \]

    is **quadratic**. 
    There are **two approaches** we can apply to handle it as a rotated second-order cone (RSOC) constraint.


    ---

    ### Option 1: Cholesky-based approach

    To encode it in a conic optimization model, we introduce an auxiliary variable \(t\) representing the quadratic form:

    \[
    t \;\ge\; \beta^\top Q \beta, \quad Q = X^\top X + \lambda I
    \]



    1. Since \(Q\) is positive semidefinite we can perform a **Cholesky decomposition**:

    \[
    Q = L L^\top
    \]

    2. Express the quadratic form as a norm:  

    \[
    \beta^\top Q \beta = \| L^\top \beta \|_2^2
    \]

    3. Encode it as a **rotated SOC constraint**:

    \[
    (1,\; 0.5\,t,\; L^\top \beta) \;\in\; \mathcal{Q}_r
    \]

    This enforces \(t \ge \| L^\top \beta \|_2^2\).

    ---

    ### Option 2: Stacked-matrix approach (no Cholesky)

    Firstly, let's introduce an auxiliary variable \(t\) representing the quadratic form again:

    \[
    t \;\ge\; \beta^\top (X^\top X + \lambda I)\beta
    \]

    1. Stack \(X\) and \(\sqrt{\lambda} I\) vertically:

    \[
    \begin{bmatrix} X \\ \sqrt{\lambda} I \end{bmatrix}
    \]

    2. Express the quadratic form as a norm:

    \[
    \beta^\top (X^\top X + \lambda I) \beta = \left\| \begin{bmatrix} X \\ \sqrt{\lambda} I \end{bmatrix} \beta \right\|_2^2
    \]

    3. Introduce a scalar variable \(t \ge \| [X; \sqrt{\lambda} I] \beta \|_2\) and write as a **rotated SOC constraint**:

    \[
    \left(1, \frac{t}{2}, \begin{bmatrix} X \\ \sqrt{\lambda} I \end{bmatrix} \beta \right) \;\in\; \mathcal{Q}_r
    \]

    This approach avoids Cholesky factorization and works well when \(X\) is sparse.

    ---

    **Note on selection:**  

    - If \(X\) is **dense**, the Cholesky-based approach is usually faster and more efficient.  
    - If \(X\) is **sparse**, the stacked-matrix approach is preferred.  

    The choice depends on the **structure of \(X\)** and the **solver/data requirements**.


    ## 3. Putting it together:

    - The final objective is linear in \(\beta, u, t\).
    - The only nonlinear part \(\beta^\top Q \beta\) has been reformulated into a **cone constraint**, so the solver can handle it efficiently.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Implementing the model in Python""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following function implements the model described earlier.  

    The quadratic term can be handled using **either of the two approaches** depending on the user’s preference and the input data:

    - If the user sets `cholesky = True` (the **default**), the function uses the **Cholesky-based transformation** (Option 1).  
    - If the user sets `cholesky = False`, the function uses the **stacked-matrix transformation** (Option 2).  

    This design allows the model to adapt to the structure of the input data and the user’s requirements while keeping the implementation flexible.

    """
    )
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, np, time):
    def RankOne(n, p, lmb, mu, y , X, k,cholesky = True):
        M = Model("RankOne")

        # Decision variables
        B = M.variable("B", p, Domain.unbounded())        # regression coefficients β
        u = M.variable("u", p, Domain.greaterThan(0.0))   # auxiliary vars for |β|
        t = M.variable("t", 1)                            # scalar var for quadratic form
        z = M.variable("z", p, Domain.binary())           # binary indicators for support

        # Objective part: yᵀy (constant, included for completeness)
        term1 = y.T @ y

        # Objective part: -2 yᵀXβ
        term2 = -2 * Expr.mul(y.reshape(1, -1), Expr.mul(X, B))

        if cholesky:
            # Quadratic term βᵀ(XᵀX + λI)β handled with rotated quadratic cone
            Q = (X.T @ X + lmb * np.identity(p))
            w, V = np.linalg.eigh(Q)        # eigen-decomposition
            L = V @ np.diag(np.sqrt(w))     # factor such that Q = L Lᵀ

            # Cone constraint: (1, 0.5t, LᵀB) ∈ Rotated Quadratic Cone
            M.constraint(
                Expr.vstack(1.0, 0.5 * t, Expr.mul(L.T, B)),
                Domain.inRotatedQCone()
            )
        else:
            # This corresponds to the stacked matrix [X; sqrt(lambda) * I_p] used for the quadratic form
            Xp = np.vstack([X, np.sqrt(lmb) * np.identity(p)])

            # Rotated Quadratic Cone constraint:
            # (1, 0.5*t, [X; sqrt(lambda) I_p] * β) ∈ Rotated Quadratic Cone
            # This encodes the quadratic term βᵀ(XᵀX + λI)β as a rotated cone constraint
            M.constraint(
                Expr.vstack(1.0, 0.5 * t, Expr.mul(Xp, B)),
                Domain.inRotatedQCone()
            )


        # L1 penalty term μ∑ u_i
        term4 = mu * Expr.sum(u)

        # Cardinality constraint: at most k nonzeros
        M.constraint(Expr.sum(z) <= k)

        # Relating u and B: |β_i| ≤ u_i
        M.constraint(B <= u)
        M.constraint(-B <= u)

        # Either-or condition: z_i = 1 OR β_i = 0
        M.disjunction((z == 1) | (B == 0))

        # Final objective: min (yᵀy - 2 yᵀXβ + βᵀQβ + μ‖β‖₁)
        M.objective(ObjectiveSense.Minimize, term2 + t + term4 + term1)

        # Export model and solve
        M.writeTask("RankOne.ptf")
        s = time.time()
        M.solve()
        elapsed = time.time() - s
        print(f"Elapsed Time: {elapsed:.4f} s")
        print(f"Objective value: {M.primalObjValue():.4f}")
        # Print solution status
        ModelStatus = M.getPrimalSolutionStatus()
        print("Model Status: ", ModelStatus)

        return M
    return (RankOne,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The described model is implemented using Mosek above.

    ```python
        M.disjunction((z == 1) | (B == 0))
    ```

    The line above illustrates how **Mosek can implement disjunction constraints** without requiring any additional transformations, such as using Big-M values. 

    To run the model, we use the *diabetes dataset* shared by the referenced paper. We download, and load the dataset online in the code block below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Loading the Data""")
    return


@app.cell
def _(StandardScaler, glob, io, os, pd, requests, zipfile):
    # URL of the data
    url = "https://atamturk.ieor.berkeley.edu/data/sparse.regression/data.sparse.regression.zip"

    # Download and unzip
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    extract_dir = "data_sparse_regression"
    z.extractall(extract_dir)  # extracts to a local folder

    # Find a file that looks like the diabetes dataset (e.g., diabetes*.csv / .txt)
    candidates = glob.glob(os.path.join(extract_dir, "**", "diabetes*.*"), recursive=True)
    if not candidates:
        # If you know the exact name, set it here:
        # file_path = os.path.join(extract_dir, "data.sparse.regression", "diabetesQ.csv")
        raise FileNotFoundError("No file matching 'diabetes*' was found in the extracted contents.")

    file_path = candidates[0]
    print("Using file:", file_path)

    # Read the dataset (let pandas sniff the delimiter if not comma)
    df = pd.read_csv(file_path, sep=None, engine="python")
    print("Columns in dataset:", df.columns.tolist())

    # Assuming the last column is the target variable
    X_nonSTD = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_nonSTD = X_nonSTD.astype(float)

    # Standardize features: zero mean, unit variance
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X_nonSTD)

    print("Shape of X (features):", X.shape)
    print("Shape of y (target):", y.shape)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then using the loaded data, we can run the problem. As it can be seen from the printed out message below, the model is solved to feasibility.""")
    return


@app.cell
def _(RankOne, X, y):
    n,p = X.shape[0],X.shape[1]
    lmb = 5
    mu = 1
    k = p*0.8

    RankOne(n, p, lmb, mu, y , X, k,True)
    return


if __name__ == "__main__":
    app.run()