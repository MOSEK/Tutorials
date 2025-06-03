import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A *geometric program (GP)* is a type of mathematical optimization problem characterized by objective and constraint functions that have a special form. To describe the said special form, one will require the following definitions:

        The smallest terms creating the GPs are called *monomials*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Monomial functions

        Let \( x_1, \dots, x_n \) denote \( n \) real positive variables, and \( x = (x_1, \dots, x_n) \) a vector with components \( x_i \). A real-valued function \( f \) of \( x \), with the form

        \[
        f(x) = c x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n}
        \]

        where \( c > 0 \) and \( a_i \in \mathbb{R} \), is called a **monomial function**, or more informally, a **monomial** (of the variables \( x_1, \dots, x_n \)). We refer to the constant \( c \) as the **coefficient** of the monomial, and we refer to the constants \( a_1, \dots, a_n \) as the **exponents** of the monomial.

        As an example, \( 2.3 x_1^2 x_2^{-0.15} \) is a monomial of the variables \( x_1 \) and \( x_2 \), with coefficient 2.3 and \( x_2 \)-exponent \( -0.15 \).

        Any positive constant is a monomial, as is any variable. Monomials are closed under multiplication and division: if \( f \) and \( g \) are both monomials, then so are \( f * g \) and \( f / g \) (this includes scaling by any positive constant). A monomial raised to any power is also a monomial:

        \[
        f(x)^y = (c x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n})^y = c^y x_1^{a_1 y} x_2^{a_2 y} \cdots x_n^{a_n y}
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Posynomial functions

        A sum of one or more monomials, i.e., a function of the form

        $$
        f(x) = \sum_{k=1}^{K} c_k x_1^{a_{1k}} x_2^{a_{2k}} \cdots x_n^{a_{nk}},
        $$

        where $c_k > 0$, is called a *posynomial function* or, more simply, a *posynomial* (with $K$ terms, in the variables $x_1, \ldots, x_n$). Any monomial is also a posynomial. 

        For example each of the following terms are monomials (thus also posynomials)

        $$
        0.5{x}{y}, \quad 2x^2\sqrt{z}, \quad xyz
        $$

        And the sumation of these monomials is a polynomial.

        $$
        0.5{x}{y} + 2x^2\sqrt{z} + xyz
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Standard Form of a Geometric Program

        A **Geometric Program (GP)** is an optimization problem of the following form:

        \[
        \begin{aligned}
        \text{minimize} \quad & f_0(x) \\
        \text{subject to} \quad & f_i(x) \leq 1, \quad i = 1, \dots, m \\
        & x_j > 0, \quad j = 1, \dots, n
        \end{aligned}
        \]

        Where:

        - \( f_0 : \mathbb{R}_{+}^n \to \mathbb{R}_{+} \) is a **posynomial objective function**,
        - Each \( f_i : \mathbb{R}_{+}^n \to \mathbb{R}_+ \) is a **posynomial inequality constraint**.

        The problem is **not convex** in this form due to the nonlinear structure of monomials and posynomials. However, it can be convexified using a logarithmic transformation and can also be represented in an exponential cone. The next section provides a tool that solves the geometric programs, handling the transformations automatically.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Geometric Programming Tool

        The tool initializes an optimization model and provides the following core functions for model construction and analysis:

        - **`printModel()`** — Outputs the current model structure in a readable format.
        - **`addObjective(Monomial)`** — Defines and adds the objective function to the model.
        - **`addConstraint(List of Monomials)`** — Adds one or more constraints to the model formulation.
        - **`Solve()`** — Solves the model once all components (objective and constraints) are defined.
        - **`analyse()`** — Prints key specifications and diagnostics of the model.
        - **`evaluateObjective()`** — Computes the value of the objective function for the obtained solution.

        The class and function definitions, as well as the model implementation, can be found in the Appendix.

        With only using these functionalities, one can solve a geometric program without further implementation. Let's go over an example. Say we model the following problem: 

        $\quad\quad\quad$ Minimize:

        $$
        x + y^2 z
        $$

        $\quad\quad\quad$ Subject to:

        $$
        0.1\sqrt{x} + 2y^{-1} \leq 1,
        $$

        $$
        z^{-1} + yx^{-2} \leq 1.
        $$


        To initialize a geometric program, call the *GeometricProgramming(m,n)* class, with inputs m = number of constraint, n = number of variables.
        """
    )
    return


@app.cell
def _(GeometricProgramming):
    gp = GeometricProgramming(2, 3) # m,n = (#) of constraints, (#) of variables
    return (gp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        The objective function in a geometric program is, by definition, minimized. The `addObjective([Monomial, ...])` function accepts either a single `Monomial` object or a list of `Monomial` objects as input.

        A `Monomial` object can be constructed using the following method:

        `Monomial.add(coefficient, indices, alphas)`

        - **`coefficient`**: \( c \), a positive scalar  
        - **`indices`**: an array of selected variable indices \( i \in \{1, \dots, N\} \)  
          _Example_: To add a monomial involving \( y \) and \( z \) from the list \( x, y, z, w \), input `indices = [1,2]`  
        - **`alphas`**: an array of exponents \( a_i \) corresponding to each selected variable
        """
    )
    return


@app.cell
def _(Monomial, gp):
    gp.addObjective([
        Monomial.add(1, [0], [1]), # x
        Monomial.add(1, [1,2], [2,1]) # y^2 * z
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By definition, all constraint posynomials in a geometric program must be less than or equal to 1. Similar to the objective function, constraints can be added using the `addConstraint(constraintIndex, [Monomial, ...])` function. 

        This function accepts a constraint index and either a single `Monomial` object or a list of `Monomial` objects as input. Monomials can be added individually or collectively in a single call.
        """
    )
    return


@app.cell
def _(Monomial, gp):
    #First Constraint, <= 1
    gp.addConstraint(0, [
        Monomial.add(0.1, [0], [0.5]), # 0.1 * x^(0.5)
        Monomial.add(2, [1], [-1]) # 2 * y^-1
    ])

    #Second Constraint - First Monomial 
    gp.addConstraint(1, Monomial.add(1, [2], [-1])) # z^-1

    #Second Constraint - Second Monomial, <= 1
    gp.addConstraint(1, Monomial.add(1, [0,1], [-2,1])) #x^-2 * y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `printModel()` function provides a convenient way to display the current model, including the objective function and all defined constraints, in a readable format.""")
    return


@app.cell
def _(gp):
    gp.printModel()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Once the entire model has been defined, the problem can be solved using the `Solve()` function. This function returns both the solution object and the values of the decision variables. If needed, the user can further interact with the solution object for additional information.

        Using the retrieved decision variable values, the user can also compute the objective function value.
        """
    )
    return


@app.cell
def _(gp):
    sol, variables = gp.Solve() #solve the model
    return sol, variables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The objective function value can be calculated using the `evaluateObjective(variables)` function, based on the decision variable values obtained from the solution.""")
    return


@app.cell
def _(gp, np, sol, variables):
    obj_val = gp.evaluateObjective(variables) #Calculate the objective value
    print(f"Objective value: {obj_val:.6f}")

    print("Optimal values of (x_i):")
    for i, val in enumerate(sol):
        print(f"x_{i} = {np.exp(val):.6f}")
    return i, obj_val, val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyser""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Writing well-structured geometric programs can sometimes be a non-trivial task. To assist with this, the analyse() function is included in the tool. This function provides a comprehensive summary and diagnostic of the model, helping users identify potential issues before solving.

        - **Constraints and Variables**: Prints the total number of constraints and decision variables.
        - **Monomial Coefficients**: Shows the range of monomial coefficients (min–max), if any are added.
        - **Alpha Values**: Displays the range of non-zero absolute exponent values (alphas).
        - **Degree of Difficulty**: `totalMonomials - number of variables - 1`
        - **Variable Sign Check**: Issues warnings if a variable appears only with positive or only with negative exponents in all monomials, which helps identify potential modeling issues.

        <ins>Variable Sign Check</ins> is a critical part of the model analysis, as it determines the potential solvability issues of the geometric program and can often be overlooked.

        If, for a decision variable, all the defined alpha values (exponents) have the same sign—either all positive or all negative—this may indicate a structural issue in the model that could hinder its solvability. When such a problem is encountered, the analyser function can help the user to identify the problem. 

        Let's look at an example that illustrates this issue. Firstly the model below displays an example where the decision variable $x$ combines both positive and negative alpha values. Therefore, MOSEK can retrieve the optimal solution easily.
        """
    )
    return


@app.cell
def _(GeometricProgramming, Monomial):
    gp2 = GeometricProgramming(0, 1) # m,n = (#) of constraints, (#) of variables
    gp2.addObjective([
        Monomial.add(1, [0], [2]), # x^2
        Monomial.add(1, [0], [-3])]) # y^ (-3)

    gp2.printModel()
    sol2,variables2 = gp2.Solve() #solve the model
    return gp2, sol2, variables2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As seen below, the analyse function does not warn the user, only presents the characteristics of the model.""")
    return


@app.cell
def _(gp2):
    gp2.analyse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In the modified model below, the second monomial is discarded leaving the decision variable $x$ only with a positive alpha value. Therefore, the $x$ value will converge to _zero_ while being strictly bigger then _zero_, resulting in the absence of an optimal solution.""")
    return


@app.cell
def _(GeometricProgramming, Monomial):
    gp3 = GeometricProgramming(0, 1) # m,n = (#) of constraints, (#) of variables
    gp3.addObjective(Monomial.add(1, [0], [2])) # x^2

    gp3.printModel()
    sol3,variables3 = gp3.Solve() #solve the model
    return gp3, sol3, variables3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Due to the described problem, the analyser warns the user about the weakness of the model.""")
    return


@app.cell
def _(gp3):
    gp3.analyse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly to the example before, if the decision variable $x$ is only left with a negative alpha value we will encounter infeasibility again. It can be easily seen that decision variable $x$ will converge to infinity, making the problem unbounded. When called, the analyser again warns the user about the mentioned property.""")
    return


@app.cell
def _(GeometricProgramming, Monomial):
    gp4 = GeometricProgramming(0, 1) # m,n = (#) of constraints, (#) of variables
    gp4.addObjective(Monomial.add(1, [0], [-3])) # x^(-3)

    gp4.printModel()
    sol4,variables4 = gp4.Solve() #solve the model
    gp4.analyse()
    return gp4, sol4, variables4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Although this property can sometimes prevent a model from being solvable, there are cases where it can still be handled correctly. The following example presents a model with two decision variables, one of which appears only with positive exponents. In this case, due to the structure of the feasible region, an optimal solution exists. 

        Nevertheless, this property should be treated as a potential risk and considered a diagnostic checkpoint, especially if the model fails. It can be flagged using the analyser function for early detection.
        """
    )
    return


@app.cell
def _(GeometricProgramming, Monomial):
    gp5 = GeometricProgramming(1, 2) # m,n = (#) of constraints, (#) of variables
    gp5.addObjective([Monomial.add(1, [0], [2]), #x^2
                      Monomial.add(1, [1], [-1])]) # y^(-1)

    gp5.addConstraint(0,[Monomial.add(1,[0],[1]), Monomial.add(1,[1],[1])])

    gp5.printModel()
    sol5,variables5 = gp5.Solve() #solve the model
    gp5.analyse()
    return gp5, sol5, variables5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further Details

        As explained earlier, the problem is non-convex. To solve the problem, the following transformation from the non-convex form to an exponential cone representation can be applied, which is already handled by the geometric programming class.

        ### Logarithmic Transformation (Convexification)

        To reformulate a GP as a **convex problem**, we apply a change of variables:

        \[
        x_j = e^{y_j} \quad \text{for } j = 1, \dots, n
        \quad \Rightarrow \quad x = e^y, \quad y \in \mathbb{R}^n
        \]

        A **monomial** becomes:

        \[
        f(x) = c \cdot e^{a^T y} \quad \Rightarrow \quad \log f(x) = \log c + a^T y
        \]

        A **posynomial** becomes:

        \[
        f(x) = \sum_{k=1}^K c_k \cdot e^{a_k^T y} \quad \Rightarrow \quad \log f(x) = \log \left( \sum_{k=1}^K \exp(a_k^T y + \log c_k) \right)
        \]

        This is known as the **log-sum-exp function**, which is a **convex** function of \( y \in \mathbb{R}^n \).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Convex Reformulation

        The original GP is now transformed into the following **convex optimization problem**:

        \[
        \begin{aligned}
        \text{minimize} \quad & t \\
        \text{subject to} \quad & \log\left( \sum_{k=1}^{K_0} \exp(a_k^{(0)\top} y + \log c_k^{(0)}) \right) \leq t \\
        & \log\left( \sum_{k=1}^{K_i} \exp(a_k^{(i)\top} y + \log c_k^{(i)}) \right) \leq 0, \quad i = 1, \dots, m
        \end{aligned}
        \]

        Where \( t \in \mathbb{R} \) is a **scalar upper bound variable**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exponential Cone Representation

        The objective function constraint of the form:

        \[
        \log\left( \sum_{k=1}^{K_0} \exp(a_k^{(0)\top} y + \log c_k^{(0)}) \right) \leq t
        \]

        can be equivalently written as:

        \[
        \sum_{k} u_k \leq 1, \quad \text{with} \quad (u_k, 1, (a_k^{(0)\top} y + \log c_k^{(0)} - \tau) \in \mathcal{K}_{\exp}
        \]

        Where:

        \[
        \mathcal{K}_{\exp} := \left\{ (x, y, z) \in \mathbb{R}^3 \;\middle|\; y > 0,\; y \cdot e^{x/y} \leq z \right\}
        \]

        And each constraint of the form:

        \[
        \log\left( \sum_{k=1}^{K_i} \exp\left(a_k^{(i)\top} y + \log c_k^{(i)}\right) \right) \leq 0
        \]

        can be equivalently written as:

        \[
        \sum_k u_k \leq 1, \quad \text{with} \quad (u_k, 1, a_k^{(i)\top} y + \log c_k^{(i)}) \in \mathcal{K}_{\exp}
        \]

        Where:

        \[
        \mathcal{K}_{\exp} := \left\{ (x, y, z) \in \mathbb{R}^3 \;\middle|\; y > 0,\; y \cdot e^{x/y} \leq z \right\}
        \]

        This transforms the log-sum-exp constraint into a set of **conic constraints**, using **auxiliary variables** \( u_k \) and the **exponential cone**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        Then we can apply this transformation to our original model: 

        $\quad\quad\quad$ Minimize:

        $$
        x + y^2 z
        $$

        $\quad\quad\quad$ Subject to:

        $$
        0.1\sqrt{x} + 2y^{-1} \leq 1,
        $$

        $$
        z^{-1} + yx^{-2} \leq 1.
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let:

        \[
        x = e^u, \quad y = e^v, \quad z = e^w
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Then we can apply the following transformation:


        $\quad\quad\quad$ Minimize:

        \[
        t
        \]

        $\quad\quad\quad$ Subject to:

        \[
        \log\left(e^u + e^{2v + w}\right) \leq t
        \]

        \[
        \log\left(0.1e^{u/2} + 2e^{-v}\right) \leq 0
        \]

        \[
        \log\left(e^{-w} + e^{v - 2u}\right) \leq 0
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $\quad\quad\quad$ Minimize:

        \[
        t
        \]

        $\quad\quad\quad$ Subject to:

        \[
        (p_1, 1, u - t), \quad (q_1, 1, 2v + w - t) \in \mathcal{K}_{\exp}, \quad p_1 + q_1 \leq 1,
        \]

        \[
        (p_2, 1, 0.5u + \log(0.1)), \quad (q_2, 1, -v + \log(2)) \in \mathcal{K}_{\exp}, \quad p_2 + q_2 \leq 1,
        \]

        \[
        (p_3, 1, -w), \quad (q_3, 1, v - 2u) \in \mathcal{K}_{\exp}, \quad p_3 + q_3 \leq 1.
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Summary of Variable Domains

        | Symbol | Meaning | Domain |
        |--------|---------|--------|
        | \( x \) | Original decision variables | \( \mathbb{R}_{+}^n \) |
        | \( y \) | Log-transformed variables \( y = \log x \) | \( \mathbb{R}^n \) |
        | \( c_k \) | Monomial/posynomial coefficients | \( \mathbb{R}_{+} \) |
        | \( a_k \) | Monomial exponents | \( \mathbb{R}^n \) |
        | \( t \) | Scalar upper bound variable | \( \mathbb{R} \) |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## References

        Some of the definitions used in this notebook are taken from the work [A Tutorial on Geometric Programming](https://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf) by Boyd et al.

        You can also read more about Geometric Programming and the example used in the [MOSEK Modeling Cookbook](https://docs.mosek.com/modeling-cookbook/expo.html#geometric-programming).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""#Appendix""")
    return


@app.cell
def _(mo):
    mo.md(r"""The class definitions, used functions and model implementation can be examined below.""")
    return


@app.cell
def _():
    import numpy as np
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense
    import mosek.fusion.pythonic

    class Monomial:
        def __init__(self, coefficient=None, indices=None, alphas=None, n=None):
            self.coefficient = coefficient
            self.indices = indices
            self.alphas = [0] * n if n is not None else None
            if indices is not None and alphas is not None and n is not None:
                for idx, alpha in zip(indices, alphas):
                    self.alphas[idx] = alpha

        @classmethod
        def add(cls, coefficient, indices, alphas):
            obj = cls(coefficient=coefficient, indices=indices, alphas=None)
            obj.indices_alphas = alphas  # Store original input to reconstruct later
            return obj

        def defineMonomial(self):
            terms = []
            has_coefficient = self.coefficient != 1
            for i, alpha in enumerate(self.alphas):
                if alpha != 0:
                    terms.append(f"(x_{i} ^ {alpha})")

            joined = " * ".join([f"{self.coefficient}" if has_coefficient else ""] + terms if has_coefficient else terms)

            if has_coefficient or len(terms) > 1:
                return f"({joined.strip()})"
            else:
                return joined.strip()

        @staticmethod
        def definePosynomial(monomials):
            return " + ".join([m.defineMonomial() for m in monomials])

    class GeometricProgramming:
        def __init__(self, m, n):
            self.m = m
            self.n = n
            self.constraintsMonomials = [[] for _ in range(m)]
            self.objectiveMonomials = []
            self.totalMonomials = 0

        def _fillAlphas(self, monomial):
            # Check all alphas before assigning
            for alpha in monomial.indices_alphas:
                if alpha == 0:
                    raise ValueError(f"Invalid alpha value {alpha}: all exponents must be non-zero.")

            # Safe to assign now
            monomial.alphas = [0] * self.n
            for idx, alpha in zip(monomial.indices, monomial.indices_alphas):
                monomial.alphas[idx] = alpha

        def _checkDuplicateIndex(self,monomial):
            seen = set()
            for idx in monomial.indices:
                    if idx in seen:
                        raise ValueError("Input indices has to be unique.")
                    seen.add(idx)

        def addObjective(self, monomial):
            if isinstance(monomial, list):
                for m in monomial:
                    self.addObjective(m)
            else:
                if monomial.coefficient <= 0:
                    raise ValueError("Coefficient must be positive")

                self._checkDuplicateIndex(monomial)
                self._fillAlphas(monomial)
                self.objectiveMonomials.append(monomial)
                self.totalMonomials += 1

        def addConstraint(self, index, monomial):
            if index >= self.m:
                raise ValueError(f"Constraint index {index} out of range")
            if isinstance(monomial, list):
                for m in monomial:
                    self.addConstraint(index, m)
            else:
                if monomial.coefficient <= 0:
                    raise ValueError("Coefficient must be positive")

                self._checkDuplicateIndex(monomial)
                self._fillAlphas(monomial)
                self.totalMonomials += 1
                self.constraintsMonomials[index].append(monomial)

        def evaluateObjective(self, x_values):
            if len(x_values) != self.n:
                raise ValueError(f"x_values length {len(x_values)} does not match problem dimension {self.n}")
            objective_val = 0.0
            for mon in self.objectiveMonomials:
                term = mon.coefficient
                for i, alpha in enumerate(mon.alphas):
                    term *= x_values[i] ** alpha
                objective_val += term
            return objective_val

        def analyse(self):
            print("--------- Model Analysis ---------")
            print("# of constraints: ", self.m)
            print("# of variables: ", self.n)

            all_monomials = self.objectiveMonomials + [m for sublist in self.constraintsMonomials for m in sublist]
            coefficients = [mon.coefficient for mon in all_monomials]

            if coefficients:
                print(f"Range of coefficients: [{min(coefficients)}, {max(coefficients)}]")
            else:
                print("No monomials added yet.")

            all_alphas = [abs(alpha) for mon in all_monomials for alpha in mon.alphas if alpha != 0]

            if all_alphas:
                print(f"Range of absolute alpha values: [{min(all_alphas)}, {max(all_alphas)}]")
            else:
                print("No alpha values found.")

            print("Degree of Difficulty: ", self.totalMonomials - self.n - 1, "\n")

            for i in range(self.n):
                signs = []
                for mon in all_monomials:
                    alpha = mon.alphas[i]
                    if alpha > 0:
                        signs.append(1)
                    elif alpha < 0:
                        signs.append(-1)

                if signs:
                    if all(s > 0 for s in signs):
                        print(f"Warning: Variable x_{i} only has positive exponents across all monomials.")
                    elif all(s < 0 for s in signs):
                        print(f"Warning: Variable x_{i} only has negative exponents across all monomials.")

        def printModel(self):
            print("Objective function:")
            print(f"\tMinimize: {Monomial.definePosynomial(self.objectiveMonomials)}")

            if any(self.constraintsMonomials):
                print("\nConstraints:")
                for i, con in enumerate(self.constraintsMonomials):
                    if con:
                        print(f"\tConstraint {i+1}: {Monomial.definePosynomial(con)} <= 1")
            else:
                print("\nNo constraints defined.")

            print(" ")

        def Solve(self, fileName=None):

            if not self.objectiveMonomials:
                raise ValueError("No objective monomials defined.")

            M = Model()
            x = M.variable("x", self.n)
            t = M.variable("t", 1)
            p = M.variable("p", self.totalMonomials)

            currentMonomialIndex = 0

            # Objective
            dummySum = []
            objectiveMonomials = []
            noOfMono = len(self.objectiveMonomials)
            j = 0

            if noOfMono > 1:
                for mon in self.objectiveMonomials:
                    expr = Expr.dot(mon.alphas, x.T) + np.log(mon.coefficient) - t
                    objectiveMonomials.append(expr)
                    dummySum.append(p[currentMonomialIndex + j])
                    j += 1
                M.constraint("Objective Definition", Expr.hstack(
                    p.slice(currentMonomialIndex, currentMonomialIndex + noOfMono),
                    Expr.ones(noOfMono),
                    Expr.vstack(objectiveMonomials)) == Domain.inPExpCone())
                M.constraint("Objective Dummy Sum", Expr.sum(Expr.vstack(dummySum)) <= 1)
            else:
                M.constraint("Objective Definition", Expr.dot(self.objectiveMonomials[0].alphas, x.T) <= t)

            currentMonomialIndex += noOfMono

            # Constraints (only if defined)
            for i, monomials in enumerate(self.constraintsMonomials):
                if not monomials:
                    continue  # Skip empty constraint slots

                noOfMono = len(monomials)
                constraintMonomials = []
                dummySum = []
                j = 0
                if noOfMono > 1:
                    for mon in monomials:
                        expr = Expr.dot(mon.alphas, x.T) + np.log(mon.coefficient)
                        constraintMonomials.append(expr)
                        dummySum.append(p[currentMonomialIndex + j])
                        j += 1

                    M.constraint("Conic-Constraint Definition" + str(i), Expr.hstack(
                        p.slice(currentMonomialIndex, currentMonomialIndex + noOfMono),
                        Expr.ones(noOfMono),
                        Expr.vstack(constraintMonomials)) == Domain.inPExpCone())
                    M.constraint("Constraint Dummy Sum" + str(i), Expr.sum(Expr.vstack(dummySum)) <= 1)
                else:
                    M.constraint("Constraint Definition" + str(i), Expr.dot(monomials[0].alphas, x.T) + np.log(monomials[0].coefficient) <= 0)

                currentMonomialIndex += noOfMono

            M.objective("Objective Function", ObjectiveSense.Minimize, t)
            M.solve()

            if fileName:
                M.writeTask(fileName + ".ptf")

            ModelStatus = M.getPrimalSolutionStatus()
            ProblemStatus = M.getProblemStatus()
            if ModelStatus == mosek.fusion.pythonic.SolutionStatus.Optimal:
                sol = x.level()
                vars = np.exp(sol)
                print("Model Status: Optimal")

            else:
                sol = None
                vars = None
                print("The problem does not have an optimal solution.")
                print("Problem Status: ", ProblemStatus)

            return sol, vars
    return (
        Domain,
        Expr,
        GeometricProgramming,
        Model,
        Monomial,
        ObjectiveSense,
        mosek,
        np,
    )


if __name__ == "__main__":
    app.run()