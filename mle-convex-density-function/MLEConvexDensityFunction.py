import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )
    """)
    return


@app.cell
def _():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Var
    import mosek.fusion.pythonic
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(5)
    import sys
    import marimo as mo
    import time
    return Domain, Expr, Model, ObjectiveSense, Var, mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is based on the work of Terlaky and Vial in ["Computing Maximum Likelihood Estimators of Convex Density Functions"](https://doi.org/10.1137/S1064827595286578). The following section describes the problem definition, which is taken nearly verbatim from the paper. You can also read about this problem in the [MOSEK Modeling Cookbook](https://docs.mosek.com/modeling-cookbook/powo.html#maximum-likelihood-estimator-of-a-convex-density-function). We also explain step by step how the nonlinear model is transformed into a conic model which can be fed into MOSEK.

    The problem addressed in the paper is to estimate a density function that is known to be convex. Indeed, the paper suggests using a maximum likelihood estimator, which is a solution to a linearly constrained optimization problem. Let \( Y \) be the real-valued random variable with unknown, convex density function \( g \). We want to estimate \( g : \mathbb{R}_+ \to \mathbb{R}_+ \) from observed samples.

    Let \( \{y_1, \dots, y_n\} \) be an ordered sample of \( n \) outcomes of \( Y \). We shall assume that \( y_1 < y_2 < \dots < y_n \). The estimator of \( g \geq 0 \) is a piecewise linear function \( \tilde{g} : [y_1, y_n] \to \mathbb{R}_+ \) with break points at \( (y_i, \tilde{g}(y_i)) \), \( i = 1, \dots, n \).

    Let \( x_i > 0 \), \( i = 1, \dots, n \), be the estimator of \( g(y_i) \). The objective is to maximize the likelihood function

    \[
    \text{maximize} \quad f(x) = \prod_{i=1}^{n} x_i.
    \]

    To match the convexity requirement for the one-dimensional density function, we add the constraint that the slope of \( \tilde{g} \) is nondecreasing. This is written as

    \[
    \frac{\Delta x_i}{\Delta y_i} \leq \frac{\Delta x_{i+1}}{\Delta y_{i+1}}, \quad i = 2, \dots, n-1,
    \]

    where \( \Delta x_i = x_i - x_{i-1} \) and \( \Delta y_i = y_i - y_{i-1} \).

    This can be transformed into the following constraint, which will be the first constraint of the model.

    \[
    \Delta y_{i+1} x_{i-1} + (\Delta y_i + \Delta y_{i+1}) x_i - \Delta y_i x_{i+1} \leq 0, \quad i = 2, \dots, n-1
    \]


    We also have the requirement that \( \tilde{g} \) is a density function: the area below \( \tilde{g} \) must sum up to one, which is the second constraint of the model.

    \[
    \sum_{i=1}^{n-1} \Delta y_{i+1} \left( \frac{x_i + x_{i+1}}{2} \right) = 1.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use the following 4 distributions to test our implementation. In the applied examples, the exponential distribution will be used to generate random data. Other distributions can also be tested easily using the following cells if desired.
    """)
    return


@app.cell
def _(np):
    # Define density functions, g, to sample y(i) points
    # The density is e^(-z)
    def sample_exponential(n):
        return np.random.exponential(scale=1.0, size=n)

    # Arcsine distribution: range (0 < z < 1), density: 1 / (π√(z(1-z)))
    def sample_arcsine(n):
        U = np.random.uniform(0, 1, n) # Generate uniform samples
        return np.arcsin(U) ** 2 

    # Quadratic distribution: range (0 ≤ z ≤ 2), density: 1 - z / 2
    def sample_quadratic(n):
        U = np.random.uniform(0, 1, n) # Generate uniform samples
        return 2 - np.sqrt(4 - 4 * U)  

    # Inverse distribution: range (1 ≤ z), density: 1 / z²
    def sample_inverse(n):
        u = np.random.uniform(0, 1, n)  # Generate uniform samples
        return 1 / (1 - u)
    return sample_arcsine, sample_exponential, sample_inverse, sample_quadratic


@app.cell(hide_code=True)
def _(mo):
    NodeSize = mo.ui.number(40)
    mo.md(f"Choose the amount you want to generate : {NodeSize})")
    return (NodeSize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The $y$ values are assumed to be non-decreasing, thus the generated values are sorted.
    """)
    return


@app.cell
def _(
    NodeSize,
    np,
    sample_arcsine,
    sample_exponential,
    sample_inverse,
    sample_quadratic,
):
    # Number of samples
    n = NodeSize.value

    # Generate samples from different density functions
    exponential_samples = sample_exponential(n)
    arcsine_samples = sample_arcsine(n)
    quadratic_samples = sample_quadratic(n)
    inverse_samples = sample_inverse(n)

    # Select which sample to use
    y = exponential_samples

    # In the paper the y's are assumed to be in non-decreasing order
    y.sort()

    # As parameter of the model, we need the delta values array where delta_y = y[i] - y[i-1]
    delta_y = np.diff(y)
    return delta_y, n, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exponential Cone Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To solve the described problem, one approach could be to apply an exponential cone transformation. To achieve this, we can take the natural logarithm of the objective function. Although this addition will change the obtained objective function value, the optimal solution will remain the same. Then, we can derive the following:

    \[
    \ln \left( \prod_{i=1}^{n} x_i \right) =  \sum_{i=1}^{n} \ln (x_i)
    \]

    Then our objective function will turn into:

    \[
    \text{maximize} \quad f(x) = \sum_{i=1}^{n} \ln (x_i) \
    \]

    \[
    \text{or equivalently}
    \]

    \[
    \text{minimize} f(x) = \sum_{i=1}^{n} -\ln (x_i)
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then as a second step, we can turn the model into the following:

    \[
    \text{minimize} \quad f(x) = \sum_{i=1}^{n} u_i
    \]

    where u satisfies:

    \[
    u_i \geq -\ln x_i
    \]

    The rest of the constraints in the original model remain the same. Now, we have an additional constraint for u definition with the defined objective function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, as the last step of our transformation, we model the logarithm with an exponential cone.

    The exponential cone is the set of triples \( (x_1,x_2,x_3) \) satisfying

    \[
    x_1 \geq x_2 e^{x_3 / x_2}, \quad x_2 \geq 0
    \]

    With this definition on hand, the transformation can easily be done with the following vector:

    $$\begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3
    \end{bmatrix}
    =
    \begin{bmatrix}
    x \\
    1 \\
    -u
    \end{bmatrix}
    $$

    The exponential cone condition in this case evaluates to:

    \[
    (1) \quad  x \geq e^{-u}
    \]

    \[
    (2) \quad  \ln x \geq -u
    \]

    \[
    (3) \quad  -\ln x \leq u
    \]

    and we get the desired relationship between u and x.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We end up with the following final model:

    \[
    \text{min} \quad f(x) = \sum_{i=1}^{n} u_i
    \]

    \[
    \Delta y_{i+1} x_{i-1} + (\Delta y_i + \Delta y_{i+1}) x_i - \Delta y_i x_{i+1} \leq 0, \quad i = 2, \dots, n-1
    \]

    \[
    \sum_{i=1}^{n-1} \Delta y_{i+1} \left( \frac{x_i + x_{i+1}}{2} \right) = 1.
    \]


    $$
    \begin{bmatrix}
    x \\
    1 \\
    -u
    \end{bmatrix} \in K_{EXP}
    $$
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, time):
    def ExponentialCone(delta_y,n,weight):
        start = time.time()
        M = Model()

        # Decision Variable for Conic Transformation 
        u = M.variable("u", n ,Domain.unbounded())
        # The decision variable x[i], is the estimator of g(y[i])
        x = M.variable("x", n ,Domain.unbounded())

        # The slope of g is ensured to be non-decreasing, first constraint 
        M.constraint("Slope Integrity", - Expr.mulElm(delta_y[1:n] , x[0:n-2]) + 
                    Expr.mulElm((delta_y[0:n-2] + delta_y[1:n]) , x[1:n-1]) - 
                    Expr.mulElm(delta_y[0:n-2] , x[2:n]) <= 0 )

        # The area below our density function, g, must sum up to 1, second constraint
        M.constraint("Probability Sum",  Expr.dot(((x[0:n-1] + x[1:n]) / 2 ) , delta_y) == 1.0)

        # Exponential Cone Transformation Definition
        M.constraint("Cone Definition", Expr.hstack(x,Expr.ones(n),-u), Domain.inPExpCone())

        M.objective("Objective Function", ObjectiveSense.Minimize, Expr.dot(weight,u))
        M.solve()
        sol = x.level()
        ModelStatus = M.getPrimalSolutionStatus()
        print(ModelStatus)
        end = time.time()

        iterations = M.getSolverIntInfo("intpntIter")
        print(f"Number of interior-point iterations: {iterations}")

        M.writeTask("MLE_ExponentialCone.ptf")
        print("Elapsed time: ", round(end - start,4)," s.")
        return sol
    return (ExponentialCone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also define a function to plot the fit of our estimator.
    """)
    return


@app.cell
def _(plt):
    def PlotFigure(y,sol):
        plt.figure(figsize=(10, 6))
        plt.plot(y, sol, label='Y Array', color='blue', marker='o', markersize=4, linewidth=1)
        plt.title('Plot of Y Array')
        plt.xlabel('Index')
        plt.ylabel('Y Values')
        plt.grid(True)
        plt.legend()
        plt.show()
    return (PlotFigure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The retreived result of the model is reported.
    """)
    return


@app.cell
def _(ExponentialCone, delta_y, n, np):
    sol = ExponentialCone(delta_y,n,np.ones(n))
    return (sol,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    As seen in the figure, the estimator outputs a curve resembling the exponential distribution.
    """)
    return


@app.cell
def _(PlotFigure, sol, y):
    PlotFigure(y,sol)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Power Cone Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, one could also solve the original model using power cone reformulation. Let's consider again maximizing the likelihood function:

    \[
    \text{maximize} \quad f(x) = \prod_{i=1}^{n} x_i.
    \]

    We can use geometric mean transformation to reach the power cone representation. The definition of the power cone \( \mathcal{P}_{n}^{\alpha_1, \dots, \alpha_m} \), where \( \alpha_i > 0 \) and \( \sum \alpha_i = 1 \)}  is the following:


    \[
    x_1^{\alpha_1} \cdots x_m^{\alpha_m} \geq \sqrt{x_{m+1}^2 + \dots + x_n^2}, \quad x_1, \dots, x_m \geq 0
    \]

    Maximizing \( f(x) \) is the same as solving the problem

    \[
    \text{maximize} \quad t
    \]

    \[
     \quad \prod_{i=1}^{n} x_i^{1 \over n} \geq t
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This last version fits the power cone definition, thus the transformation can be applied directly. The vector defining the power cone is added to the model. At last the full model will look as the following:

    \[
    \text{maximize} \quad t
    \]

    \[
    \Delta y_{i+1} x_{i-1} + (\Delta y_i + \Delta y_{i+1}) x_i - \Delta y_i x_{i+1} \leq 0, \quad i = 2, \dots, n-1
    \]

    \[
    \sum_{i=1}^{n-1} \Delta y_{i+1} \left( \frac{x_i + x_{i+1}}{2} \right) = 1.
    \]

    $$
    \begin{bmatrix}
    x_1 \\
    x_2 \\
    \dots \\
    x_n \\
    t
    \end{bmatrix}
     \in \mathcal{P}_{n+1}^{1/n,\ldots,1/n} $$

    This special instance of the power cone appears also in MOSEK as the [geometric mean cone](https://docs.mosek.com/modeling-cookbook/powo.html#the-power-cone-s).
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, Var, n, time):
    def PowerCone(delta_y):
        start = time.time()
        M = Model()

        # Decision Variable for Conic Transformation 
        u = M.variable("u", 1 ,Domain.unbounded())
        # The decision variable x[i], is the estimator of g(y[i])
        x = M.variable("x", n ,Domain.greaterThan(0.0))

        # The slope of g is ensured to be non-decreasing 
        M.constraint("Slope Integrity", - Expr.mulElm(delta_y[1:n] , x[0:n-2]) + 
                    Expr.mulElm((delta_y[0:n-2] + delta_y[1:n]) , x[1:n-1]) - 
                    Expr.mulElm(delta_y[0:n-2] , x[2:n]) <= 0)

        # The area below our density function, g, must sum up to 1
        M.constraint("Probability Sum",  Expr.dot(((x[0:n-1] + x[1:n]) / 2 ) , delta_y)  == 1.0)

        # Power Cone Transformation Definition
        M.constraint(Var.vstack(x, u), Domain.inPPowerCone([1/n] * n))

        M.objective("Objective Function", ObjectiveSense.Maximize, u)

        M.solve()

        sol = x.level()
        ModelStatus = M.getPrimalSolutionStatus()
        print(ModelStatus)
        end = time.time()

        M.writeTask("MLE_PowerCone.ptf")
        iterations = M.getSolverIntInfo("intpntIter")
        print(f"Number of interior-point iterations: {iterations}")

        print("Elapsed time: ", round(end - start,4) ," s.")
        return sol
    return (PowerCone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The retreived result of the model is reported.
    """)
    return


@app.cell
def _(PowerCone, delta_y):
    sol_PC = PowerCone(delta_y)
    return (sol_PC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once more, as seen in the figure, the estimator outputs a curve resembling the exponential distribution.
    """)
    return


@app.cell
def _(PlotFigure, sol_PC, y):
    PlotFigure(y,sol_PC)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The performance of two methods is evaluated using different sample sizes. Each sample is solved 100 times with both methods, and the total elapsed time and interior-point iterations are recorded in the table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Formatted Table</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%; /* Make it fit the container */
                max-width: 900px; /* Set a reasonable max width */
                margin: auto; /* Center the table */
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: center; /* Ensure text is properly aligned */
            }
            th {
                background-color: #f2f2f2;
            }
            .table-container {
                display: flex;
                justify-content: center;
                width: 100%;
            }
        </style>
    </head>
    <body>

    <div class="table-container">
        <table>
                <tr>
            <th rowspan="2" style="text-align: center;">Number of Samples</th>
            <th colspan="2" style="text-align: center;">Exponential Cone</th>
            <th colspan="2" style="text-align: center;">Power Cone</th>
                </tr>
            <tr>
                <th>Elapsed Time</th>
                <th>Interior-Point Iterations</th>
                <th>Elapsed Time</th>
                <th>Interior-Point Iterations</th>
            </tr>
            <tr>
                <td>10</td>
                <td>0.51</td>
                <td>900</td>
                <td>0.55</td>
                <td>1000</td>
            </tr>
            <tr>
                <td>100</td>
                <td>1.60</td>
                <td>2600</td>
                <td>2.26</td>
                <td>1800</td>
            </tr>
            <tr>
                <td>500</td>
                <td>8.75</td>
                <td>4480</td>
                <td>11.71</td>
                <td>2300</td>
            </tr>
            <tr>
                <td>1000</td>
                <td>16.89</td>
                <td>5600</td>
                <td>30.77</td>
                <td>3100</td>
            </tr>
            <tr>
                <td>2000</td>
                <td>109.04</td>
                <td>10600</td>
                <td>73.32</td>
                <td>4000</td>
            </tr>
            <tr>
                <td>10000</td>
                <td>408.44</td>
                <td>11600</td>
                <td>241.27</td>
                <td>4200</td>
            </tr>
        </table>
    </div>

    </body>
    </html>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The power cone demonstrates significantly better performance in terms of total interior-point iterations. While the number of iterations increases for both methods as the sample size grows, the power cone shows a slower rate of increase compared to the exponential cone.

    Due to the way the power cone is currently implemented in Mosek, the time per interior-point iteration in the power case is higher than in the exponential cone case. It is expected that in a future version of Mosek, the time per iteration will be almost identical for both formulations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering Scheme
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the scale of the problem increases, it becomes harder to solve. To handle this situation, a clustering scheme is proposed. To determine a good enough estimator for a distribution, points that are too close to each other can be handled as one. While the points are merged by taking their averages, we add their weight to the objective function to reflect the density of these merged points. As a result, we end up using less points in our model. Therefore, the resulting clustered y points get resorted, and the value of n is updated.

    The clusters are built in a manner to ensure that the difference between the adjacent points does not exceed the resolution. The resolution can be tuned according to the sample size and the distribution function.
    """)
    return


@app.cell
def _(np):
    def ClusterPoints(y):
        r = 2e-4 #resolution
        clustered_y = []
        new_cluster = []
        w = []
        for z in y:
            if not new_cluster:
                new_cluster.append(z)
            else:
                if z - new_cluster[-1] <= r:
                    new_cluster.append(z)
                else:
                    w.append(len(new_cluster))
                    clustered_y.append(np.mean(new_cluster))
                    new_cluster = [z]

        # As parameter of the model, we need the delta values array where delta_y = y[i] - y[i-1]
        clustered_y.sort()
        delta_y = np.diff(clustered_y)
        n = len(clustered_y)
        print("New sample set length went down from ",len(y), " to ", n, ".")
        return clustered_y, delta_y, w, n
    return (ClusterPoints,)


@app.cell
def _(ClusterPoints, y):
    clustered_y, clustered_delta_y, w, clustered_n = ClusterPoints(y)
    return clustered_delta_y, clustered_n, clustered_y, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We solve the model once more with the new points. Please note that the weight array is not a dummy array, an array of ones as it was in previous implemention, and is actually applied to the objective function.
    """)
    return


@app.cell
def _(ExponentialCone, clustered_delta_y, clustered_n, w):
    sol_clustered = ExponentialCone(clustered_delta_y, clustered_n, w)
    return (sol_clustered,)


@app.cell
def _(PlotFigure, clustered_y, sol_clustered):
    PlotFigure(clustered_y, sol_clustered)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a more meaningful implementation, please increase the number of points to a significantly big number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. The **MOSEK** logo and name are trademarks of <a href="http://mosek.com">Mosek ApS</a>. The code is provided as-is. Compatibility with future release of **MOSEK** or the `Fusion API` are not guaranteed. For more information contact our [support](mailto:support@mosek.com).
    """)
    return


if __name__ == "__main__":
    app.run()