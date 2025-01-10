import marimo

__generated_with = "0.9.33"
app = marimo.App(width="medium")


@app.cell
def __():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense
    import mosek.fusion.pythonic
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import sys
    np.random.seed(5)
    return (
        Domain,
        Expr,
        Model,
        ObjectiveSense,
        mo,
        mosek,
        np,
        plt,
        sys,
        time,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        Hierarchical model structure allows users to consider multiple objectives, and the trade-offs in their models with a simple approach. 

        In health-care scheduling it is not unsual to see objectives with more than one focus points. And in this example, we are showing a hierarchical model structure where the decisions are made minimizing the overall cost while increasing the overall affinity. 

        Let's say we want to assign a set of healthcare workers, $w \in W$, to a set of patients, $p \in P$. For simplicity, the set sizes of both groups are the same, and the assignments of healthcare workers to patients are done one-to-one. For each worker assigned to a particular patient, there is a cost of assignment, $c_{wp}$. And the proximity of workers to patients is measured with the affinity parameter, $a_{wp}$. 

        Firstly, we prioritize to minimize the costs, and do not consider the affinity measure. Then, we have a basic assignment problem on hand. Which can be modeled as follows: 

        $$
        min \quad z = \sum_{w \in W} \sum_{p \in P} c_{wp} x_{wp}  
        $$


        $$s.t. \sum_{w \in W} x_{wp} = 1 \quad \quad p \in P \quad\quad\quad (1)$$


        $$
        \sum_{p \in P} x_{wp} = 1 \quad\quad w\in W \quad\quad\quad (2)
        $$

        $$
        x_{wp} \in \{0, 1\} \quad w \in W, \, p \in P
        $$
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""Let's define the model using MOSEK Fusion API.""")
    return


@app.cell
def __(Domain, Expr, Model, ObjectiveSense, cost, ones):
    def minCostModel(N):
        #Create the model, we call it M here, by calling the Model class
        M = Model()

        #The decision variable, x, is created using the Model.variable() function. 
        #Model.variable(variable_name, dimensions, variable type)
        #As defined in the model, our W and P sets are equal sized and 2 dimensional. 
        x = M.variable("x", [N,N], Domain.binary())

        #Model.constraint(constraint_name, expression)
        #The expression is constructed with the dot product operator "@", to ensure a more efficient implementation.

        #In constraint (1), the dot product will result in (sum of x[w,p] * ones[w,p] for each w), for each p. 
        M.constraint("Contraint (1)", x.T @ ones == 1) #column sum
        #In constraint (2), the dot product will result in (sum of x[w,p] * ones[w,p] for each p), for each w. 
        M.constraint("Contraint (2)", x @ ones.T == 1) #row sum

        #Model.objective(objective_function_name, objective,_type, expression)
        #Expr.dot(a,b) conducts a element-wise matrix multiplication. Then sums the cells values and outputs a scalar value.
        #Expr.dot(x,cost) : sum of x[w,p] * cost[w,p] for each w,p
        M.objective("Minimum Cost Objective Function", ObjectiveSense.Minimize, Expr.dot(x, cost))

        #solve the model
        M.solve()

        #retrieve the objective value
        objective_value_init = M.primalObjValue()

        M.writeTask("minCostModel.ptf")
        print("Objective Value:", round(objective_value_init,4))
        return objective_value_init
    return (minCostModel,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""The initial model can now be implemented. Firstly, select the number of desired workers and patients.""")
    return


@app.cell(hide_code=True)
def __(mo):
    NodeSize = mo.ui.number(10)
    mo.md(f"Choose the number of workers/patients: {NodeSize})")
    return (NodeSize,)


@app.cell(hide_code=True)
def __(NodeSize):
    N = NodeSize.value
    return (N,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Then, let's generate a data to use in our model. We randomly generate the cost matrix using a uniform distribution.""")
    return


@app.cell
def __(N, np):
    cost = np.random.uniform(low=10, high=50, size=(N, N))
    ones = np.ones(N)
    return cost, ones


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Now, we can run the model.""")
    return


@app.cell
def __(N, minCostModel):
    initialObjective = minCostModel(N)
    return (initialObjective,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        With this solution, we have managed to obtain a minimum cost assignment. Now, we also want to consider the affinity of the assignments and maximize it. With hierarchical optimization, this approach can be implemented in a simple manner. 

        Let's call the minimum cost assignment value $z^*$. We can change the objective function to maximize the affinity while constraining the model with a cost upper bound. This approach will allow us to maximizing the total affinity while still maintaining the minimal cost. 

        The objective function maximizes the total affinity, while constraint (1) limits the total cost to be at most the retrieved minimal value. The rest of the constraints remain the same.

        $$ max \quad w = \sum_{wp} a_{wp} x_{wp} $$

        $$ \quad \sum_{wp} c_{wp} x_{wp} \leq z^*  \quad\quad (1)$$

        $$ \sum_{w} x_{wp} = 1, \quad  p \in P \quad\quad (2) $$

        $$ \sum_{p} x_{wp} = 1, \quad w \in W  \quad\quad (3) $$

        $$ x_{wp} \in \{0, 1\}, \quad w \in W, p \in P $$

        One can also decide to trade off from the cost and prioritize the affinity more. To achieve this we can increase the cost by a fraction and resulting in a more relaxed constraint. Let's call this fraction, $a$. Then constraint (1), transforms into the following:

        $$ \quad \sum_{wp} c_{wp} x_{wp} \leq (1+a)z^*  \quad\quad (1)$$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Let's implement the Maximum Affinity - Minimum Cost Model according to this definition.""")
    return


@app.cell
def __(Domain, Expr, Model, ObjectiveSense, affinity, cost, ones):
    def maxAffinityMinCostModel(a, objective_value_init, N):
        #Model definition, called m
        M = Model()

        #Binary variable x[w,p] is defined where w,p in N
        x = M.variable("x", [N,N], Domain.binary())

        #In constraint (1), the dot function will execute (sum of (x[w,p] * cost[w,p]) for each w,p). 
        #RHS is relaxed by "a" percent. 
        M.constraint("Constraint (1)",  Expr.dot(x, cost) <=  (1+a)*objective_value_init)

        #In constraint (2), the dot product will result in (sum of x[w,p] * ones[w,p] for each w), for each p. 
        M.constraint("Constraint (2)",  x.T @ ones == 1) #column sum

        #In constraint (3), the dot product will result in (sum of x[w,p] * ones[w,p] for each p), for each w. 
        M.constraint("Constraint (3)",  x @ ones.T == 1) #row sum

        #Maximize the total affinifity, sum of x[w,p] * affinity[w,p] for each w,p
        M.objective("Maximum Affinity Objective Function", ObjectiveSense.Maximize, Expr.dot(x, affinity))
        M.solve()

        #retrieve the objective value
        objective_value = M.primalObjValue()
        # print("Objective Value:", objective_value)
        return M,objective_value
    return (maxAffinityMinCostModel,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Let's generate a matrix for affinity data as well. We also create a list of $a$ values and make multiple trials with the values to observe the trade off between affinty and cost. The $a$ values range from 0 to maximum 1 incrementing by 0.05.""")
    return


@app.cell(hide_code=True)
def __(mo):
    slider = mo.ui.slider(0.05, 1,step=0.05,value=0.5, show_value=True)
    mo.md(f"Choose the maximum alpha value for the trial range: {slider}")
    return (slider,)


@app.cell(hide_code=True)
def __(slider):
    alpha_range_input = slider.value
    alphaRange = (alpha_range_input/0.05) + 1
    return alphaRange, alpha_range_input


@app.cell
def __(N, alphaRange, np):
    affinity = np.random.uniform(low=10, high=50, size=(N, N))
    #The alpha values range from 0 to 1 incrementing by 0.1
    alphas = [round(i * 0.05, 2) for i in range(0, int(alphaRange))]
    return affinity, alphas


@app.cell
def __(alphas, maxAffinityMinCostModel, plt, time):
    def RunMaximumAffinityModel(N,initialObjective):
        #Record the retrieved affinities with the corresponding cost value
        costs = []
        affinities = []
        start = time.time()
        #Solve the model for every "a" value.
        for alpha in alphas:
            #Get the maximum affinity objective value
            M, maxAff_objective_value = maxAffinityMinCostModel(alpha,initialObjective,N)
            print("Objective value:", round(maxAff_objective_value,4), "   Cost:", round((1+alpha)*initialObjective,4))
            affinities.append(maxAff_objective_value)
            costs.append((1+alpha)*initialObjective)

        end = time.time()
        #Plot the Cost/Affinity Scatter Plot
        plt.scatter(costs, affinities)
        plt.xlabel("Cost")
        plt.ylabel("Affinity")
        plt.show()

        return (end-start)
    return (RunMaximumAffinityModel,)


@app.cell
def __(N, RunMaximumAffinityModel, initialObjective):
    runTime = RunMaximumAffinityModel(N,initialObjective)
    return (runTime,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        This model cuırrently is solved in seconds, but recalculating and resolving it from scratch every time the cost upper bound changes becomes slower and less efficient as the number of workers and patients increases. Instead of starting the solution process anew each time, we can parametrize the model and use a previously found initial solution, which will significantly reduce the solution time for larger instances.

        In parametrized models, the MOSEK Fusion Solver checks if the given solution is valid for the updated parameters. If so, it continues searching for the optimal value starting from the initial solution. Since we are only modifying the right-hand side of constraint (1), it is more logical to increase the $a$ values. By doing this, we relax the right-hand side and ensure the used initial solution value in other models (solution with the lowest $a$ parameter) being feasible with other parameters as well.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""To make our model parametrized, we need to define a parameter variable. With this parameter update, if it is still applicable, the previous model solution is used.""")
    return


@app.cell
def __(Domain, Expr, Model, ObjectiveSense, affinity, cost, np):
    def maxAffinityMinCostModelParametrized(a,objective_value_init,N):
        ones = np.ones(N)
        #Model definition called m
        M = Model()

        #Binary variable x[w,p] is defined where w,p in N
        x = M.variable("x", [N,N], Domain.binary())

        #Parameter to adjust the cost
        #Model.parameter(parameter_name, dimension)
        a = M.parameter("a", 1)

        #In constraint 1, the dot function will execute (sum of (x[w,p] * cost[w,p]) for each w,p). 
        #RHS is relaxed by "a" percent. 
        M.constraint("Constraint (1)",  Expr.dot(x, cost) <=  (1+a)*objective_value_init)

        #In constraint (2), the dot product will result in each (sum of x[w,p] * ones[w,p] for each w), for each p. 
        M.constraint("Constraint (2)",  x.T @ ones == 1) #column sum

        #In constraint (3), the dot product will result in each (sum of x[w,p] * ones[w,p] for each p), for each w. 
        M.constraint("Constraint (3)",  x @ ones.T == 1) #row sum

        #Maximize the total affinifity, sum of x[w,p] * affinity[w,p] for each w,p
        M.objective("Maximum Affinity Objective Function", ObjectiveSense.Maximize, Expr.dot(x, affinity))
        M.solve()

        return M
    return (maxAffinityMinCostModelParametrized,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Now, our model is parametrized. Let's run the model with the same $a$ values again. The objective results should be the same.""")
    return


@app.cell
def __(alphas, maxAffinityMinCostModelParametrized, plt, time):
    def RunMaximumAffinityParametrized(N,initialObjective):
        affinities = []
        costs = []

        start = time.time()
        #Run the model once, get the model object
        M = maxAffinityMinCostModelParametrized(0,initialObjective,N)
        #Set the parameter
        a_parameter = M.getParameter("a")

        #Solve the model for every "a" value.
        for alpha in alphas:
            #Set parameter value according to chosen "a"
            a_parameter.setValue(alpha)
            #Reoptimize by calling the solve function
            M.solve()
            #Get the objective value
            objective_value = M.primalObjValue()

            print("Objective value:", round(objective_value,4), "   Cost:", round((1+alpha)*initialObjective,4))
            affinities.append(objective_value)
            costs.append((1+alpha)*initialObjective)
        end = time.time()
        #Plot the Cost/Affinity Scatter Plot
        plt.scatter(costs, affinities)
        plt.xlabel("Cost")
        plt.ylabel("Affinity")
        plt.show()

        return (end-start)
    return (RunMaximumAffinityParametrized,)


@app.cell
def __(N, RunMaximumAffinityParametrized, initialObjective):
    runtime_Parametrized = RunMaximumAffinityParametrized(N,initialObjective)
    return (runtime_Parametrized,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        Let's test the runtime difference. You can change all the model outputs by using the interactive elements. Go to the top of the page and increase the worker/patient size. <br>
        _Hint: Try using 200 worker/patients!_
        """
    )
    return


@app.cell
def __(runTime, runtime_Parametrized):
    print("The runtime of non-parametrized Maximum Affinity Model: ", runTime, "s.")
    print("The runtime of parametrized Maximum Affinity Model: ", runtime_Parametrized,"s.")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""You can also observe the trade-off relation between cost and affinity by changing the alpha range from the slider!""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""This work is inspired by Ryan O'Neill's blog post about the implementation of hierarchical optimization. Click [here](https://ryanjoneil.github.io/posts/2024-11-08-hierarchical-optimization-with-gurobi/) to check out the blog post.""")
    return


if __name__ == "__main__":
    app.run()