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
    # Irreducible Infeasible Set (IIS)

    When a large optimization problem is infeasible, it is very common that infeasibility can be localized to a small subset of mutually contradicting constraints. Locating such a small set can help better understand and, if needed, correct the infeasibility issue.

    An *Irreducible Infeasible Subset* (IIS) is, by definition, an infeasible set of constraints such that all its proper subset are feasible. It is therefore a minimal witness of infeasibility. Every infeasible problem has one or more IIS, which may be disjoint or overlapping and can be of different sizes.

    In this notebook we implement a standard method of locating an IIS for linear programs (LPs), namely a *Deletion Filter*. (see [Feasibility and Infeasibility in Optimization](https://link.springer.com/book/10.1007/978-0-387-74932-7)). This algorithm is as follows

    - Initialize $S$ to be the set of all constraints.
    - Iterate over all constraints and temporarily remove each from $S$:
        - If the problem is still infeasible, then permanently remove this constraint from $S$.
        - If the problem becomes feasible, then put the constraint back to $S$.
    - After iterating through all constraints return $S$ as the IIS.

    As we can see the algorithm will solve as many intermediate problems as there are constraints. If the IIS is small, as it often happens, then the intermediate problems will quickly shrink in size.

    There are many implementation details which we discuss later, together with the code.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Implementation with MOSEK ##

    There are many MOSEK-specific implementation details which we now outline.

    - We work with the Optimizer API for Python and objects of class ``mosek.Task``. 
    - Instead of creating a new task for each iteration of the algorithm we re-use the same task object. 
    - Instead of removing constraints from $S$ we make them unbounded by setting their ``boundkey`` to be open in the direction we are testing thus *effectively removing* the bound. When a constraint is supposed to return to $S$ we just restore the original bound.
    - We initialize $S$ with all the effective bounds in the problem: variable upper/lower bounds and constraint upper/lower bounds. A fixed or ranged bound is treated as two bounds (upper and lower) so each of them can participate in the IIS independently. 
    - We use the simplex solver to allow efficient hot-starts since we will be solving many similar linear problems sequentially.
    - We allow infeasible linear problems and mixed-integer linear problems whose root relaxation is infeasible.
    """
    )
    return


@app.cell
def _():
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
    from mosek import Task, iparam, optimizertype, miomode, soltype, prosta, boundkey

    import sys, random

    # Functions to inspect and transform bounds
    hasup = lambda bk: bk in [boundkey.up, boundkey.ra, boundkey.fx]
    haslo = lambda bk: bk in [boundkey.lo, boundkey.ra, boundkey.fx]
    relaxup = lambda bk: boundkey.fr if bk == boundkey.up else boundkey.lo
    relaxlo = lambda bk: boundkey.fr if bk == boundkey.lo else boundkey.up

    # Prepare a task for the deletion filter routine
    def prepareTask(task):
        # Use the simplex algporithm to exploit hot-start
        task.putintparam(iparam.optimizer, optimizertype.free_simplex)
        # Allow mixed-integer models by working with their continuous relaxation instead
        task.putintparam(iparam.mio_mode, miomode.ignored)
        # Remove objective
        task.putclist(range(task.getnumvar()), [0.0]*task.getnumvar())

    # Check if a task is feasible
    def feasibilityStatus(task):
        task.optimize()
        psta = task.getprosta(soltype.bas)
        if psta in [prosta.prim_and_dual_feas]:
            return True
        elif psta in [prosta.prim_infeas]:
            return False
        else:
            return None # Could be numerical issues

    # Runs the DeletionFilter on a task, with prescribed ordering of constraints and variables
    # Return: a pair (completed successfully ?, IIS)
    def deletionFilter(task, order):
        # We first assume that the IIS consists of everything
        iis = list(order)

        for (idx, what, bound) in order:
            getbound = task.getconbound if what == 'c' else task.getvarbound
            putbound = task.putconbound if what == 'c' else task.putvarbound
            relaxbound = relaxup if bound == 'u' else relaxlo

            # Inspect the element of the task with index idx, either variable or constraint
            bk, bl, bu = getbound(idx)

            # Remove the bound completely (make it unbounded) and see if the task becomes feasible
            putbound(idx, relaxbound(bk), bl, bu)
            feas = feasibilityStatus(task)

            if feas == True:
                # Restore the constraint/variable back to its bounds and continue trying the next one
                putbound(idx, bk, bl, bu)
            elif feas  == False:
                # Task is still infeasible, this constraint/variable will be ignored (leave it unbounded)
                iis.remove((idx, what, bound))
            else:
                # None - there were numerical issues, give up and return the current list as IIS
                return False, iis

        return True, iis
    return (
        Task,
        deletionFilter,
        feasibilityStatus,
        haslo,
        hasup,
        prepareTask,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## More implementation details ##

    Here are some general considerations when implementing the Deletion Filter algorithm, not specifically related to MOSEK:

    - The order in which constraints are tested in the loop affects the resulting IIS. Here we always use a random order, but one could fix specific ordering to put emphasis on various features of the IIS (form example prefer variable bounds to constraint bounds etc.)
    - The intermediate problems can become borderline feasible/infeasible, naturally leading to numerical issues. If that happens we just terminate and return the current $S$.
    """
    )
    return


@app.cell
def _(deletionFilter, feasibilityStatus, haslo, hasup, prepareTask, random):
    # Computes IIS for a problem
    def computeIIS(task, method='random'):
        # Initially solve the problem
        prepareTask(task)
        if feasibilityStatus(task) != False:
            print("The task is not infeasible, nothing to do")
            return True, []

        # Find all essential (not free) bounds in the problem
        # Format: (index, constraint or variable ? , lower or upper bound ? )
        allItems = [(i, 'c', 'u') for i in range(task.getnumcon()) if hasup(task.getconbound(i)[0])] + \
                   [(i, 'c', 'l') for i in range(task.getnumcon()) if haslo(task.getconbound(i)[0])] + \
                   [(j, 'v', 'u') for j in range(task.getnumvar()) if hasup(task.getvarbound(j)[0])] + \
                   [(j, 'v', 'l') for j in range(task.getnumvar()) if haslo(task.getvarbound(j)[0])]               

        if method == 'random':
            random.shuffle(allItems)

        return deletionFilter(task, allItems)
    return (computeIIS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Wrapping up and an example ##

    Finally we put together all the ingredients and find an IIS for an example from https://docs.mosek.com/latest/pythonapi/debugging-infeas.html
    """
    )
    return


@app.cell
def _(Task, computeIIS):
    # Print a text representation of the IIS
    def printIIS(task, iis):
        sgn = lambda x: '-' if x < 0.0 else '+'
        varname = lambda t, j: t.getvarname(j) if t.getvarnamelen(j) > 0 else f"x[{j}]"
        conname = lambda t, i: f"{t.getconname(i)}: " if t.getconnamelen(i) > 0 else ""
        btoineq = lambda b, bl, bu: f" <= {bu}" if bound == 'u' else f" >= {bl}"
        for (idx, what, bound) in iis:
            if what == 'v':
                bk, bl, bu = task.getvarbound(idx)
                print(f"+ {varname(task,idx)}{btoineq(bound, bl, bu)}") 
            else:
                bk, bl, bu = task.getconbound(idx)
                nz, sub, val = task.getarow(idx)
                expr = ' '.join(f"{sgn(v)} {abs(v)} {varname(task,j)}" for (j,v) in zip(list(sub), list(val)))
                print(f"{conname(task, idx)}{expr}{btoineq(bound, bl, bu)}") 

    def IISFromPtf(ptftask):
        with Task() as task:
            task.readptfstring(ptftask)
            success, iis = computeIIS(task)    

            if success:
                print(f"IIS computation completed successfully, size = {len(iis)}")
                printIIS(task, iis)
            else:
                print(f"IIS computation interrupted prematurely because of numerical issues, size = {len(iis)}")

    # Example:
    IISFromPtf("""Task
    Objective obj
        Minimize + x11 + 2 x12 + 5 x23 + 2 x24 + x31 + 2 x33 + x34
    Constraints
        s0 [-inf;200] + x11 + x12
        s1 [-inf;1000] + x23 + x24
        s2 [-inf;1000] + x31 + x33 + x34
        d0 [1100] + x11 + x31
        d1 [200] + x12
        d2 [500] + x23 + x33
        d3 [500] + x24 + x34
    Variables
        x11 [0;+inf]
        x12 [0;+inf]
        x23 [0;+inf]
        x24 [0;+inf]
        x31 [0;+inf]
        x33 [0;+inf]
        x34 [0;+inf]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The example here represents a supply-demand network. Infeasibility is caused by the fact that stores 1 and 2 have higher joint demand than the plants 1 and 3 can supply. The IIS found reflects this situation. (Note that you can get various IIS when running this example; the smallest and also most straightforward one has size 6).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![MOSEK ApS](https://docs.mosek.com/latest/pythonapi/_images/transportp.png )""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Extensions and improvements ##

    The source code of this Deletion Filter implementation can also be downloaded from the accompanying file [iis_deletion.py](iis_deletion.py). It is a very simple, basic algorithm which we provide as a proof-of-concept example. Here we outline possible extensions one could try:

    - Run the deletion filter a few times, possibly in a multithreaded fashion, and take the smallest IIS found.
    - Start not from the full set of constraints, but only from those which appear in the Farkas infeasibility certificate (equivalently are found by the ``task.getinfeasiblesubproblem()`` method.) In fact in many practical cases (especially due to modeling/coding errors and when a certificate is found in presolve) the certificate is already an IIS. Note, however, that this restricts the possible IIS the algorithm can find to those contained in a particular certificate.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. The **MOSEK** logo and name are trademarks of <a href="http://mosek.com">Mosek ApS</a>. The code is provided as-is. Compatibility with future release of **MOSEK** or the `MOSEK Optimizer API for Python` are not guaranteed. For more information contact our [support](mailto:support@mosek.com).""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
