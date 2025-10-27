import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Utility based option pricing with transaction costs and diversification""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this notebook, we present the <i>Fusion</i> implementation of the utility based option pricing model, presented by [Andersen et. al. (1999)](https://www.sciencedirect.com/science/article/pii/S0168927498001044). The purpose of the model is to estimate the reservation purchase price of a European call option, written on a risky security when there is proportional transaction costs in the market.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Consider an economy evolving over T periods, ${0,t_1,t_2,t_3,...,t_T = \overline{T}}$ where $\overline{T}$ is the time horizon in years. We consider that the investor has one risk-free security, such as a bond, that pays at a constant interest rate of $r\geq 0$, and $m$ risky securities with price processes denoted by $(S_1,S_2,...,S_m)$. The price of the risky securities evolves in an event tree such that: 

    $$(S_{1,n},S_{2,n},..,S_{m,n}) = \begin{cases}
    (a_1 S_{1,n-1}, a_2 S_{2,n-1},..,a_m S_{m,n-1}), \,\, \text{with probability } q_1, \\
    (b_1 S_{1,n-1}, b_2 S_{2,n-1},..,b_m S_{m,n-1}), \,\, \text{with probability } q_2, \\
    (c_1 S_{1,n-1}, c_2 S_{2,n-1},..,c_m S_{m,n-1}), \,\, \text{with probability } q_3 = (1-q_1-q_2)
    \end{cases}
    $$


    where the three possibilities lead to an event tree of splitting index 3 (In the fusion model presented below, we have considered a general case where one can have a different splitting index). The event tree for two risky securities can be visualized as shown in the figure below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let, $\mathbf{\alpha_{n}} = (\alpha_1,\alpha_2,...,\alpha_m)$ denote the number of units of the risky security held by the investor at time $t_n$, and $\beta_n$ denote the dollar amount held by the investor in bonds at the same time. Then, the <i>'budget constraint'</i> on the bonds an investor can buy in the next period is given by:


    $$\beta_n(I_n) \leq (1+r)\beta_{n-1}(I_n) + \sum_{i=1}^{m} S_{i,n}(I_n)[\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n) - \theta_i |\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n)|]$$


    where $\theta_i$ is the transaction cost for trading of risky security $i$ and $I_n$ denotes the path being considered (in other words, the sequence of events in the event tree). The total number of paths possible for a tree of splitting index $s$, over $n$ time periods is $s^n$. Initially, if we consider that the wealth of the investor is $W_0$, then of course the first budget constraint becomes:

    $$\beta_0 \leq W_0 - \sum_{i=1}^{m} S_{i,0}[\alpha_{i,0} + |\alpha_{i,0}|]$$


    Additionally, in the final period the investor will sell all of the risky securities, thus ending up with a wealth $W_T(I_T)$ for path $I_T$, such that:

    $$W_T(I_T) \leq (1+r)\beta_{T-1}(I_T) + \sum_{i=1}^{m} S_{i,T}(I_T)[\alpha_{i,T-1}(I_T) - \theta_i |\alpha_{i,T-1}(I_T)|]$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.) Portfolio choice""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The goal of the investor is to choose a portfolio strategy (i.e. the sequence {${\mathbf{\alpha_n(I_n)},\beta_n(I_n)}$}) that maximizes the expected utility. The expected utility is given by:

    $$E[U(W_T)] = \sum_{I_{T}\in F_T} q_1^{A(I_T)}q_2^{B(I_T)}(1-q_1-q_2)^{T-A(I_T)-B(I_T)}U(W_T(I_T))$$


    Note that the summation is over all the possible paths for a tree of a given split index and T time-periods (the set denoted by $F_T$). $A(I_T)$ and $B(I_T)$ denote the number of times the first and second possibilities are considered in every path, respectively. Following the equations presented above, the complete optimization problem can be written as:


    $$U^*(W_0) = \text{maximize}_{({\mathbf{\alpha}_n(I_n),\beta_n(I_n))}_{I_n\in F_T,{n=1,2,..,T-1}}}\,\,\,E[U(W_T)]$$


    $$\text{subject to:  } \beta_0 \leq W_0 - \sum_{i=1}^{m} S_{i,0}[\alpha_{i,0} + |\alpha_{i,0}|],$$

    $$\beta_n(I_n) \leq (1+r)\beta_{n-1}(I_n) + \sum_{i=1}^{m} S_{i,n}(I_n)[\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n) - \theta_i |\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n)|],$$

    $$W_T(I_T) \leq (1+r)\beta_{T-1}(I_T) + \sum_{i=1}^{m} S_{i,T}(I_T)[\alpha_{i,T-1}(I_T) - \theta_i |\alpha_{i,T-1}(I_T)|],\,\,\, \forall I_T\in F_T$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.) Price vector process""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The continuous price process that we intended to approximate is:

    $$\frac{d\tilde{S_i}}{\tilde{S_i}} = \mu_idt + \sum_{j=1}^{m}\sigma_{ij} dW_j \,\, , \,\, \forall i = 1,2,...,m$$

    where $\mu_i$ and $\sigma_{ij}$ are positive constants and the $W_j$ denote un-correlated standard Wiener processes. To construct a discrete approximation, we first define a stochastic vector as follows:

    $$(\epsilon_1,\epsilon_2,..,\epsilon_m) = \begin{cases}
    (\epsilon_1 (\omega_1),\epsilon_2 (\omega_1),..,\epsilon_m (\omega_1)), \,\, \text{with probability } q_1, \\
    (\epsilon_1 (\omega_2),\epsilon_2 (\omega_2),..,\epsilon_m (\omega_2)), \,\, \text{with probability } q_2, \\
    (\epsilon_1 (\omega_3),\epsilon_2 (\omega_3),..,\epsilon_m (\omega_3)), \,\, \text{with probability } q_3 = (1-q_1-q_2)
    \end{cases}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The equation describing the event tree (first equation of the notebook) becomes a discrete approximation of the above-stated stochastic process if we set:

    $$a_i = e^{\mu_i \Delta t}\bigg( \frac{e^{\big[\sum_{j=1}^{m}\sigma_{ij}\epsilon_j(\omega_1)\sqrt{\Delta t}\big]}}{z_i}\bigg)$$

    $$b_i = e^{\mu_i \Delta t}\bigg( \frac{e^{\big[\sum_{j=1}^{m}\sigma_{ij}\epsilon_j(\omega_2)\sqrt{\Delta t}\big]}}{z_i}\bigg)$$

    $$c_i = e^{\mu_i \Delta t}\bigg( \frac{e^{\big[\sum_{j=1}^{m}\sigma_{ij}\epsilon_j(\omega_3)\sqrt{\Delta t}\big]}}{z_i}\bigg)$$



    In the limit $T\to \infty$ (or alternatively $\Delta t \to 0$), the approximation approaches the continuous process.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.) Investor's reservation purchase price for a European call option""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Suppose that an investor is interested in buying a European call option on the security $S_1$, with time to maturity $\overline{T}$ and a strike price $K>0$. At the expiration time, the investor would get a payment of $\text{max}(S_{1,T} - K, 0)$ (assuming cash settlement and also that the investor will not be re-selling the option once it has been purchased.) Our goal is now to estimate the highest price this investor is willing to pay, to purchase such an option.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If the investor does not purchase the call option and only trades in the risky securities $S_i$ and the bonds, then the portfolio is given by the optimization problem stated above. However, if the investor buys the call option at a reservation purchase price of $C$, then their portfolio becomes the following:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$P(C,W_0) = \text{maximize}_{({\mathbf{\alpha}_n(I_n),\beta_n(I_n))}_{I_n\in F_T,{n=1,2,..,T-1}}}\,\,\,E[U(W_T)]$$


    $$\text{subject to:   }\beta_0 \leq W_0 - C - \sum_{i=1}^{m} S_{i,0}[\alpha_{i,0} + |\alpha_{i,0}|],$$

    $$\beta_n(I_n) \leq (1+r)\beta_{n-1}(I_n) + \sum_{i=1}^{m} S_{i,n}(I_n)[\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n) - \theta_i |\alpha_{i,n-1}(I_n) - \alpha_{i,n}(I_n)|],$$

    $$W_T(I_T) \leq (1+r)\beta_{T-1}(I_T) + \sum_{i=1}^{m} S_{i,T}(I_T)[\alpha_{i,T-1}(I_T) - \theta_i |\alpha_{i,T-1}(I_T)|] + \text{max}(S_{1,T}(I_T) - K,0), \,\,\, \forall I_T\in F_T$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The highest price that investor is willing to pay for the given call option, is therefore given by the price for which the maximum expected utility in the two portfolios becomes equal, making the investor indifferent to the choices. Thus, the final optimization problem that we need to consider is:

    $$C^* = \text{maximize}_{({\mathbf{\alpha}_n(I_n),\beta_n(I_n))}_{I_n\in F_T,{n=1,2,..,T-1}}} \,\,\, C$$

    $$\text{subject to: } E[U(W_T)] \geq U^*(W_0) $$

    along with the other constraints mentioned in the optimization problem for the portfolio in which the option is purchased.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.) Utility function""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the optimization problems mentioned above, we need to optimize the expected utility. The utility functions that we consider are members of a family called HARA utility functions (Hyperbolic Absolute Risk Aversion). The general expression for HARA utility is:

    $$U(W) = \frac{1-\gamma}{\gamma}\bigg(\frac{aW}{1-\gamma} + b\bigg)^\gamma \,;\, a>0$$

    where $W > ((\gamma - 1)b)/a$. Note that the exponential and logarithmic utility functions are also members of the HARA-class.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Another thing to consider is the Arrow-Pratt measure of the Absolute risk aversion. For the HARA-class, it is:

    $$ARA(W) = \bigg(\frac{W}{1-\gamma} + \frac{b}{a}\bigg)^{-1}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fusion Implementation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we proceed with the construction of the fusion model. We start by making a Tree class that will represent the event tree discussed above.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Parameters for Initializing the Tree Class

    When creating a Tree object, the following parameters define the investment environment and decision-making setup:

    - **M** – The optimization model, typically a MOSEK Fusion `Model`, that will hold variables, constraints, and the objective.  
    - **T** – The number of discrete time steps in the investment horizon.  
    - **W₀** – The investor’s initial wealth at time 0.  
    - **S** – The price-scaling matrix that adjusts asset prices over time.  
    - **θ** – The vector of transaction costs, representing costs incurred when trading risky assets.  
    - **Sᵥ₀** – The initial price vector for the risky securities at time 0.  
    - **r** – The risk-free interest rate applied to bonds.  
    - **U = [a, b, γ]** – The set of parameters defining the investor’s HARA (Hyperbolic Absolute Risk Aversion) utility function, capturing risk preferences.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### HARA utility as a Power cone:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The objective function is:

    $$\text{maximize   }E[U(W_T)] = \sum_{I_T\in F_T}P(I_T) \bigg(\frac{1-\gamma}{\gamma}\bigg)  \bigg(\frac{aW(I_T)}{1-\gamma} + b\bigg)^\gamma$$

    where $P(I_T)$ is the probability of a given path. We can re-write this as:


    $$\text{maximize   } \sum_{I_T\in F_T}P(I_T) \bigg(\frac{1-\gamma}{\gamma}\bigg) h(I_T)$$

    $$\text{subject to :   } h(I_T)\leq \bigg(\frac{aW(I_T)}{1-\gamma} + b\bigg)^\gamma$$


    Now, the constraint can be expressed as a Power cone. However, there are a few possible cases:

    1.) $\gamma > 1$: In this case, the conic representation is:

    $$\Bigg(h(I_T),1,\bigg(\frac{aW(I_T)}{1-\gamma} + b\bigg) \Bigg) \in \mathcal{P}_3^{1/\gamma}$$

    2.) $0 < \gamma < 1$:

    $$\Bigg(\bigg(\frac{aW(I_T)}{1-\gamma} + b\bigg),1, h(I_T) \Bigg) \in \mathcal{P}_3^{\gamma}$$

    3.) $\gamma < 0$:

    $$\Bigg(h(I_T),\bigg(\frac{aW(I_T)}{1-\gamma} + b\bigg),1  \Bigg) \in \mathcal{P}_3^{1/(1-\gamma)}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Exponential utility as an exponential cone:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The objective function in the case of exponential utility will be given by:

    $$\text{maximize   }E[U(W_T)] = \sum_{I_T\in F_T}P(I_T) \bigg(1 - e^{- [\text{ARA}(W)] W(I_T)}\bigg)
    $$

    if the Absolute Risk aversion is nonzero. However, if ARA$(W) = 0$, then the objective function is simply:

    $$\text{maximize   }E[U(W_T)] = \sum_{I_T\in F_T}P(I_T) W(I_T)$$

    For the non-zero ARA$(W)$, we express the objective function as follows:

    $$\text{maximize   } \sum_{I_T\in F_T}P(I_T) h(I_T)$$

    $$\text{subject to:  } (1-h(I_T))\geq e^{-[\text{ARA}(W)]W(I_T)}$$

    and the constraint is readily expressed as an exponential cone:

    $$\big((1-h(I_T)),1,-[\text{ARA}(W)]W(I_T)\big) \in K_{exp}$$
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
    import sys
    import matplotlib.pyplot as plt
    import time
    return Domain, Expr, Model, ObjectiveSense, np, sys


@app.cell
def _(Domain, Expr, Level, np):
    class Tree:    
        # Instantiate the Tree class.
        def __init__(self, M, T, W_0, S, theta, S_v_0, r, U):
            self.M = M
            self.W_0 = W_0
            self.T = T
            self.S_v = S
            self.r = r    
            # Shape of the cost scaling matrix will give us the split index and the number of risky securities.
            self.s, self.n = S.shape
            self.cost = np.asarray([theta])
            self.S_v_0 = S_v_0
            # Parameters for the utility function.
            self.U = U
            self.S_v_t = S_v_0
            # This is to decide which utility will be used.
            self.util_dispatch = {'HARA': self.HARA_util, 'EXP': self.exp_util}
            self.levels = []

        # Method to make all the levels, or the time steps in the tree.
        def level_make(self):
            # Number of risky securities, i.e. alpha{i,t=0}
            self.a0 = self.M.variable('a_0', [1, self.n])
            # Bonds at t=0.
            self.b0 = self.M.variable('b_0', 1)
            # Making the first budget constraint.
            self.budg_ex = Expr.sub(
                Expr.sub(self.W_0, self.b0),
                Expr.dot(self.S_v_0, Expr.mulElm(1 + self.cost, self.a0))
            )
            self.root_budget = self.M.constraint('Budget_0', self.budg_ex, Domain.greaterThan(0.0))
            # This list will hold the level objects associated to this tree.
            a, b = self.a0, self.b0
            for i in range(1, self.T + 1):
                # Appending level n, based on the level (n-1).
                self.levels.append(Level(i, self, a, b))
                a = self.levels[i - 1].a_new
                b = self.levels[i - 1].b_new
            # a_T is zero ().

        # Method to make the HARA utility function and the corresponding Power-cone constraint.
        def HARA_util(self, W):
            self.h = self.M.variable('h', self.s**self.T, Domain.greaterThan(0.0))

            # HARA utility is only defined for when W > (gamma - 1)a/b.
            self.M.constraint(W, Domain.greaterThan(self.U[1] * (self.U[2] - 1) / self.U[0]))

            I = Expr.constTerm([self.s**self.T], 1.0)
            self.E1 = Expr.add(
                Expr.mul(self.U[0] / (1 - self.U[2]), W),
                Expr.constTerm(W.getShape(), self.U[1])
            )

            # Different cases for different gamma values
            if self.U[2] > 1.0:
                self.M.constraint('Obj_terms_HARA', Expr.hstack(self.h, I, self.E1), Domain.inPPowerCone(1 / self.U[2]))
            elif self.U[2] < 0.0:
                self.M.constraint('Obj_terms_HARA', Expr.hstack(self.h, self.E1, I), Domain.inPPowerCone(1 / (1 - self.U[2])))
            else:
                self.M.constraint('Obj_terms_HARA', Expr.hstack(self.E1, I, self.h), Domain.inPPowerCone(self.U[2]))

            return Expr.mul(((1 - self.U[2]) / self.U[2]), self.h)

        # Method to make the Exponential utility constraints.
        def exp_util(self, W):
            # Zero ARA(W) case
            if self.U == 0:
                return W
            # Nonzero ARA(W) case
            self.h = self.M.variable('h', self.s**self.T)
            I = Expr.constTerm([self.s**self.T], 1.0)
            E_exp = Expr.mul(-self.U, W)
            self.M.constraint('Obj_terms_exp',
                              Expr.hstack(Expr.sub(I, Expr.mul(self.U, self.h)), I, E_exp),
                              Domain.inPExpCone())
            return self.h
    return (Tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The <i>Tree</i> class defined above has the sub-class <i>Level</i>. The budget constraints involved in our model connect variables that lie in consecutive levels. Therefore, we can visualize all the constraints between two levels in the following manner (Note: for ease of visualization, we take the split index as $s = 3$, and a time period of n = 3):

    $$
    \begin{pmatrix}
    a\begin{bmatrix}
    S_{(1,2)} \\
    S_{(2,2)} \\
    S_{(3,2)} \\
    \end{bmatrix}\\
    b\begin{bmatrix}
    S_{(1,2)} \\
    S_{(2,2)} \\
    S_{(3,2)} \\
    \end{bmatrix}\\
    c\begin{bmatrix}
    S_{(1,2)} \\
    S_{(2,2)} \\
    S_{(3,2)} \\
    \end{bmatrix}\\
    \end{pmatrix}
    :
    \begin{pmatrix}
    \begin{bmatrix}
    \mathbf{\alpha_{(1,2)}},\beta_{(1,2)} \\
    \mathbf{\alpha_{(2,2)}},\beta_{(2,2)} \\
    \mathbf{\alpha_{(3,2)}},\beta_{(3,2)} \\
    \end{bmatrix} \\
    \begin{bmatrix}
    \mathbf{\alpha_{(1,2)}},\beta_{(1,2)} \\
    \mathbf{\alpha_{(2,2)}},\beta_{(2,2)} \\
    \mathbf{\alpha_{(3,2)}},\beta_{(3,2)} \\
    \end{bmatrix} \\
    \begin{bmatrix}
    \mathbf{\alpha_{(1,2)}},\beta_{(1,2)} \\
    \mathbf{\alpha_{(2,2)}},\beta_{(2,2)} \\
    \mathbf{\alpha_{(3,2)}},\beta_{(3,2)} \\
    \end{bmatrix} \\
    \end{pmatrix}  \longrightarrow
    \begin{pmatrix}
    \mathbf{\alpha_{(1,3)}},\beta_{(1,3)} \\
    \mathbf{\alpha_{(2,3)}},\beta_{(2,3)} \\
    \mathbf{\alpha_{(3,3)}},\beta_{(3,3)} \\
    \mathbf{\alpha_{(4,3)}},\beta_{(4,3)} \\
    \mathbf{\alpha_{(5,3)}},\beta_{(5,3)} \\
    \mathbf{\alpha_{(6,3)}},\beta_{(6,3)} \\
    \mathbf{\alpha_{(7,3)}},\beta_{(7,3)} \\
    \mathbf{\alpha_{(8,3)}},\beta_{(8,3)} \\
    \mathbf{\alpha_{(9,3)}},\beta_{(9,3)} \\
    \end{pmatrix}
    $$

    Here, the first column vector is a <i>"vstack"</i> of the price vector at level $n=2$, multiplied by the scaling factors for the price. Therefore, the first column is the new price vector (for $n=3$). In the second column, $\alpha$'s will be vectors if there are multiple risky securities. Also, $\alpha_{(i,n)}$ and $\beta_{(i,n)}$ correspond to the number of risky securities and the bonds, respectively, for the $i^{\text{th}}$ path at the $n^{\text{th}}$ time period. It is to be realized that for a given time period, $1\leq i \leq s^n$.

    The above-stated representation shows that we can <i>"repeat"</i> the variables of the previous level $s$ number of times and then it becomes quite easy to implement the budget constraints in <i>Fusion</i>. One can also extend the above shown method to a higher split index.
    """
    )
    return


@app.cell
def _(Domain, Expr, np):
    # Each level corresponds to a time step in the event tree.
    class Level:
        # l corresponds to the time step; a_old, b_old belong to (l-1) in the tree. 
        def __init__(self, l, Tree, a_old, b_old):
            if l == Tree.T:
                # If the current level is the final time period, then all risky securities are considered sold.
                self.a_new = Expr.constTerm([Tree.s**l, Tree.n], 0.0)
                # Final step's bonds are the wealth W(I_T); later used in the utility function.
                self.b_new = Tree.M.variable('W_T', Tree.s**l)
            else:
                # Risky securities for time step l.
                self.a_new = Tree.M.variable(f'a_{l}', [Tree.s**l, Tree.n])
                # Bonds in dollars for time step l.
                self.b_new = Tree.M.variable(f'b_{l}', Tree.s**l)

            # Variable for the quadratic cone to implement the absolute value constraint.
            self.t_new = Tree.M.variable(f't_{l}', [Tree.s**l, Tree.n], Domain.greaterThan(0.0))
            Tree.cost = np.repeat(Tree.cost, Tree.s, axis=0)

            # Repeating/Stacking the (l-1) level variables for budget constraints. 
            A = Expr.repeat(a_old, Tree.s, 0)
            B = Expr.repeat(b_old, Tree.s, 0)

            # Price vector of previous level, scaled and stacked vertically.
            Tree.S_v_t = np.vstack([np.multiply(j, Tree.S_v_t) for j in Tree.S_v])

            # Expressions for budget constraints.
            self.bond_sub = Expr.sub(Expr.mul(B, (1 + Tree.r)), self.b_new)
            self.secu_sub = Expr.sub(self.a_new, A)
            self.transact = Expr.add(self.secu_sub, Expr.mulElm(Tree.cost, self.t_new))
            self.secu_exp = Expr.mulDiag(self.transact, Tree.S_v_t.transpose())

            # Linear budget constraint.
            self.budg_constr = Tree.M.constraint(f'Budget_{l}',
                                                 Expr.sub(self.bond_sub, self.secu_exp),
                                                 Domain.greaterThan(0.0))
            # Quadratic cone for absolute value requirement.
            Tree.M.constraint(f'a{l}_abs', Expr.stack(2, self.t_new, self.secu_sub), Domain.inQCone())
    return (Level,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Now that we have the basic structure of the tree ready, we can easily make the <i>Fusion</i> model as follows...""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will represent the paths $I_T$ by their numbers. Therefore, we have (split_index)$^T$ paths. Each Path number represented in the base of the split index will give us a unique id for each path.""")
    return


@app.cell
def _(np):
    def base_rep(b, i, size):
        n = np.zeros(size)
        for j in range(size):
            n[j] = i % b
            if i // b == 0:
                break
            else:
                i = i // b
        return np.flip(n)

    def path_id_make(split_index, T):
        path_id = []
        for i in range(split_index ** T):
            path_id.append(base_rep(split_index, i, T))
        return np.asarray(path_id).astype(int)

    def path_route_calc(path_id, split_index, q, *args):
        s = np.zeros(split_index)
        for _p in path_id:
            s[_p] = s[_p] + 1
        if args:
            path_S1 = []
            for j in range(split_index):
                path_S1.append(args[0][j] ** s[j])
            path_S1T = np.prod(np.asarray(path_S1))
            return path_S1T
        else:
            path_prob = np.prod(q ** s)
            return path_prob

    def price_vector_z(sigma, eps, dt, q):
        E = np.exp(np.matmul(eps, sigma) * np.sqrt(dt))
        return np.matmul(q, E)

    def price_vector_abc(sigma, eps, dt, q, mu, Z):
        E1 = np.multiply(np.exp(np.matmul(eps, sigma) * np.sqrt(dt)), 1 / Z)
        E2 = np.repeat([np.exp(mu * dt)], eps.shape[0], axis=0)
        return np.multiply(E2, E1)
    return path_id_make, path_route_calc, price_vector_abc, price_vector_z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Portfolio without call option:""")
    return


@app.cell
def _(Expr, Model, ObjectiveSense, Tree, path_id_make, path_route_calc, sys):
    def Portfolio(port_params, util_type='HARA', wrtLog=True):
        T, W_0, S, theta, S_v_0, r, U, q = port_params
        M = Model('PORTFOLIO')
        if wrtLog:
            M.setLogHandler(sys.stdout)
        Tree_1 = Tree(M, T, W_0, S, theta, S_v_0, r, U)
        Tree_1.level_make()
        H = Tree_1.util_dispatch[util_type](Tree_1.levels[T - 1].b_new)
        path_ids = path_id_make(Tree_1.s, T)
        path_probs = [path_route_calc(_p, Tree_1.s, q) for _p in path_ids]
        Obj = Expr.dot(H, path_probs)
        M.objective('PORTFOLIO_OBJ', ObjectiveSense.Maximize, Obj)
        M.solve()
        utility_W0 = M.primalObjValue()
        ut_time = M.getSolverDoubleInfo('optimizerTime')
        ut_iter = M.getSolverIntInfo('intpntIter')
        return (M, Tree_1, utility_W0, path_ids, Obj, [ut_iter, ut_time])
    return (Portfolio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Portfolio with call option:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This model involves modifying the initial and the final budget constraints, adding a constraint on the new utility, creating a new objective (the option price) and re-optimizing the previous model. Fusion allows re-optimizing (this saves a considerable amount of time, which would otherwise be spent in re-building the model).""")
    return


@app.cell
def _(Domain, Expr, ObjectiveSense, Portfolio, np, path_route_calc):
    def Option_Portfolio(port_params, K, util_type='HARA', writeLog=True, solver_info=False):
        T, W_0, S, theta, S_v_0, r, U, q = port_params
        M, Tree_1, utility_W0, path_ids, Obj, obj_info = Portfolio(port_params, util_type=util_type, wrtLog=writeLog)
        path_Svs = np.asarray([path_route_calc(_p, S.shape[0], q, S[:, 0]) for _p in path_ids])
        call_profit = (path_Svs - K + abs(path_Svs - K)) / 2
        Call = M.variable('Call_Price', 1, Domain.inRange(0.0, S_v_0[0]))
        Tree_1.root_budget.update(Expr.neg(Call), Call)
        Tree_1.levels[T - 1].budg_constr.update(call_profit.tolist())
        M.constraint('E_geq_U', Expr.sub(Obj, utility_W0), Domain.greaterThan(0.0))
        M.objective('Call_price_OBJECTIVE', ObjectiveSense.Maximize, Call)
        M.solve()
        if solver_info:
            price_time = M.getSolverDoubleInfo('optimizerTime')
            price_iter = M.getSolverIntInfo('intpntIter')
            n_cons = M.getSolverIntInfo('optNumcon')
            n_vars = M.getSolverIntInfo('optNumvar')
            return (M.primalObjValue(), n_cons, n_vars, obj_info[0], obj_info[1], price_iter, price_time)
        else:
            return M.primalObjValue()
    return (Option_Portfolio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, to demonstrate the model, we solve it for two simple cases. The values taken below are the same as mentioned in the paper (<a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>).""")
    return


@app.cell
def _(Option_Portfolio, np, price_vector_abc, price_vector_z):
    T = 5
    sigma = np.asarray([0.2])
    q = np.ones(3) / 3
    eps = np.asarray([[np.sqrt(2)], [-1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
    T_horizon = 1 / 4
    dT = T_horizon / T
    theta = [0.005]
    S_v_0 = [1.0]
    r = 1.06 ** dT - 1
    W_0 = 1.0
    gamma = 0.3
    b = 1
    c = 0.2
    a = b / (1 / c + W_0 / (gamma - 1))
    K = 1
    mu = 0.15
    Z = price_vector_z(sigma, eps, dT, q)
    S_coeffs = np.expand_dims(price_vector_abc(sigma, eps, dT, q, mu, Z), axis=1)
    _util_paras = np.asarray([a, b, gamma])
    _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, _util_paras, q]
    print('\n\nCall option price = {}'.format(Option_Portfolio(_input_pars, K)))
    return K, S_coeffs, S_v_0, T, W_0, b, c, eps, mu, q, r, sigma, theta


@app.cell
def _(K, Option_Portfolio, S_coeffs, S_v_0, T, W_0, c, q, r, theta):
    _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, c, q]
    print('\n\nCall option price = {}'.format(Option_Portfolio(_input_pars, K, util_type='EXP')))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Test-cases:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Reservation price sensitivity for the choice of utility function:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(a.) Dependency of the reservation purchase price on $\gamma$, $ARA(W_0)$ and $\overline{T}$.</b> (Consider Table 3, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(K, Option_Portfolio, S_coeffs, S_v_0, T, W_0, b, c, np, q, r, theta):
    gamma_1 = [-4.0, -2.0, -0.9, -0.3, 0.3, 0.6]
    print('ARA = 0.2 ; Time Horizon = 1/4 year')
    print('{0:^6}  {1:^9}'.format('gamma', 'C'))
    for _g in gamma_1:
        a_1 = b / (1 / c + W_0 / (_g - 1))
        _util_paras = np.asarray([a_1, b, _g])
        _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, _util_paras, q]
        print('{0:^ 5.1f}  {1:^9.7f}'.format(_g, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, c, q]
    print('{0:^5}  {1:^9.7f}'.format('exp', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return (gamma_1,)


@app.cell
def _(
    K,
    Option_Portfolio,
    S_coeffs,
    S_v_0,
    T,
    W_0,
    b,
    gamma_1,
    np,
    q,
    r,
    theta,
):
    c_11 = 1.0
    print('ARA = 1.0 ; Time Horizon = 1/4 year')
    print('{0:^6}  {1:^9}'.format('gamma', 'C'))
    for _g in gamma_1[0:4]:
        a_2 = b / (1 / c_11 + W_0 / (_g - 1))
        _util_paras = np.asarray([a_2, b, _g])
        _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, _util_paras, q]
        print('{0:^ 5.1f}  {1:^9.7f}'.format(_g, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T, W_0, S_coeffs, theta, S_v_0, r, c_11, q]
    print('{0:^5}  {1:^9.7f}'.format('exp', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return


@app.cell
def _(
    K,
    Option_Portfolio,
    S_v_0,
    T,
    W_0,
    b,
    eps,
    gamma_1,
    mu,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    sigma,
    theta,
):
    T_horizon_1 = 9
    dT_1 = T_horizon_1 / T
    _Z = price_vector_z(sigma, eps, dT_1, q)
    S_coeffs_1 = np.expand_dims(price_vector_abc(sigma, eps, dT_1, q, mu, _Z), axis=1)
    r_1 = 1.06 ** dT_1 - 1
    c_2 = 0.2
    print('ARA = 0.2 ; Time Horizon = 9 years')
    print('{0:^6}  {1:^9}'.format('gamma', 'C'))
    for _g in gamma_1:
        a_3 = b / (1 / c_2 + W_0 / (_g - 1))
        _util_paras = np.asarray([a_3, b, _g])
        _input_pars = [T, W_0, S_coeffs_1, theta, S_v_0, r_1, _util_paras, q]
        print('{0:^ 5.1f}  {1:^9.7f}'.format(_g, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T, W_0, S_coeffs_1, theta, S_v_0, r_1, c_2, q]
    print('{0:^5}  {1:^9.7f}'.format('exp', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return S_coeffs_1, r_1


@app.cell
def _(
    K,
    Option_Portfolio,
    S_coeffs_1,
    S_v_0,
    T,
    W_0,
    b,
    gamma_1,
    np,
    q,
    r_1,
    theta,
):
    c_3 = 1.0
    print('ARA = 1.0 ; Time Horizon = 9 years')
    print('{0:^6}  {1:^9}'.format('gamma', 'C'))
    for _g in gamma_1[0:4]:
        a_4 = b / (1 / c_3 + W_0 / (_g - 1))
        _util_paras = np.asarray([a_4, b, _g])
        _input_pars = [T, W_0, S_coeffs_1, theta, S_v_0, r_1, _util_paras, q]
        print('{0:^ 5.1f}  {1:^9.7f}'.format(_g, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T, W_0, S_coeffs_1, theta, S_v_0, r_1, c_3, q]
    print('{0:^5}  {1:^9.7f}'.format('exp', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(b.) Dependency of the reservation purchase price on the initial level of Relative Risk Aversion, $RRA(W_0)$</b> (Consider Table 4, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    K,
    Option_Portfolio,
    S_v_0,
    eps,
    mu,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    sigma,
    theta,
):
    T_1 = 5
    T_horizon_2 = 1 / 4
    dT_2 = T_horizon_2 / T_1
    r_2 = 1.06 ** dT_2 - 1
    W_0_1 = [1.0, 4.0, 8.0]
    gamma_2 = -1.0
    b_1 = 1
    c_4 = 0.2
    Z2 = price_vector_z(sigma, eps, dT_2, q)
    S_coeffs_2 = np.expand_dims(price_vector_abc(sigma, eps, dT_2, q, mu, Z2), axis=1)
    print('ARA = 0.2 ; Time Horizon = 1/4 year')
    print('{0:^5}  {1:^5}  {2:^9}'.format('W_0', 'RRA', 'C'))
    for _w0 in W_0_1:
        a_5 = b_1 / (1 / c_4 + _w0 / (gamma_2 - 1))
        _util_paras = np.asarray([a_5, b_1, gamma_2])
        _input_pars = [T_1, _w0, S_coeffs_2, theta, S_v_0, r_2, _util_paras, q]
        print('{0:^5.0f}  {1:^5.1f}  {2:^9.7f}'.format(_w0, c_4 * _w0, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T_1, 1.0, S_coeffs_2, theta, S_v_0, r_2, c_4, q]
    print('{0:^5}  {1:^5}  {2:^9.7f}'.format('exp', '', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return T_1, W_0_1, b_1, c_4, gamma_2


@app.cell
def _(
    K,
    Option_Portfolio,
    S_v_0,
    T_1,
    W_0_1,
    b_1,
    c_4,
    eps,
    gamma_2,
    mu,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    sigma,
    theta,
):
    T_horizon_3 = 9
    dT_3 = T_horizon_3 / T_1
    r_3 = 1.06 ** dT_3 - 1
    Z3 = price_vector_z(sigma, eps, dT_3, q)
    S_coeffs_3 = np.expand_dims(price_vector_abc(sigma, eps, dT_3, q, mu, Z3), axis=1)
    print('ARA = 0.2 ; Time Horizon = 9 years')
    print('{0:^5}  {1:^5}  {2:^9}'.format('W_0', 'RRA', 'C'))
    for _w0 in W_0_1:
        a_6 = b_1 / (1 / c_4 + _w0 / (gamma_2 - 1))
        _util_paras = np.asarray([a_6, b_1, gamma_2])
        _input_pars = [T_1, _w0, S_coeffs_3, theta, S_v_0, r_3, _util_paras, q]
        print('{0:^5.0f}  {1:^5.1f}  {2:^9.7f}'.format(_w0, c_4 * _w0, Option_Portfolio(_input_pars, K, writeLog=False)))
    _input_pars = [T_1, 1.0, S_coeffs_3, theta, S_v_0, r_3, c_4, q]
    print('{0:^5}  {1:^5}  {2:^9.7f}'.format('exp', '', Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(c.) Convergence of the reservation purchase price.</b> (Consider Table 5, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    K,
    Option_Portfolio,
    S_v_0,
    eps,
    mu,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    sigma,
    theta,
):
    T_horizon_4 = 1 / 4
    W_0_2 = 1.0
    gamma_3 = 0.3
    b_2 = 0.0
    c_5 = 0.7
    a_7 = 1 - gamma_3
    _util_paras = np.asarray([a_7, b_2, gamma_3])
    C_trio = []
    print('{0:^3}  {1:12}  {2:12}  {3:^12}'.format(' ', '-exp(-0.7W)', '(W^0.3)/0.3', ' '))
    print('{0:^3}  {1:^12}  {2:^12}  {3:^12}'.format('T', 'C_exp', 'C_pow', 'CRR'))
    for T_2 in np.arange(1, 12):
        dT_4 = T_horizon_4 / T_2
        r_4 = 1.06 ** dT_4 - 1
        Z4 = price_vector_z(sigma, eps, dT_4, q)
        S_coeffs_4 = np.expand_dims(price_vector_abc(sigma, eps, dT_4, q, mu, Z4), axis=1)
        _input_pars = [T_2, W_0_2, S_coeffs_4, theta, S_v_0, r_4, _util_paras, q]
        _C_pow = Option_Portfolio(_input_pars, K, writeLog=False)
        _input_pars = [T_2, W_0_2, S_coeffs_4, [0.0], S_v_0, r_4, _util_paras, q]
        CRR = Option_Portfolio(_input_pars, K, writeLog=False)
        _input_pars = [T_2, W_0_2, S_coeffs_4, theta, S_v_0, r_4, c_5, q]
        _C_exp = Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)
        print('{0:^3d}  {1:^12.10f}  {2:^12.10f}  {3:^12.10f}'.format(T_2, _C_exp, _C_pow, CRR))
        C_trio.append([_C_exp, _C_pow, CRR])
    return T_horizon_4, W_0_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(d.) Reservation purchase price dependence on absolute risk aversion (Exponential utility).</b> (Consider Table 6, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    K,
    Option_Portfolio,
    S_v_0,
    T_horizon_4,
    W_0_2,
    eps,
    mu,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    sigma,
):
    T_3 = 9
    dT_5 = T_horizon_4 / T_3
    theta_1 = [0.005]
    r_5 = 1.06 ** dT_5 - 1
    c_6 = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0]
    Z5 = price_vector_z(sigma, eps, dT_5, q)
    S_coeffs_5 = np.expand_dims(price_vector_abc(sigma, eps, dT_5, q, mu, Z5), axis=1)
    print('{0:^8}  {1:^12}'.format('ARA(W_0)', 'C[ARA(W_0)]'))
    C_ara = []
    for kappa in c_6:
        _input_pars = [T_3, W_0_2, S_coeffs_5, theta_1, S_v_0, r_5, kappa, q]
        _C_exp = Option_Portfolio(_input_pars, K, util_type='EXP', writeLog=False)
        print('{0:^8.2f}  {1:^12.10f}'.format(kappa, _C_exp))
        C_ara.append(_C_exp)
    return S_coeffs_5, T_3, dT_5, r_5, theta_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(e.) Reservation purchase price dependence on absolute risk aversion (Power utility).</b> (Consider Table 7, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(K, Option_Portfolio, S_coeffs_5, S_v_0, T_3, W_0_2, np, q, r_5, theta_1):
    _eta = np.linspace(0.1, 0.9, 9)
    a_8 = 1
    b_3 = 0.0
    print('{0:^5}  {1:^12}'.format('eta', 'C[ARA(W_0)]'))
    C_eta = []
    for n in _eta:
        gamma_4 = 1 - n
        _util_paras = np.asarray([a_8, b_3, gamma_4])
        _input_pars = [T_3, W_0_2, S_coeffs_5, theta_1, S_v_0, r_5, _util_paras, q]
        _C_pow = Option_Portfolio(_input_pars, K, writeLog=False)
        print('{0:^5.2f}  {1:^12.10f}'.format(n, _C_pow))
        C_eta.append(_C_pow)
    return a_8, b_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(f.) Reservation purchase price dependence on initial wealth (Power utility with $\eta = 0.7$).</b> (Consider Table 8, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    K,
    Option_Portfolio,
    S_coeffs_5,
    S_v_0,
    T_3,
    a_8,
    b_3,
    np,
    q,
    r_5,
    theta_1,
):
    _eta = 0.7
    W_0_3 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    gamma_5 = 1 - _eta
    _util_paras = np.asarray([a_8, b_3, gamma_5])
    print('{0:^5}  {1:^12}'.format('W_0', 'C_pow(W_0)'))
    C_w0 = []
    for _w0 in W_0_3:
        _input_pars = [T_3, _w0, S_coeffs_5, theta_1, S_v_0, r_5, _util_paras, q]
        _C_pow = Option_Portfolio(_input_pars, K, writeLog=False)
        print('{0:^5.2f}  {1:^12.10f}'.format(_w0, _C_pow))
        C_w0.append(_C_pow)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### The effect of diversification opportunities on the reservation purchase price.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The model presented above is fairly general, and therefore we can easily extend its application to the case where trade takes place in multiple securities. For illustration, we present the case where there are two risky securities, following <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>. For this example, we have:

    $$
    \sigma^{T} = \begin{pmatrix}
    \sigma_{11}\, ,\,\sigma_{21} \\
    \sigma_{12}\, ,\,\sigma_{22} \\
    \end{pmatrix} = \begin{pmatrix}
    \sigma_{11}\, ,\, 0.2\rho\\
    0.0\, , \, \sqrt{(0.2)^2 - \sigma_{21}^2} \\
    \end{pmatrix}
    $$

    where $\rho$ is the correlation between logarithms of the two securities. The $\epsilon$ matrix is given by, 

    $$\epsilon = \begin{pmatrix}
    \epsilon_1 (\omega_1),\epsilon_2 (\omega_1) \\
    \epsilon_1 (\omega_2),\epsilon_2 (\omega_2) \\
    \epsilon_1 (\omega_3),\epsilon_2 (\omega_3)
    \end{pmatrix} = \begin{pmatrix}
    \sqrt{2}\,\,,\,\,0 \\
    -1/\sqrt{2},\sqrt{3/2} \\
    -1/\sqrt{2},\sqrt{3/2}
    \end{pmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Moreover, we will consider different values for the transaction costs. There are three situations considered below: completely frictionless market, friction in one security, friction in both securities.

    <b>Note</b>: In the code, we use the transpose of the sigma matrix, hence the shapes of the arrays that represent $\sigma$ and $\epsilon$ must be maintained similar to what is shown in the code below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(g.) Reservation purchase price of a call option for T=9 as a function of $\rho$ and $\theta_i$ in the presence of two risky securities.</b> (Consider Table 10, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    Option_Portfolio,
    T_3,
    dT_5,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    r_5,
):
    W_0_4 = 1.0
    gamma_6 = 0.3
    b_4 = 1
    c_7 = 0.2
    a_9 = b_4 / (1 / c_7 + W_0_4 / (gamma_6 - 1))
    _util_paras = np.asarray([a_9, b_4, gamma_6])
    K_1 = 1
    sig11 = 0.2
    sig12 = 0.0
    eps_1 = np.asarray([[np.sqrt(2), 0], [-1 / np.sqrt(2), np.sqrt(3 / 2)], [-1 / np.sqrt(2), -np.sqrt(3 / 2)]])
    S_v_0_1 = [1.0, 1.0]
    mu_1 = np.asarray([0.15, 0.15])
    rho = [1, 0.5, 0.0, -0.5, -0.9]
    theta_2 = [0.0, 0.0]
    print('theta_1 = {0:5.3f} ; theta_2 = {1:5.3f}'.format(theta_2[0], theta_2[1]))
    print('{0:^5}  {1:^12}'.format('rho', 'C(S1,S2)'))
    for _p in rho:
        _sig21 = _p * 0.2
        sigma_1 = np.asarray([[sig11, _sig21], [sig12, np.sqrt(0.2 ** 2 - _sig21 ** 2)]])
        Z6 = price_vector_z(sigma_1, eps_1, dT_5, q)
        S_coeffs_6 = price_vector_abc(sigma_1, eps_1, dT_5, q, mu_1, Z6)
        _input_pars = [T_3, W_0_4, S_coeffs_6, theta_2, S_v_0_1, r_5, c_7, q]
        print('{0:^ 4.1f}  {1:^12.10f}'.format(_p, Option_Portfolio(_input_pars, K_1, util_type='EXP', writeLog=False)))
    return K_1, S_v_0_1, W_0_4, c_7, eps_1, mu_1, rho, sig11, sig12


@app.cell
def _(
    K_1,
    Option_Portfolio,
    S_v_0_1,
    T_3,
    W_0_4,
    c_7,
    dT_5,
    eps_1,
    mu_1,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    r_5,
    rho,
    sig11,
    sig12,
):
    theta_3 = [0.005, 0.0]
    print('theta_1 = {0:5.3f} ; theta_2 = {1:5.3f}'.format(theta_3[0], theta_3[1]))
    print('{0:^5}  {1:^12}'.format('rho', 'C(S1,S2)'))
    for _p in rho:
        _sig21 = _p * 0.2
        sigma_2 = np.asarray([[sig11, _sig21], [sig12, np.sqrt(0.2 ** 2 - _sig21 ** 2)]])
        Z7 = price_vector_z(sigma_2, eps_1, dT_5, q)
        S_coeffs_7 = price_vector_abc(sigma_2, eps_1, dT_5, q, mu_1, Z7)
        _input_pars = [T_3, W_0_4, S_coeffs_7, theta_3, S_v_0_1, r_5, c_7, q]
        print('{0:^ 4.2f}  {1:^12.10f}'.format(_p, Option_Portfolio(_input_pars, K_1, util_type='EXP', writeLog=False)))
    return


@app.cell
def _(
    K_1,
    Option_Portfolio,
    S_v_0_1,
    T_3,
    W_0_4,
    c_7,
    dT_5,
    eps_1,
    mu_1,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    r_5,
    rho,
    sig11,
    sig12,
):
    theta_4 = [0.005, 0.005]
    print('theta_1 = {0:5.3f} ; theta_2 = {1:5.3f}'.format(theta_4[0], theta_4[1]))
    print('{0:^5}  {1:^12}'.format('rho', 'C(S1,S2)'))
    for _p in rho:
        _sig21 = _p * 0.2
        sigma_3 = np.asarray([[sig11, _sig21], [sig12, np.sqrt(0.2 ** 2 - _sig21 ** 2)]])
        Z8 = price_vector_z(sigma_3, eps_1, dT_5, q)
        S_coeffs_8 = price_vector_abc(sigma_3, eps_1, dT_5, q, mu_1, Z8)
        _input_pars = [T_3, W_0_4, S_coeffs_8, theta_4, S_v_0_1, r_5, c_7, q]
        print('{0:^ 4.2f}  {1:^12.10f}'.format(_p, Option_Portfolio(_input_pars, K_1, util_type='EXP', writeLog=False)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(h.) Reservation purchase price of a call option for T=9, as a function of $\theta_i$, in the case of two risky securities; $\rho = -0.9$.</b> (Consider Table 11, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell
def _(
    K_1,
    Option_Portfolio,
    S_v_0_1,
    T_3,
    W_0_4,
    c_7,
    dT_5,
    eps_1,
    mu_1,
    np,
    price_vector_abc,
    price_vector_z,
    q,
    r_5,
    sig11,
    sig12,
):
    rho_1 = -0.9
    _sig21 = rho_1 * 0.2
    sigma_4 = np.asarray([[sig11, _sig21], [sig12, np.sqrt(0.2 ** 2 - _sig21 ** 2)]])
    Z9 = price_vector_z(sigma_4, eps_1, dT_5, q)
    S_coeffs_9 = price_vector_abc(sigma_4, eps_1, dT_5, q, mu_1, Z9)
    theta_list = [0.0008, 0.0016, 0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    print('{0:^5}  {1:^12}'.format('theta', 'C(S1,S2)'))
    for theta_i in theta_list:
        theta_5 = [theta_i, theta_i]
        _input_pars = [T_3, W_0_4, S_coeffs_9, theta_5, S_v_0_1, r_5, c_7, q]
        print('{0:^5.4f}  {1:^12.10f}'.format(theta_i, Option_Portfolio(_input_pars, K_1, util_type='EXP', writeLog=False)))
    return (sigma_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<b>(i.) Computational efficiency for different values of T (time periods).</b> (Consider Table 12, <a href="https://www.sciencedirect.com/science/article/pii/S0168927498001044">Andersen et. al.</a>)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The columns of the output table in the following cell denote: T (time periods), cons. (number of constraints), vars. (number of variables), it. (number of iterations for the <i>no-option</i> model as well as the <i>option-price</i> model), time. (time taken, in seconds for both models).""")
    return


@app.cell
def _(
    K_1,
    OptimizerError,
    Option_Portfolio,
    S_v_0_1,
    SolutionError,
    W_0_4,
    c_7,
    dT_5,
    eps_1,
    mu_1,
    price_vector_abc,
    price_vector_z,
    q,
    sigma_4,
):
    theta_6 = [0.005, 0.005]
    print('{0:^2}  {1:^8}  {2:^10}  {3:^11}  {4:^11}'.format('', '', '', 'No-option', 'Option-price'))
    print('{0:>2}  {1:>8}  {2:>10}  {3:^3}  {4:^6}  {5:^3}  {6:^6}'.format('T', 'cons.', 'vars.', 'it.', 'time', 'it.', 'time'))
    total_info_arr = []
    for T_4 in range(1, 10):
        try:
            T_horizon_5 = 1 / 4
            dt = T_horizon_5 / T_4
            r_6 = 1.06 ** dT_5 - 1
            Z10 = price_vector_z(sigma_4, eps_1, dT_5, q)
            S_coeffs_10 = price_vector_abc(sigma_4, eps_1, dT_5, q, mu_1, Z10)
            _input_pars = [T_4, W_0_4, S_coeffs_10, theta_6, S_v_0_1, r_6, c_7, q]
            call, n_cons, n_var, ut_it, ut_time, price_it, price_time = Option_Portfolio(_input_pars, K_1, util_type='EXP', writeLog=False, solver_info=True)
            print('{0:>2d}  {1:>8d}  {2:>10d}  {3:>3d}  {4:>6.3f}  {5:>3d}  {6:>6.3f}'.format(T_4, n_cons, n_var, ut_it, ut_time, price_it, price_time))
            total_info_arr.append([call, n_cons, n_var, ut_it, ut_time, price_it, price_time])
        except MemoryError as e:
            print(e)
            break
        except SolutionError as s:
            print(s)
        except OptimizerError as e:
            print(e)
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The calculations shown above were performed on a desktop with 15.6 GB of RAM and an Intel$^\circledR$ Core$^{\text{TM}}$ i7-6770HQ CPU @ 2.6 GHz $\times$ 8.""")
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
