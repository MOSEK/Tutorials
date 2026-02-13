import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    High-frequency transformers are widely used in power electronic systems where compact size and high efficiency are required. Their design involves selecting appropriate winding and operating parameters while respecting physical and performance limitations.

    This notebook focuses on the systematic selection of transformer winding parameters in order to **minimize total power loss**, subject to electrical, geometric, and thermal constraints. The problem is formulated as an optimization model and solved using geometric programming.

    The model is based on the work presented in [Application of Geometric Programming to Transformer Design](https://ieeexplore.ieee.org/document/1532335) (IEEE Transactions on Magnetics, vol. 41, no. 11, pp. 4261-4269, Nov. 2005) by Rabih A. Jabr. 

    Geometric Programming problems are solved by convexifying them in a standard way into an [exponential cone problem](https://docs.mosek.com/modeling-cookbook/expo.html#geometric-programming). For the implementation we use a Geometric Programming Toolbox developed in [our previous notebook](https://docs.mosek.com/tutorials-notebooks/gp-toolbox/gptoolbox.html). It accepts a high-level GP model and transforms it into a conic problem expressed using the Python Fusion API. It is included here as an appendix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling Scope

    The optimization framework developed in this work is designed to support a wide range of transformer design scenarios rather than a single fixed configuration. Its formulation is general and can be adapted to different operating frequencies, magnetic materials, and design objectives through appropriate parameter choices.

    The same model structure applies to both low- and high-frequency transformers. Differences between these regimes are captured through operating frequency and material-dependent loss parameters, allowing the framework to represent laminated-core transformers at low frequency as well as ferrite-based designs at high frequency. In the *high-frequency* case study presented here, a *Mn–Zn ferrite* core is used to demonstrate the method under realistic operating conditions.

    Magnetic material behavior is incorporated through analytical core-loss models and flux-density constraints. Core geometry may either be fixed, representing a preselected commercial core, or treated as a design variable when core sizing is part of the optimization. This flexibility allows the framework to address problems ranging from winding optimization to joint core–winding design.

    Winding design is modeled in detail, including the number of primary and secondary turns, current density, conductor dimensions, and layer structure. Geometric feasibility is enforced through constraints that ensure the windings fit within the available core window while accounting for insulation and manufacturing allowances. High-frequency effects are captured using effective resistance models.

    Electrical, magnetic, geometric, and thermal effects are modeled simultaneously. Core loss, copper loss, voltage regulation, magnetizing current, and temperature rise are represented using analytical expressions that can be formulated in monomial or posynomial form, enabling efficient optimization while capturing the key trade-offs in practical transformer design.

    Different design objectives can be explored within the same framework by modifying the optimization objective, including:

    - Minimum mass

    - Minimum loss

    - Maximum apparent power (VA).

    The example presented in this work focuses on a minimum-loss design for a high-frequency transformer with fixed core geometry and foil windings. The optimization selects the winding configuration and operating point, specifically the number of turns, number of layers, flux density, and current density, such that losses are minimized while all electrical, thermal, and geometric constraints are satisfied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mathematical Model

    ## Decision Variables

    All decision variables are strictly positive.

    | Symbol | Description | Unit |
    |------|------------|------|
    | \( N_p \) | Number of primary turns | – |
    | \( N_s \) | Number of secondary turns | – |
    | \( N_{ls} \) | Number of secondary winding layers | – |
    | \( B \) | Peak magnetic flux density | T |
    | \( J \) | Current density | A/m\(^2\) |

    ---

    ## Auxiliary Variables

    | Symbol | Description | Unit |
    |------|------------|------|
    | \( \bar{P}_{c} \) | Upper bound on core loss | W |
    | \( \bar{P}_{cu} \) | Upper bound on copper loss | W |
    | \( \bar{V}_{R,r} \) | Upper bound on resistive voltage regulation | – |
    | \( \bar{V}_{R,x} \) | Upper bound on reactive voltage regulation | – |
    | \( \bar{I}_m \) | Upper bound on magnetizing current | A |
    | \( \bar{h} \) | Thermal resistance factor | K/W |

    ---

    ## Parameters

    | **Symbol** | Description | Unit |  | **Symbol** | Description | Unit |
    |-----------|------------|------|--|-----------|------------|------|
    | **\( V_p \)** | Rated primary voltage | V |  | **\( V_s \)** | Rated secondary voltage | V |
    | **\( P_o \)** | Rated output power | W |  | **\( f \)** | Operating frequency | Hz |
    | **\( I_p \)** | Rated primary current | A |  | **\( I_s \)** | Rated secondary current | A |
    | **\( A_c \)** | Core cross-sectional area | m\(^2\) |  | **\( V_c \)** | Core volume | m\(^3\) |
    | **\( c \)** | Core window depth | m |  | **\( t \)** | Core thickness | m |
    | **\( b_w \)** | Window width | m |  | **\( h_w \)** | Window height | m |
    | **\( \rho_w \)** | Copper resistivity | Ω·m |  | **\( d \)** | Foil thickness | m |
    | **\( F_r \)** | AC/DC resistance ratio | – |  | **\( F_\ell \)** | Layer utilization factor | – |
    | **\( B_{\max} \)** | Maximum flux density | T |  | **\( \Delta T_{\max} \)** | Maximum temperature rise | K |
    | **\( k_c \)** | Core loss coefficient | – |  | **\( \alpha \)** | Frequency exponent | – |
    | **\( \beta \)** | Flux density exponent | – |  | **\( k_v \)** | Waveform constant | – |



    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All formulas are numbered as in the original paper. The derivation of the geometric programming formulation and the detailed discussion of the individual constraints can be found in the referenced work. The implemented model used in this study is defined in the following section.

    ### Objective Function

    \[
    \min \quad \bar{P}_{c} + \bar{P}_{cu}
    \]


    ### Induced Voltage and Turns Ratio Constraints

    \[
    \tag{33}
    2^{3/2}\,\pi\,k_f\,f\,E_p^{-1}\,N_p\,B\,c\,t = 1
    \]

    \[
    \tag{34}
    E_s\,E_p^{-1}\,N_s^{-1}\,N_p = 1
    \]


    ### Geometric / Window Utilization Constraints

    \[
    \tag{35}
    2\,k_{Cu}^{-1}\,N_p\,I_p\,J^{-1}\,b_w^{-1}\,h_w^{-1} = 1
    \]

    \[
    \tag{36}
    F_\ell^{-1}\, d^{-1}\, I_p\, J^{-1}\, N_p\, N_s^{-1}\, N_{ls}\, h_w^{-1} = 1
    \]


    ### Thermal Constraints

    \[
    \tag{37}
    \frac{
    \bar{P}_{c}\,\bar{h}^{1/4}\,c^{-1/2}\,t^{-1/2}\,b_w^{-1/2}\,h_w^{-1/2}
    }{
    1.42\sqrt{2}\,k_a\,\Delta T^{5/4}
    } +
    \frac{
    \bar{P}_{cu}\,\bar{h}^{1/4}\,c^{-1/2}\,t^{-1/2}\,b_w^{-1/2}\,h_w^{-1/2}
    }{
    1.42\sqrt{2}\,k_a\,\Delta T^{5/4}
    }
    \;\le\; 1
    \]

    \[
    \tag{38}
    h_w\,\bar{h}^{-1} + 2c\,\bar{h}^{-1} \le 1
    \]


    ### Loss Models and (Optional) Efficiency-Based Loss Bound

    \[
    \tag{39}
    4k_f K_c \rho_c f^{\alpha} B^{\beta} c t b_w\, \bar{P}_c^{-1}
    \;+\;
    8k_f K_c \rho_c f^{\alpha} B^{\beta} c^2 t\, \bar{P}_c^{-1}
    \;+\;
    4k_f K_c \rho_c f^{\alpha} B^{\beta} c t b_w\, \bar{P}_c^{-1}
    \;\le\; 1
    \]

    \[
    \tag{40}
    4k_{Cu}F_r\rho_w c b_w h_w J^{2}\,\bar{P}_{cu}^{-1}
    \;+\;
    2k_{Cu}F_r\rho_w t b_w h_w J^{2}\,\bar{P}_{cu}^{-1}
    \;+\;
    \pi k_{Cu}F_r\rho_w b_w^{2} h_w J^{2}\,\bar{P}_{cu}^{-1}
    \;\le\; 1
    \]

    \[
    \tag{41}
    \frac{\eta_m}{(1-\eta_m)P_o}\,\bar{P}_{cu}
    \;+\;
    \frac{\eta_m}{(1-\eta_m)P_o}\,\bar{P}_{c}
    \;\le\; 1
    \]



    ### Magnetizing / No-Load Current Constraints

    \[
    \tag{42}
    \frac{\sqrt{2}}{\mu}\, B\, N_p^{-1}\, h_w\, \bar{I}_m^{-1}
    \;+\;
    \frac{\sqrt{2}}{\mu}\, B\, N_p^{-1}\, b_w\, \bar{I}_m^{-1}
    \;+\;
    \frac{2\sqrt{2}}{\mu}\, B\, N_p^{-1}\, c\, \bar{I}_m^{-1}
    \;\le\; 1
    \]

    \[
    \tag{43}
    B_{\text{sat}}^{-1}\, B \le 1
    \]

    \[
    \tag{44}
    k_{\varphi}^{-2}\, I_p^{-2}\, \bar{I}_m^2
    \;+\;
    E_p^{-2}\, k_{\varphi}^{-2}\, I_p^{-2}\, \bar{P}_c^{2}
    \;\le\; 1
    \]


    ### Voltage Regulation Constraints

    \[
    \tag{45}
    8F_r\rho_w\,\text{pf}\,N_p c\, J\, \bar{V}_R^{-1}
    \;+\;
    4F_r\rho_w\,\text{pf}\,N_p t\, J\, \bar{V}_R^{-1}
    \;+\;
    2\pi F_r\rho_w\,\text{pf}\,N_p b_w\, J\, \bar{V}_R^{-1}
    \;\le\; 1
    \]

    \[
    \tag{46}
    \frac{8\pi}{3p^{2}}\, I_p\, r_f\, f\, \mu_0\, N_p^{2}\, c\, b_w\, h_w^{-1}\, \bar{V}_X^{-1}
    \;+\;
    \frac{4\pi}{3p^{2}}\, I_p\, r_f\, f\, \mu_0\, N_p^{2}\, t\, b_w\, h_w^{-1}\, \bar{V}_X^{-1}
    \;+\;
    \frac{2\pi^{2}}{3p^{2}}\, I_p\, r_f\, f\, \mu_0\, N_p^{2}\, b_w^{2}\, h_w^{-1}\, \bar{V}_X^{-1}
    \;\le\; 1
    \]

    \[
    \tag{48}
    E_p^{-1}\, {VR}_{m}^{-1}\, \bar{V}_R
    \;+\;
    E_p^{-1}\, {VR}_{m}^{-1}\, \bar{V}_X
    \;\le\; 1
    \]


    ### Discrete / Structural Bounds

    \[
    \tag{47}
    N_{ls}^{-1} \le 1,
    \qquad
    m\,N_s^{-1}\,N_{ls} \le 1
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As a helper for writing down the model, given its size, we introduce a compact variable indexing scheme.
    The optimization variables are listed in `var_names` and stored internally as a vector.
    The dictionary `vid` maps each symbolic variable name to its corresponding index in this vector, enabling an efficient numerical representation of monomials in the GP solver.
    The variable `n` denotes the total number of optimization variables.
    """)
    return


@app.cell
def _():
    var_names = [
            "c","t","bw","hw","Np","Ns","Nls","B","J",
            "Pcbar","Pcubar","VRrbar","VRxbar","Imbar","hbar"
        ]
    vid = {name:i for i,name in enumerate(var_names)}
    n = len(var_names)
    return n, vid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monomial Construction Helper

    The function `mon(coef, exps)` is a helper routine used to construct monomial terms in a readable way.

    The argument `coef` represents the positive scalar coefficient of the monomial, while `exps` is a dictionary that maps variable names to their corresponding exponents.
    Using the variable index mapping `vid`, the function converts symbolic variable names into numerical indices required by the GP solver.

    Variables with zero exponent are omitted to keep the monomial representation sparse.
    The function returns a `Monomial` object in the internal format expected by the optimization toolbox.
    """)
    return


@app.cell
def _(Monomial, vid):
    def mon(coef, exps):
        idx, alp = [], []
        for k, v in exps.items():
            if v == 0: 
                continue
            idx.append(vid[k])
            alp.append(v)
        return Monomial.add(coef, idx, alp)
    return (mon,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The function `fix_var(gp, k, name, value)` enforces a fixed value for an optimization variable using geometric programming–compatible constraints.

    Since GP does not support equality constraints directly, the equality \( x = \text{value} \) is imposed by two monomial inequalities:

    \[
    \frac{x}{\text{value}} \le 1, \qquad \frac{\text{value}}{x} \le 1.
    \]

    These constraints are added sequentially to the GP model starting at index `k`, and the updated constraint index is returned.
    """)
    return


@app.cell
def _(mon):
    def fix_var(gp, k, name, value):
        # Enforce x = value using two monomial inequalities:
        #   x/value <= 1
        #   value/x <= 1
        gp.addConstraint(k,   [mon(1.0/value, {name: 1})]);  k += 1
        gp.addConstraint(k,   [mon(value,     {name:-1})]);  k += 1
        return k
    return (fix_var,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equality Constraint Helper

    The function `add_eq_as_two_ineq(gp, k, coef, exps)` enforces a monomial equality constraint of the form

    \[
    \text{coef} \cdot \prod_i x_i^{a_i} = 1
    \]

    using two GP-compatible inequality constraints:

    \[
    \text{coef} \cdot \prod_i x_i^{a_i} \le 1, \qquad
    \frac{1}{\text{coef}} \cdot \prod_i x_i^{-a_i} \le 1.
    \]

    This construction is necessary because geometric programming supports only inequality constraints.
    The function appends the two constraints starting at index `k` and returns the updated constraint index.
    """)
    return


@app.cell
def _(mon):
    def add_eq_as_two_ineq(gp, k, coef, exps):
        # Enforce coef * Π x^a = 1  via two inequalities:
        #   coef * Π x^a <= 1
        #   (1/coef) * Π x^{-a} <= 1
        gp.addConstraint(k,   [mon(coef, exps)])
        gp.addConstraint(k+1, [mon(1.0/coef, {name: -a for name, a in exps.items()})])
        return k + 2
    return (add_eq_as_two_ineq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Parameters

    This section defines the numerical parameters used in the geometric programming model.

    The parameters include fixed core dimensions for the minimum-loss design, electrical specifications (rated power, voltages, frequency, and power factor), winding and conductor choices, core and copper material properties, electromagnetic constants, and additional empirical factors used in the loss, thermal, and regulation models.
    """)
    return


@app.cell
def _(math):
    # ---- Fixed core dimensions (minimum-loss design) ----
    c0  = 0.005
    t0  = 0.015
    bw0 = 0.010
    hw0 = 0.025

    # ---- Spec (HF ferrite example) ----
    S   = 1200.0      # VA
    Ep  = 300.0       # V
    Es  = 75.0        # V
    f   = 100e3       # Hz
    pf  = 0.80
    rf  = math.sqrt(1.0 - pf**2)
    VRm = 0.03
    eta_m = 0.97
    k_phi = 0.02
    dT  = 65.0

    # Full-load primary current
    Ip = S / Ep       # 4.0 A

    # ---- Winding choices (HF copper foil) ----
    m_layers = 2
    d_foil   = 0.238e-3
    kCu      = 0.6

    # ---- Core material: Mn–Zn ferrite ----
    rho_c = 4800.0
    Kc    = 1.9e-3
    alpha = 1.24
    beta  = 2.0
    Bsat  = 0.4

    # ---- Copper material ----
    rho_Cu = 8920.0
    rho_w  = 2.26e-8

    # ---- EM constants ----
    mu0 = 4e-7 * math.pi
    mu  = mu0 * 2000.0

    # ---- Remaining model factors ----
    kf = 0.95
    ka = 8 * 100**1.75
    Fr = 1.32
    Fl = 0.9
    return (
        Bsat,
        Ep,
        Es,
        Fl,
        Fr,
        Ip,
        Kc,
        S,
        VRm,
        alpha,
        beta,
        bw0,
        c0,
        dT,
        d_foil,
        f,
        hw0,
        kCu,
        k_phi,
        ka,
        kf,
        m_layers,
        mu,
        mu0,
        pf,
        rf,
        rho_Cu,
        rho_c,
        rho_w,
        t0,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following part, all the defined Geometric Programming Model is implemented. For different design options and frequency selection, the objective function and the active constraints can change accordingly.

    For minimum loss design the efficiency constraint (41), is not used as defined in the paper. The full model can be displayed using `printModel()` function of GP Toolbox.
    """)
    return


@app.cell
def _(
    Bsat,
    Ep,
    Es,
    Fl,
    Fr,
    GeometricProgramming,
    Ip,
    Kc,
    VRm,
    add_eq_as_two_ineq,
    alpha,
    beta,
    bw0,
    c0,
    dT,
    d_foil,
    f,
    fix_var,
    hw0,
    kCu,
    k_phi,
    ka,
    kf,
    m_layers,
    math,
    mon,
    mu,
    mu0,
    n,
    pf,
    rf,
    rho_c,
    rho_w,
    t0,
):
    gp = GeometricProgramming(28, n, verbose=True)

    # Objective: minimize total loss (HF minimum-loss design)
    gp.addObjective([
        mon(1.0, {"Pcbar": 1}),
        mon(1.0, {"Pcubar": 1}),
        ])

    k = 0
    # --- Fixed core dimensions for HF minimum-loss design (held fixed)
    k = fix_var(gp, k, "c",  c0)
    k = fix_var(gp, k, "t",  t0)
    k = fix_var(gp, k, "bw", bw0)
    k = fix_var(gp, k, "hw", hw0)

    # (33)
    k = add_eq_as_two_ineq(
        gp, k,
        (2.0**1.5)*math.pi*f*kf*(Ep**-1.0),
        {"Np":1,"B":1,"c":1,"t":1}
    )

    # (34)
    k = add_eq_as_two_ineq(
        gp, k,
        Es*(Ep**-1.0),
        {"Np":1,"Ns":-1}
    )

    # (35)
    k = add_eq_as_two_ineq(
        gp, k,
        2.0*(kCu**-1.0)*Ip,
        {"Np":1,"J":-1,"bw":-1,"hw":-1}
    )


    # (36)
    k = add_eq_as_two_ineq(
        gp, k,
        (Fl**-1.0)*(d_foil**-1.0)*Ip,
        {"J":-1,"Np":1,"Ns":-1,"Nls":1,"hw":-1}
    )

    # --- (37) temperature rise
    denom_T = 1.42*math.sqrt(2.0)*ka*(dT**(5/4))
    gp.addConstraint(k, [
        mon(1.0/denom_T, {"Pcbar":1,"hbar":0.25,"c":-0.5,"t":-0.5,"bw":-0.5,"hw":-0.5}),
        mon(1.0/denom_T, {"Pcubar":1,"hbar":0.25,"c":-0.5,"t":-0.5,"bw":-0.5,"hw":-0.5}),
    ]); k += 1

    # --- (38) hw + 2c <= hbar
    gp.addConstraint(k, [
        mon(1.0, {"hw":1,"hbar":-1}),
        mon(2.0, {"c":1,"hbar":-1}),
    ]); k += 1

    # --- (39) core loss upper bound
    Ccore = kf * Kc * rho_c * (f ** alpha)
    gp.addConstraint(k, [
        mon(4.0 * Ccore, {"B": beta, "c": 1, "t": 1, "hw": 1, "Pcbar": -1}),
        mon(8.0 * Ccore, {"B": beta, "c": 2, "t": 1, "Pcbar": -1}),
        mon(4.0 * Ccore, {"B": beta, "c": 1, "t": 1, "bw": 1, "Pcbar": -1}),
    ]); k += 1


    # --- (40) copper loss upper bound
    Ccu = kCu*Fr*rho_w
    gp.addConstraint(k, [
        mon(4.0*Ccu, {"c":1,"bw":1,"hw":1,"J":2,"Pcubar":-1}),
        mon(2.0*Ccu, {"t":1,"bw":1,"hw":1,"J":2,"Pcubar":-1}),
        mon(math.pi*Ccu, {"bw":2,"hw":1,"J":2,"Pcubar":-1}),
    ]); k += 1

    # # --- (41) efficiency
    # coeff_eta = eta_m/((1.0-eta_m)*Po)
    # gp.addConstraint(k, [
    #     mon(coeff_eta, {"Pcubar":1}),
    #     mon(coeff_eta, {"Pcbar":1}),
    # ]); k += 1

    # --- (42) magnetizing current
    CIm = math.sqrt(2.0)/mu
    gp.addConstraint(k, [
        mon(CIm, {"B":1,"Np":-1,"hw":1,"Imbar":-1}),
        mon(CIm, {"B":1,"Np":-1,"bw":1,"Imbar":-1}),
        mon(2.0*CIm, {"B":1,"Np":-1,"c":1,"Imbar":-1}),
    ]); k += 1

    # --- (43) saturation
    gp.addConstraint(k, [mon(Bsat**-1.0, {"B":1})]); k += 1

    # --- (44) no-load current ratio
    Cnl = (k_phi**-2.0)*(Ip**-2.0)
    gp.addConstraint(k, [
        mon(Cnl, {"Imbar":2}),
        mon(Cnl*(Ep**-2.0), {"Pcbar":2}),
    ]); k += 1


    # --- (45) resistive regulation
    Cr = Fr*rho_w*pf
    gp.addConstraint(k, [
        mon(8.0*Cr, {"Np":1,"c":1,"J":1,"VRrbar":-1}),
        mon(4.0*Cr, {"Np":1,"t":1,"J":1,"VRrbar":-1}),
        mon(2.0*math.pi*Cr, {"Np":1,"bw":1,"J":1,"VRrbar":-1}),
    ]); k += 1

    # --- (46) reactive regulation (substitute p = 2Ns/(m*Nls))
    inv_p2 = (m_layers**2)/4.0
    Cx = Ip*rf*f*mu0*inv_p2
    gp.addConstraint(k, [
        mon((8.0*math.pi/3.0)*Cx, {"Np":2,"c":1,"bw":1,"hw":-1,"VRxbar":-1,"Nls":2,"Ns":-2}),
        mon((4.0*math.pi/3.0)*Cx, {"Np":2,"t":1,"bw":1,"hw":-1,"VRxbar":-1,"Nls":2,"Ns":-2}),
        mon((2.0*(math.pi**2)/3.0)*Cx, {"Np":2,"bw":2,"hw":-1,"VRxbar":-1,"Nls":2,"Ns":-2}),
    ]); k += 1

    # --- (47) Nls >= 1 and Ns >= m*Nls
    gp.addConstraint(k, [mon(1.0, {"Nls":-1})]); k += 1
    gp.addConstraint(k, [mon(float(m_layers), {"Nls":1,"Ns":-1})]); k += 1

    # --- (48) total regulation
    coeff_vr = (Ep**-1.0)*(VRm**-1.0)
    gp.addConstraint(k, [
        mon(coeff_vr, {"VRrbar":1}),
        mon(coeff_vr, {"VRxbar":1}),
    ]); k += 1

    # gp.printModel()
    return (gp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solve the model using `Solve()` function. This function takes input of the desired log file location and name.
    """)
    return


@app.cell
def _(gp):
    sol, x_vals, iterations, elapsedTime = gp.Solve("xfmr_min_mass")
    return elapsedTime, iterations, x_vals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Extract the decision variable values.
    """)
    return


@app.cell
def _(
    Es,
    Ip,
    S,
    bw0,
    c0,
    elapsedTime,
    hw0,
    iterations,
    kCu,
    kf,
    m_layers,
    math,
    pf,
    rho_Cu,
    rho_c,
    t0,
    vid,
    x_vals,
):
    # Extract GP solution variables 
    Np  = x_vals[vid["Np"]]
    Ns  = x_vals[vid["Ns"]]
    Nls = x_vals[vid["Nls"]]
    B   = x_vals[vid["B"]]        # T
    J   = x_vals[vid["J"]]        # A/m^2
    Pc  = x_vals[vid["Pcbar"]]    # W
    Pcu = x_vals[vid["Pcubar"]]   # W

    # Fixed geometry (m)
    c, t, bw, hw = c0, t0, bw0, hw0

    # Geometry (cm)
    c_cm  = c * 100.0
    t_cm  = t * 100.0
    bw_cm = bw * 100.0
    hw_cm = hw * 100.0

    # Winding structure
    p_val = 2.0 * Ns / (m_layers * Nls)

    # Currents
    Ip_val = Ip
    Is_val = S / Es

    # Copper areas (mm^2)
    Acu_p_mm2 = (Ip_val / J) * 1e6
    Acu_s_mm2 = (Is_val / J) * 1e6

    # Masses (kg - g)
    m_core = (
        4.0*kf*rho_c * (c*t*hw) +
        8.0*kf*rho_c * (c**2*t) +
        4.0*kf*rho_c * (c*t*bw)
    )

    m_cu = (
        4.0*kCu*rho_Cu * (c*bw*hw) +
        2.0*kCu*rho_Cu * (t*bw*hw) +
        math.pi*kCu*rho_Cu * (bw**2*hw)
    )

    m_total_g = (m_core + m_cu) * 1e3

    # Magnetic / electrical quantities
    B_mT  = B * 1e3
    J_mm2 = J * 1e-6

    # Efficiency
    Po = S * pf
    eta_pct = 100.0 * Po / (Po + Pc + Pcu)

    results = [
        ("c",        c_cm,                     "cm"),
        ("t",        t_cm,                     "cm"),
        ("bw",       bw_cm,                    "cm"),
        ("hw",       hw_cm,                    "cm"),
        ("Np",       int(round(Np)),           "-"),
        ("Ns",       int(round(Ns)),           "-"),
        ("Nlp",      1,                        "-"),
        ("Nls",      int(round(Nls)),          "-"),
        ("p",        int(round(p_val)),        "-"),
        ("A_cu,p",   Acu_p_mm2,                "mm²"),
        ("A_cu,s",   Acu_s_mm2,                "mm²"),
        ("m_c",      m_core * 1e3,             "g"),
        ("m_cu",     m_cu * 1e3,               "g"),
        ("m_c+m_cu", m_total_g,                "g"),
        ("B",        B_mT,                     "mT"),
        ("J",        J_mm2,                    "A/mm²"),
        ("Pc",       Pc,                       "W"),
        ("Pcu",      Pcu,                      "W"),
        ("η",        eta_pct,                  "%"),

        # ---- separator ----
        ("__SEP__",  None,                     None),

        # ---- solver stats ----
        ("Iterations",  iterations,            ""),
        ("Elapsed time", elapsedTime,           "s"),
    ]

    return (results,)


@app.function(hide_code=True)
def print_table(results, title="Summary"):
    q_w, v_w, u_w = 14, 12, 8
    total = q_w + 1 + v_w + 1 + u_w

    def hline(ch="─"):
        print(ch * total)

    print(title)
    hline("═")
    print(f"{'Quantity':<{q_w}} {'Value':>{v_w}} {'Unit':>{u_w}}")
    hline("─")

    for q, v, u in results:
        if q == "__SEP__":
            hline("─")
            continue

        # Format numbers
        if isinstance(v, float):
            if u == "s":              # time → 3 decimals
                v_str = f"{v:.3f}"
            else:                     # everything else → 2 decimals
                v_str = f"{v:,.2f}"
        else:
            v_str = str(v)

        u_str = "" if u is None else str(u)
        print(f"{q:<{q_w}} {v_str:>{v_w}} {u_str:>{u_w}}")

    hline("═")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following table displays the resulting values of Transformer Design at 100 kHz. The values follow the values showed in the paper at Table 2.
    """)
    return


@app.cell
def _(results):
    print_table(results, title="Transformer design")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix
    """)
    return


@app.cell
def _():
    import numpy as np
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense
    import mosek.fusion.pythonic
    import math

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
        def __init__(self, m, n, verbose=False):
            self.m = m
            self.n = n
            self.constraintsMonomials = [[] for _ in range(m)]
            self.objectiveMonomials = []
            self.totalMonomials = 0
            self.verbose = verbose

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

            if self.verbose:
                import sys
                M.setLogHandler(sys.stdout)

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
            t = M.getSolverDoubleInfo("optimizerTime")  # seconds
            num_iter = M.getSolverIntInfo("intpntIter")
            return sol, vars, num_iter, t
    return GeometricProgramming, Monomial, math


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. The **MOSEK** logo and name are trademarks of <a href="http://mosek.com">Mosek ApS</a>. The code is provided as-is. Compatibility with future release of **MOSEK** or the `Fusion API` are not guaranteed. For more information contact our [support](mailto:support@mosek.com).
    """)
    return


if __name__ == "__main__":
    app.run()