Binary quadratic problems
=========================

A very basic branch-and-bound solver prototype for binary quadratic problems 

```
max  x^T Q x  +  P x  +  R
s.t. x \in {0,1}
```

using SDP relaxations.

Content
--------

* ``binquad.ipynb`` - presentation and discussion
* ``branchbound.py`` - implementation of the solver
* ``example.py`` - basic usage example
* ``qp.py`` - experiments with BiqMac and random instances
* ``bls.py`` - experiments with binary least squares
* ``stats`` - statistics
* ``biq`` - sample of data from BiqMac
