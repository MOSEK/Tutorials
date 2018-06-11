from mosek import *
import sys
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# Encode bricks and positions
def encodeBrick(n, m, t, p, q, l):
    return p*m*t + q*t + l
def encodePos(n, m, p, q):
    return p*m + q

# The shape of a rectangle 
def shape_rectangle(a, b):
    return list(product(range(0, a), range(0, b)))

# Shapes of a subset of Tetris blocks
shapes_tetris = [
    [(0,0), (0,1), (0,2), (-1,1)],
    [(0,0), (0,1), (0,2), (1,1)],
    [(0,0), (0,1), (-1,1), (-2,1)],
    [(0,0), (0,1), (1,1), (1,2)],
    [(0,0), (0,1), (-1,1), (-1,2)],
    [(0,0), (1,0), (1,1), (1,2)]
]

# Anchor a shape at a given point p,q
# return the list of coordinates it occupies
# or return None if it would go outside the board or cover a forbidden spot (from noncov)
def anchorShape(shp, n, m, p, q, noncov=[]):
    pts = [(p + x, q + y) for x,y in shp]
    if all(0<= x and x<n and 0<=y and y<m and (x,y) not in noncov for x,y in pts):
        return pts
    else:
        return None

# Plot a solution
def display(n, m, sol, T, col):
    fig,ax = plt.subplots(1)
    # Plot all small squares for each brick
    for p,q,k in sol:
        for x,y in anchorShape(T[k], n, m, p, q):
            ax.add_patch(patches.Rectangle((x,y), 1, 1, linewidth=0, facecolor=col[k]))
    # Plot grid
    xs, ys = np.linspace(0, n, n+1), np.linspace(0, m, m+1)
    for x in xs: plt.plot([x, x], [ys[0], ys[-1]], color='black')
    for y in ys: plt.plot([xs[0], xs[-1]], [y, y], color='black') 
    ax.axis([0, n, 0, m])
    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()

# Check if the linear relaxation is infeasible
# And if so, print the infeasibility certificate
# as a labeling of the rectangle grid.
def attemptCertificate(n, m, noncov, task):
    # Now we make the problem continuous
    task.putvartypelist(range(task.getnumvar()), [variabletype.type_cont] * task.getnumvar())
    task.optimize()
    if task.getprosta(soltype.itr) == prosta.prim_infeas:
        # Read the dual variables containing the certificate
        y = np.zeros(n * m, dtype=float)
        task.getyslice(soltype.itr, 0, n * m, y)
        for p in range(n):
            print(' '.join('    ' if (p,q) in noncov else '{: 3.1f}'.format(y[encodePos(n, m, p, q)])for q in range(m)))
        print('Certificate with sum = {0}'.format(sum(y)))
    else:
        print('No certificate found')


# Build a model for n x m rectangle with brick shapes T
# noncov is the list of fields not to be covered
# exact = True -  exact covering
# exact = False - find a covering of maximal area
# rep   = max number of repetitions of each brick, 0 denotes no limit
def model(n, m, t, T, noncov=[], exact=True, rep=0, timelimit=60.0, logging=None):
    numvar = n * m * t
    numcon = n * m

    with Env() as env:
        with env.Task(numcon, numvar) as task:
            # Add variables and make them binary
            task.appendvars(numvar)
            task.appendcons(numcon)
            task.putvartypelist(range(numvar), [variabletype.type_int] * numvar)
            task.putvarboundslice(0, numvar, [boundkey.ra] * numvar, [0.0] * numvar, [1.0] * numvar)

            # List of forbidden positions
            forb = []

            for p,q,k in product(range(n), range(m), range(t)):
                # Find points covered by the shape
                pts = anchorShape(T[k], n, m, p, q, noncov)
                bCode = encodeBrick(n,m,t,p,q,k)
                if pts is None:
                    forb.append(bCode)
                else:
                    task.putacol(bCode, [encodePos(n,m,x,y) for x,y in pts], [1.0] * len(pts))

            # Require all fields to be exactly once or at most once
            # Except for the positions in noncov
            key = boundkey.fx if exact else boundkey.up
            task.putconboundslice(0, numcon, [key] * numcon, [1.0] * numcon, [1.0] * numcon)
            task.putconboundlist([encodePos(n,m,x,y) for x,y in noncov], [boundkey.fx] * len(noncov), [0.0] * len(noncov), [0.0] * len(noncov))

            # Objective - total area covered
            # This makes no difference in the exact covering (feasibility) problem            
            areas = [ (encodeBrick(n,m,t,p,q,k), len(T[k])) for p,q,k in product(range(n), range(m), range(t)) ]
            subj, val = zip(*areas)
            task.putclist(subj, val)
            task.putobjsense(objsense.maximize)

            # Forbidden brick placements
            task.putvarboundlist(forb, [boundkey.fx] * len(forb), [0.0] * len(forb), [0.0] * len(forb))

            # Use each brick at most rep times
            if rep > 0:
                task.appendcons(t)
                task.putconboundslice(numcon, numcon + t, [boundkey.up] * t, [rep] * t, [rep] * t)
                for k in range(t):
                    task.putarow(numcon + k, [ encodeBrick(n,m,t,p,q,k) for p,q in product(range(n), range(m)) ], [1.0] * (n*m))

            # Optimize and get the results back
            if logging:
                task.set_Stream(streamtype.log, logging)
            task.putdouparam(dparam.mio_max_time, timelimit)
            task.optimize()

            prosta = task.getprosta(soltype.itg)
            if prosta == prosta.prim_infeas:
                print("No covering\nLooking for infeasibility certificate for the relaxation")
                attemptCertificate(n, m, noncov, task)
            else:
                xx = np.zeros(numvar, dtype=float)
                task.getxx(soltype.itg, xx)
                sol = [(p,q,k) for p,q,k in product(range(n), range(m), range(t)) if xx[encodeBrick(n,m,t,p,q,k)] > 0.8]
                display(n, m, sol, T, ['blue', 'yellow', 'green', 'red', 'violet', 'orange'])
                if not exact:
                    print("Covered area {0}, best bound found {1}, total board area {2}".format(
                        int(task.getprimalobj(soltype.itg)), 
                        int(task.getdouinf(dinfitem.mio_obj_bound)),
                        n*m-len(noncov)))


def ex1():
    n, m = 21, 21
    T = [shape_rectangle(1,8), shape_rectangle(8,1), shape_rectangle(1,9), shape_rectangle(9,1)]
    t =len(T)
    model(n, m, t, T)

def ex2():
    n, m = 22, 27
    T = [shape_rectangle(8,2), shape_rectangle(5,2), shape_rectangle(1,7)]
    t = len(T)
    model(n, m, t, T)

def ex3():
    model(11, 3, len(shapes_tetris), shapes_tetris, noncov=[], exact=False, rep=1)  

def ex4():
    n, m = 11, 17
    T = shapes_tetris
    t = len(T)
    noncov = [(0,0), (1,3), (9,13), (8,8), (7,7), (5,5), (4,4), (3,3), (3,1), (8,12)]
    model(n, m, t, T, noncov, exact = False, rep = 0, timelimit = 20.0, logging = streamprinter)

def ex5():
    n, m = 15, 15
    T = [shape_rectangle(1,8), shape_rectangle(8,1), shape_rectangle(1,11), shape_rectangle(11,1)]
    t = len(T)
    model(n, m, t, T)

def ex6():
    n, m = 12, 12
    T = [shape_rectangle(1,3), shape_rectangle(3,1)]
    t = len(T)
    noncov = [(0, 0), (0, m-1), (n-1, 0)]
    model(n, m, t, T, noncov)

def ex7():
    n, m = 32, 32
    T = [shape_rectangle(1,13), shape_rectangle(13,1), shape_rectangle(14,1), shape_rectangle(1, 14)]
    t = len(T)
    model(n, m, t, T, exact = False, timelimit = 30.0, logging=streamprinter)

ex7()
