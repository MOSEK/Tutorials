# A set of routines for detection of ISS 
# (Irreducible Infeasible Subsets) in linear problems
#
# Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
# Usage: 
#  python3 iis_deletion.py filename
#
# where "filename" name of an input file with an infeasible LP/MILP
from mosek import *
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

def IISFromFile(fname):
	with Task() as task:
		task.readdata(fname)
		success, iis = computeIIS(task)	
	
		if success:
			print(f"{fname}: IIS computation completed successfully, size = {len(iis)}")
			printIIS(task, iis)
		else:
			print(f"{fname}: IIS computation interrupted prematurely because of numerical issues, size = {len(iis)}")

IISFromFile(sys.argv[1])
