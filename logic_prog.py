from kanren import run, var, fact, isvar, membero
from kanren.assoccomm import eq_assoccomm as eq
from kanren.assoccomm import commutative, associative
from kanren.core import success, fail, goaleval, condeseq, eq, var
from sympy.ntheory.generate import prime, isprime
import itertools as it


# define the mathematical operations to be used
add = 'add'
mul = 'mul'

# specify the type of processes
fact(commutative, mul)
fact(commutative, add)
fact(associative, mul)
fact(associative, add)

# define the variables
a, b = var('a'), var('b')

# match the expression with original pattern (5+a)*b
Original_pattern = (mul, (add, 5, a), b)
exp1 = (mul, 2, (add, 3, 1))
exp2 = (add, 5, (mul, 8, 1))

# print the output
print(run(0, (a,b), eq(Original_pattern, exp1)))
print(run(0, (a,b), eq(Original_pattern, exp2)))


# define a function to check prime numbers
def prime_check(x):
    if isvar(x):
        return condeseq([(eq, x, p)] for p in map(prime, it.count(1)))
    else:
        return success if isprime(x) else fail


# declare a variable to be used
x = var()
print((set(run(0,x,(membero,x,(12,14,15,19,20,21,22,23,29,30,41,44,52,62,65,85)),
               (prime_check,x)))))
print((run(10,x,prime_check(x))))