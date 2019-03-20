## Discrete Optimization Coursera course
Goal is to solve optimization problems that are best modelled discretely.
A continuous model is preferred because gradient based methods like stochastic gradient descent
will work much faster (so try to make the problem continuous before falling back on these methods).

Recent research has looked into embedding discrete spaces in continuous latent space: [auto chemical design](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

### Modelling
Problems can be modelled using a graph or integer linear programming.

### Solutions
Either exact (complete) or heuristic.
Use heuristic when:
- problem too large or high dimensional
- formulating problem in exact form too difficult
- feasible solution needed quickly

#### Brute force (exact)
Just enumerate all solutions, check feasibility and cost function. Often O(n!)

#### Branch and bound (exact)
Common general method of dividing into subproblems.
Avoid full enumeration by pruning by bounds checking.

#### Heuristic Principles
1. Construction: incrementally build up solution, such as greedily selecting best
2. Improvement: modify initial feasible solution to improve cost via some kind of neighbourhood (local search, such as with simulated annealing)
3. Partitioning: divide into subproblems which are solved separately (DP)
4. Parallel search: several solutions considered simultaneously then compared against each other (genetic)