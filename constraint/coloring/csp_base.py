import typing
import abc


class Variable:
    """A variable with a domain of possible values for a constraint satisfaction problem.

    Methods are usually accompanied by a "index" method alternative that may be more efficient if you have
    the index of a value. The index of a value will never change since you cannot remove domain values.

    Iteration over a variable is iterating over its visible (unpruned) values.
    """

    def __init__(self, values=()):
        self._values = list(values)
        self._visible_values = [True for _ in self._values]

        self._assigned_value = None

        self._current_iter = 0

    def add_value(self, value):
        self._values.append(value)
        self._visible_values.append(True)

    def assign(self, value):
        self._assigned_value = value

    def unassign(self):
        self._assigned_value = None

    def is_assigned(self):
        return self._assigned_value is not None

    def assigned_value(self):
        return self._assigned_value

    def restore_domain(self):
        for i in range(len(self._visible_values)):
            self._visible_values[i] = True

    def domain_size(self):
        if self._assigned_value is not None:
            return 1
        # only show the visible domains
        return sum(self._visible_values)

    # assume all "index" methods are using an in-bounds index
    def in_domain(self, value):
        if self._assigned_value is not None:
            return self._assigned_value == value
        try:
            i = self._values.index(value)
            return self._visible_values[i]
        except ValueError:
            return False

    def in_domain_index(self, index):
        if self._assigned_value is not None:
            return self._values[index] == self._assigned_value
        return self._visible_values[index]

    def prune(self, value, unprune=False):
        self._visible_values[self._values.index(value)] = unprune

    def prune_at_index(self, index, unprune=False):
        # call with unprune = True to unprune instead of prune
        self._visible_values[index] = unprune

    # iterate over the visible values of a domain
    def __iter__(self):
        if self._assigned_value is not None:
            return iter([self._assigned_value])

        self._current_iter = 0
        return self

    def __next__(self):
        max_length = len(self._values)
        # this iteration makes it safe to prune current value as we iterate
        while self._current_iter < max_length and not self._visible_values[self._current_iter]:
            self._current_iter += 1

        if self._current_iter >= max_length:
            raise StopIteration

        self._current_iter += 1
        return self._values[self._current_iter - 1]


class Constraint(abc.ABC):
    """A constraint in a constraint satisfaction problem.

    Has 2 jobs:
    1. check feasibility
    2. prune domains
    """

    def __init__(self, variables: typing.List[Variable]):
        self._scope = variables

    def scope(self):
        return self._scope

    def unassigned(self):
        """List of index of unassigned variables in its scope"""
        unassigned = []
        for v in self._scope:
            if not v.is_assigned():
                unassigned.append(v)
        return unassigned

    @abc.abstractmethod
    def check_feasible(self):
        return True

    @abc.abstractmethod
    def prune_domains(self):
        """
        Prune the domain of the variables in this constraint's scope by exploiting the structure of the constraint.

        By default does forward checking.
        """
        return self.forward_check()

    def forward_check(self):
        pruned = []

        domain_wipeout = False
        # forward checking is only for if there's only one unassigned variable
        ua = self.unassigned()
        if len(ua) != 1:
            return domain_wipeout, pruned

        unassigned_variable = ua[0]

        # check all values of this variable
        for val in unassigned_variable:
            unassigned_variable.assign(val)
            if not self.check_feasible():
                unassigned_variable.prune(val)
                pruned.append((unassigned_variable, val))

        unassigned_variable.unassign()

        domain_wipeout = unassigned_variable.domain_size() == 0
        return domain_wipeout, pruned


class FunctionConstraint(Constraint):
    def __init__(self, variables: typing.List[Variable], call_able):
        super().__init__(variables)
        self._func = call_able

    def check_feasible(self):
        return self._func(*self._scope)


class CSP:
    """Class for packing up a set of variables and constraints into a CSP problem.

    Contains various utility routines for accessing the problem.
    The variables of the CSP can be added later or on initialization.
    The constraints must be added later
    """

    def __init__(self, variables):
        self._vars = []
        self._cons = []
        self._vars_to_cons = {}
        for v in variables:
            self.add_variable(v)

    def add_variable(self, variable):
        if not isinstance(variable, Variable):
            raise TypeError("Variable is not of type Variable")

        self._vars.append(variable)
        self._vars_to_cons[variable] = []

    def add_constraint(self, constraint):
        if not isinstance(constraint, Constraint):
            raise TypeError("Constraint is not of type Constraint")

        for v in constraint.scope():
            if v not in self._vars_to_cons:
                raise RuntimeError("Trying to add constraint with an unknown variable to the CSP object")
            self._vars_to_cons[v].append(constraint)

        self._cons.append(constraint)

    def constraints(self) -> typing.List[Constraint]:
        return self._cons

    def variables(self) -> typing.List[Variable]:
        return self._vars

    def constraints_involving_variable(self, variable) -> typing.List[Constraint]:
        return list(self._vars_to_cons[variable])
