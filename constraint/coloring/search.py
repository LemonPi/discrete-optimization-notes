from csp_base import CSP
from csp_base import Variable
import propagator
import logging
import typing
import math

logger = logging.getLogger(__name__)


def get_min_domain_variable(csp):
    # by default use the fail-first principle and return the variable with the smallest domain
    min_domain = math.inf
    var = None
    for v in csp.variables():
        if not v.is_assigned() and v.domain_size() < min_domain:
            min_domain = v.domain_size()
            var = v

    return var


class BacktrackSearch:
    def __init__(self, csp: CSP, prop=propagator.ForwardCheck(),
                 select_next_variable: typing.Callable[[CSP], Variable] = get_min_domain_variable):
        self._csp = csp
        self._propagator = prop
        self._get_next_var = select_next_variable
        self._solution = None

    def restore_all_domains(self):
        for v in self._csp.variables():
            if v.is_assigned():
                v.unassign()
            v.restore_domain()

    def restore_pruning(self, prunings: typing.List[typing.Tuple[Variable, typing.Any]]):
        for var, val in prunings:
            # TODO consider if we can store value indices instead since that'll be an order of magnitude improvement
            var.prune(val, unprune=True)

    def search(self):
        """Get first solution to CSP. Returns None if no solution, otherwise the solution."""

        self.restore_all_domains()

        # initial propagation
        domain_wipeout, prunings = self._propagator(self._csp)
        # infeasible
        if domain_wipeout:
            return None

        domain_wipeout = self.recurse()

        self.restore_pruning(prunings)

        if domain_wipeout:
            return None
        return self._solution

    def recurse(self):
        """Return whether we had a domain wipeout"""

        var = self._get_next_var(self._csp)

        # done since no more variables to assign
        if var is None:
            # reached a solution
            self._solution = [v.assigned_value() for v in self._csp.variables()]
            return False

        for val in var:
            var.assign(val)
            domain_wipeout, prunings = self._propagator(self._csp, var)

            if not domain_wipeout:
                return self.recurse()

            # else this choice of val for var caused a domain wipeout so we need to restore and try something else
            self.restore_pruning(prunings)
            var.unassign()

        # if there was no choice for any var to not have domain wipeout, our previous choices forced us to wipeout
        return True
