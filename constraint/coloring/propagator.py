import abc
from csp_base import CSP


class Propagator(abc.ABC):
    """A constraint propagator that prunes domains"""

    @abc.abstractmethod
    def __call__(self, csp: CSP, newly_instantiated_variable=None):
        """Prunes csp's variables and returns (domain wipeout, [(pruned variable, pruned value), ...])"""


class ForwardCheck(Propagator):
    def __call__(self, csp: CSP, newly_instantiated_variable=None):
        total_pruned = []
        for con in csp.constraints():
            domain_wipeout, pruned = con.forward_check()
            total_pruned.extend(pruned)

            if domain_wipeout:
                return domain_wipeout, total_pruned

        return False, total_pruned


class GeneralArcConsistency(Propagator):
    def __call__(self, csp: CSP, newly_instantiated_variable=None):
        # TODO implement
        ...
