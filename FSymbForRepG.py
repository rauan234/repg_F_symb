import sympy as sp
import itertools

from RepTheoryBasic import FiniteGroup, SubgroupRelation, LeftRep, triv_alg, left_rep_to_bimod, multi_range
from FSymbGeneral import IrredBimodsCompleteSet, Fork, FSymb, PentagonEquation


def multi_range(shape):
    return itertools.product(*[range(n) for n in shape])


# check that x is a tuple of objects of type t
def is_tuple_of(x, t):
    return isinstance(x, tuple) and all([isinstance(entry, t) for entry in x])


# a complete set of representatives of irreps of G, with fusion multiplicities and intertwiners computed optionally.
# the irreps are untwisted, i.e., the cocycle is trivial.
class RepG:
    def __init__(self, G, irreps):
        # group
        assert isinstance(G, FiniteGroup)
        self.G = G

        # irreps
        assert is_tuple_of(irreps, LeftRep)
        alg = irreps[0].alg
        assert alg.G == G
        assert alg.cocycle_trivial  # the group algebra must be untwisted
        assert all([rep.alg == alg for rep in irreps])
        self.irreps = irreps

        # put irreps into a IrredBimodsCompleteSet object
        triv_alg_instance = triv_alg()
        bimods = tuple([left_rep_to_bimod(rep, triv_alg_instance) for rep in irreps])
        self.irr = IrredBimodsCompleteSet(alg, triv_alg_instance, bimods)

        # characters
        self.chi = None
        self.__compute_chars()

        # compute the Hom spaces
        irr = self.irr
        self.fk = Fork(irr, irr, irr, "rep")

        # optionally, we may compute the F-symbols
        self.F = None

        # optionally, we may check that F satisfies the pentagon equation,
        # keeping self.pent_eq as a certificate
        self.pent_eq = None

    # compute the characters
    def __compute_chars(self):
        self.chi = [
            [sp.simplify(sp.Trace(rep.rho[g])) for g in range(self.G.n)]
            for rep in self.irreps
        ]

    def compute_F_symb(self):
        self.F = FSymb(self.fk, self.fk, self.fk, self.fk)

    def get_F(self, irr_ind, map_ind):
        return self.F.get(irr_ind, map_ind)

    # check that the F-symbols are unitary
    def check_F_unitary(self):
        assert self.F is not None

        self.F.check_unitary()

        print(f"Checked the unitarity of F for Rep(G) with |G| = {self.G.n}")

    # check that the F-symbols satisfy the pentagon equation
    def check_pentagon_eqs(self):
        assert self.F is not None

        # the F-symbol satisfies a pentagon equation with just itself
        F = self.F
        eqn = PentagonEquation(F, F, F, F, F)

        self.pent_eq = eqn

        print(f"Checked the pentagon equation for Rep(G) with |G| = {self.G.n}")

    # print the character table for the irreps
    def print_char_table(self):
        print("Characters of irreps")
        for char in self.chi:
            print(char)

    # print the fusion multiplicities coming from tensor products of irreps
    def print_N(self):
        print("Fusion multiplicities N")
        for (i, j) in multi_range((self.n_irr,) * 2):
            print((i, j), self.N[i][j])

    # print the F-symbols in a nice way
    def print_F_symb(self):
        assert self.F is not None

        print("F-symbols")
        self.F.print()


class RepSubgroupTwisted:
    def __init__(self, repg, subg_rel, alg, irreps):
        # reps of the parent group
        assert isinstance(repg, RepG)
        self.repg = repg

        # specify in what way alg.G is a subgroup of repg.G
        assert isinstance(subg_rel, SubgroupRelation)
        assert subg_rel.G == repg.G and subg_rel.H == alg.G
        self.subg_rel = subg_rel

        # a complete set of irreps
        assert is_tuple_of(irreps, LeftRep)
        assert all([rep.alg == alg for rep in irreps])
        self.alg = alg
        self.irreps = irreps

        # put irreps into a IrredBimodsCompleteSet object
        triv_alg_instance = triv_alg()
        bimods = tuple([left_rep_to_bimod(rep, triv_alg_instance) for rep in irreps])
        self.irr = IrredBimodsCompleteSet(alg, triv_alg_instance, bimods)

        # compute the Hom spaces for left and right module action of Rep(G) on Rep(H, psi)
        irr_G = self.repg.irr
        irr_H = self.irr
        self.fk_left = Fork(irr_G, irr_H, irr_H, ("res_l", subg_rel))   # left module action
        self.fk_right = Fork(irr_H, irr_G, irr_H, ("res_r", subg_rel))  # right module action

        # optionally, we may compute the F-symbols
        self.F_left = None
        self.F_right = None
        self.F_lr = None

    # Rep(G) on the midle and left, one Rep(H, psi) on the right.
    # then the strand types are
    # 123 GGH (output), 4 H (input), GH (intermediate).
    def compute_F_left(self):
        fk_tprod = self.repg.fk
        fk_res = self.fk_left
        # fk125 is GG
        # fk534 is HG
        # fk236 is HG
        # fk164 is HG
        self.F_left = FSymb(fk_tprod, fk_res, fk_res, fk_res)

    # Rep(G) on the right and middle, one Rep(H, psi) on the left.
    # then the strand types are
    # 123 HGG (output), 4 H (input), 56 HG (intermediate).
    def compute_F_right(self):
        fk_tprod = self.repg.fk
        fk_res = self.fk_right
        # fk125 is HG
        # fk534 is HG
        # fk236 is GG
        # fk164 is HG
        self.F_right = FSymb(fk_res, fk_res, fk_tprod, fk_res)

    # Rep(G) on the right and left, one Rep(H, psi) in the middle.
    # then the strand types are
    # 123 GHG (output), 4 H (input), 56 HH (intermediate).
    def compute_F_lr(self):
        fk_res_l = self.fk_left
        fk_res_r = self.fk_right
        # fk125 is GH
        # fk534 is HG
        # fk236 is HG
        # fk164 is GH
        self.F_lr = FSymb(fk_res_l, fk_res_r, fk_res_r, fk_res_l)

    def check_F_unitary(self):
        assert self.F_left is not None
        assert self.F_right is not None
        assert self.F_lr is not None

        self.F_left.check_unitary()
        self.F_right.check_unitary()
        self.F_lr.check_unitary()

        print(f"Checked unitarity of F_left, F_right, F_lr for Rep(H, psi) with |H| = {self.alg.G.n}")

    # check four pentagon equations that F_l, F_r, F_lr satisfy between themselves
    # and the parent Rep(G) F-symbols
    def check_pentagon_eqs(self):
        assert self.F_left is not None
        assert self.F_right is not None
        assert self.F_lr is not None
        assert self.repg.F is not None

        F_repg, F_l, F_r, F_lr = self.repg.F, self.F_left, self.F_right, self.F_lr

        # three Rep(G) acting from the right on one Rep(H, psi),
        # so that Rep(H, psi) is on output 0
        self.eq_0 = PentagonEquation(F_r, F_r, F_repg, F_r, F_r)  # right module and repg tprod

        # as above, but now Rep(H, psi) is on output number 1 instead of output number 0
        self.eq_1 = PentagonEquation(F_lr, F_r, F_r, F_lr, F_lr)  # right module and left-right bimodule

        # as above, but now Rep(H, psi)
        self.eq_2 = PentagonEquation(F_l, F_lr, F_lr, F_lr, F_l)  # left module and left-right bimodule

        # three Rep(G) acting from the left on one Rep(H, psi),
        # so that Rep(H, psi) is on output 3
        self.eq_3 = PentagonEquation(F_l, F_l, F_l, F_l, F_repg)  # left module and repg tprod

        print(f"Checked four pentagon equations for Rep(H, psi) with |H| = {self.alg.G.n}")

    # print the left module F-symbols in a nice way
    def print_F_left(self):
        assert self.F_left is not None

        print("Left module action F-symbols")
        self.F_left.print()

    # print the right module F-symbols in a nice way
    def print_F_right(self):
        assert self.F_right is not None

        print("Right module action F-symbols")
        self.F_right.print()

    # print the bimodule F-symbols in a nice way
    def print_F_lr(self):
        assert self.F_lr is not None

        print("Bimodule action F-symbols")
        self.F_lr.print()


class FunctorCat:
    def __init__(self, reph_dom, reph_codom):
        assert isinstance(reph_dom, RepSubgroupTwisted)
        assert isinstance(reph_codom, RepSubgroupTwisted)
        assert reph_dom.repg is reph_codom.repg  # must share the same Rep(G) object
        self.reph_dom = reph_dom
        self.reph_codom = reph_codom
        self.repg = reph_dom.repg