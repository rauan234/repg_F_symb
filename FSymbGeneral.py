import sympy as sp
import itertools

from RepTheoryBasic import SubgroupRelation, GpAlg, Bimod, HomSpace, left_rep_to_bimod


# a multidimensional list of a specified shape whose entries are all None
def multidim_list_of_none(shape):
    if shape:  # recursion
        return [multidim_list_of_none(shape[1:]) for _ in range(shape[0])]
    else:  # base case
        return None


def multi_range(shape):
    return itertools.product(*[range(n) for n in shape])


# check that x is a tuple of objects of type t
def is_tuple_of(x, t):
    return isinstance(x, tuple) and all([isinstance(entry, t) for entry in x])


# a collection of irreducible bimodules that exhaust all equivalence classes of such bimodules.
# NOTE: while testing for irreducibility and completeness in RepG is done using characters,
# the same approach does not seem to work for twisted reps. for this reason, we create this class.
class IrredBimodsCompleteSet:
    def __init__(self, alg_left, alg_right, bimods):
        assert isinstance(alg_left, GpAlg)
        assert isinstance(alg_right, GpAlg)
        self.alg_left = alg_left
        self.alg_right = alg_right

        assert is_tuple_of(bimods, Bimod)
        assert all([bm.left.alg == alg_left and bm.right.alg == alg_right for bm in bimods])
        self.bimods = bimods
        self.n_bm = len(bimods)

        self.__check_irred_and_complete()

    def __check_irred_and_complete(self):
        # test that the bimods are irreducible and non-isomorphic
        for (i, j) in multi_range((self.n_bm,) * 2):
            hom = HomSpace(self.bimods[i], self.bimods[j])
            hom.compute_onb()
            dim = hom.dim
            if i == j:
                assert dim == 1  # each bimod must be irreducible
            else:
                assert dim == 0  # the bimods must be non-isomorphic

        # test completeness using the sum of squares of dimensions criterion
        s = sum([bm.dim**2 for bm in self.bimods])
        assert s == self.alg_left.G.n * self.alg_right.G.n


class Fork:
    def __init__(self, irr_out_top, irr_out_bot, irr_in, prod_spec):
        assert isinstance(irr_in, IrredBimodsCompleteSet)
        assert isinstance(irr_out_top, IrredBimodsCompleteSet)
        assert isinstance(irr_out_bot, IrredBimodsCompleteSet)
        self.irr_in = irr_in
        self.irr_out_top = irr_out_top
        self.irr_out_bot = irr_out_bot
        self.irr_tuple = (self.irr_out_top, self.irr_out_bot, self.irr_in)

        # self.prod is a function.
        # NOTE: whether the product can be initialized as desired is checked not here but in __set_prod_to_res_left, etc.
        # moreover, oftentimes an invalid prod_spec would only raise errors in __compute_hom when computing hom spaces
        # because Bimods are not successfully initialized. thus, WARNING: successful init does not guarantee validity
        # of prod.
        self.prod_spec = prod_spec
        self.prod = lambda x: x
        if prod_spec == "rep":       # tensor product of bimodules defined similarly to representations
            self.__set_prod_to_rep_tprod()
        elif prod_spec == "module":  # tensor product of modules
            self.__set_prod_to_module_tprod()
        else:                        # Rep(G) acts by restriction on Rep(H, psi)
            assert isinstance(prod_spec, tuple) and len(prod_spec) == 2
            name, subg_rel = prod_spec

            if name == "res_l":    # Rep(H, psi) as a left module cat
                self.__set_prod_to_res_left(subg_rel)
            elif name == "res_r":  # Rep(H, psi) as a right module cat
                self.__set_prod_to_res_right(subg_rel)
            else:
                raise ValueError(f"Unknown prod_spec: {prod_spec}")

        # compute Hom spaces
        self.hom_list = multidim_list_of_none((self.irr_out_top.n_bm, self.irr_out_bot.n_bm, self.irr_in.n_bm))
        self.N = multidim_list_of_none((self.irr_out_top.n_bm, self.irr_out_bot.n_bm, self.irr_in.n_bm))
        self.N_max = -999
        self.__compute_hom()

    # tensor product of bimodules similar to tensor product of representations
    def __set_prod_to_rep_tprod(self):
        def prod(bm1, bm2, alg_left, alg_right):
            assert isinstance(bm1, Bimod) and isinstance(bm2, Bimod)
            # whether alg_left and alg_right are valid is checked in birep_tprod
            return bm1.birep_tprod(bm2, alg_left=alg_left, alg_right=alg_right)

        self.prod = prod

    # tensor product of modules
    def __set_prod_to_module_tprod(self):
        def prod(bm1, bm2, alg_left, alg_right):
            assert isinstance(bm1, Bimod) and isinstance(bm2, Bimod)
            # whether alg_left and alg_right are valid is checked in birep_tprod
            return bm1.module_tprod(bm2, alg_left=alg_left, alg_right=alg_right)

        self.prod = prod

    # Rep(H, psi) as a left module category over Rep(G)
    def __set_prod_to_res_left(self, subg_rel):
        # Rep(G) acts on the left; check that it's indeed Rep(G, phi) with phi trivial
        assert self.irr_out_top.alg_left.cocycle_trivial

        # check that the right action is trivial, so that we are dealing with representations
        # NOTE: if desired, can easily extend to nontrivial right actions.
        assert self.irr_out_top.alg_right.is_trivial()
        assert self.irr_out_bot.alg_right.is_trivial()
        assert self.irr_in.alg_right.is_trivial()

        assert isinstance(subg_rel, SubgroupRelation)

        def prod(bm1_big, bm2, alg_left, alg_right):
            assert isinstance(bm1_big, Bimod) and isinstance(bm2, Bimod)
            # validity of alg_left and alg_right, as well as subg_rel, is checked in tprod

            # restrict from G to H
            rep1 = bm1_big.left.restrict(subg_rel)

            # compute the tensor product as H-reps
            rep2 = bm2.left
            tprod = rep1.tprod(rep2, alg=alg_left)
            result = left_rep_to_bimod(tprod, alg_right=alg_right)

            return result

        self.prod = prod

    # Rep(H, psi) as a right module category over Rep(G)
    def __set_prod_to_res_right(self, subg_rel):
        # check that the right action is trivial, so that we are dealing with representations
        # NOTE: if desired, can easily extend to nontrivial right actions.
        assert self.irr_out_bot.alg_right.is_trivial()
        assert self.irr_in.alg_right.is_trivial()

        assert isinstance(subg_rel, SubgroupRelation)

        def prod(bm1, bm2_big, alg_left, alg_right):
            assert isinstance(bm1, Bimod) and isinstance(bm2_big, Bimod)
            # validity of alg_left and alg_right, as well as subg_rel, is checked in tprod

            # restrict from G to H
            rep2 = bm2_big.left.restrict(subg_rel)

            # compute the tensor product as H-reps
            rep1 = bm1.left
            tprod = rep1.tprod(rep2, alg=alg_left)
            result = left_rep_to_bimod(tprod, alg_right=alg_right)

            return result

        self.prod = prod

    def __compute_hom(self):
        for (i, j, k) in multi_range((self.irr_out_top.n_bm, self.irr_out_bot.n_bm, self.irr_in.n_bm)):
            # domain
            dom = self.irr_in.bimods[k]

            # codomain
            # NOTE: need it to have the same left/right algebras as dom (which is same as self.irr_in l/r algebras)
            bm1, bm2 = self.irr_out_top.bimods[i], self.irr_out_bot.bimods[j]
            codom = self.prod(bm1, bm2,
                              self.irr_in.alg_left, self.irr_in.alg_right)
            assert isinstance(codom, Bimod)
            assert codom.left.alg == dom.left.alg and codom.right.alg == dom.right.alg
            assert codom.dim_ambient == bm1.dim_ambient * bm2.dim_ambient  # if want, could relax this condition later

            # compute orthonormal basis for this hom space
            hom = HomSpace(dom, codom)
            hom.compute_onb()

            self.hom_list[i][j][k] = hom
            self.N[i][j][k] = hom.dim
            self.N_max = max(self.N_max, hom.dim)


class FSymb:
    def __init__(self, fk125, fk534, fk236, fk164):
        assert isinstance(fk125, Fork)
        assert isinstance(fk534, Fork)
        assert isinstance(fk236, Fork)
        assert isinstance(fk164, Fork)
        self.fk125 = fk125
        self.fk534 = fk534
        self.fk236 = fk236
        self.fk164 = fk164

        # irreducible bimods
        self.irr1 = fk125.irr_out_top  # out 1
        self.irr2 = fk236.irr_out_top  # out 2
        self.irr3 = fk236.irr_out_bot  # out 3
        self.irr4 = fk164.irr_in       # input
        self.irr5 = self.fk125.irr_in  # intermediate
        self.irr6 = self.fk236.irr_in  # intermediate
        self.irr_tuple = (self.irr1, self.irr2, self.irr3, self.irr4, self.irr5, self.irr6)

        # check that the forks are of the right format
        assert self.fk125.irr_tuple == (self.irr1, self.irr2, self.irr5)
        assert self.fk534.irr_tuple == (self.irr5, self.irr3, self.irr4)
        assert self.fk236.irr_tuple == (self.irr2, self.irr3, self.irr6)
        assert self.fk164.irr_tuple == (self.irr1, self.irr6, self.irr4)

        # compute the F-symbols
        self.trees_top_last = multidim_list_of_none(self.__shape_irreds(1, 2, 3, 4, 5))
        self.trees_bot_last = multidim_list_of_none(self.__shape_irreds(1, 2, 3, 4, 6))
        self.F = multidim_list_of_none(self.__shape_irreds(1, 2, 3, 4, 5, 6))
        self.__compute_trees_top_last()
        self.__compute_trees_bot_last()
        self.__compute_F_vals()

    # given a list of indexes 1, ..., 6, return self.irr_i.n_bm for each index i
    def __shape_irreds(self, *indexes):
        irr_tuple = (self.irr1, self.irr2, self.irr3, self.irr4, self.irr5, self.irr6)
        def aux(i):
            assert isinstance(i, int)
            assert 1 <= i <= 6
            return irr_tuple[i-1].n_bm

        return tuple([aux(i) for i in indexes])

    # compute splitting trees where the top splitting occurs closer to the end
    def __compute_trees_top_last(self):
        for (i, j, k, l, s) in multi_range(self.__shape_irreds(1, 2, 3, 4, 5)):
            # shape for a,b is variable
            shape_ab = (self.fk125.N[i][j][s], self.fk534.N[s][k][l])

            # store trees in a 2d list of the right shape
            self.trees_top_last[i][j][k][l][s] = multidim_list_of_none(shape_ab)
            for (a, b) in multi_range(shape_ab):
                # 4 splitting into 5, 3
                split534 = self.fk534.hom_list[s][k][l].onb[b]

                # 5 splitting into 1, 2; then tuck on identity on 3
                split125 = self.fk125.hom_list[i][j][s].onb[a]
                split125_extended = sp.kronecker_product(
                        split125,
                        sp.eye(self.irr3.bimods[k].dim_ambient)
                    )

                # combine
                tree = sp.simplify(split125_extended * split534)

                self.trees_top_last[i][j][k][l][s][a][b] = tree

    # compute splitting trees where the bottom splitting occurs closer to the end
    def __compute_trees_bot_last(self):
        for (i, j, k, l, t) in multi_range(self.__shape_irreds(1, 2, 3, 4, 6)):
            # shape for c, d is variable
            shape_cd = (self.fk236.N[j][k][t], self.fk164.N[i][t][l])

            # store trees in a 2d list of the right shape
            self.trees_bot_last[i][j][k][l][t] = multidim_list_of_none(shape_cd)
            for (c, d) in multi_range(shape_cd):
                # 4 splitting into 1, 6
                split164 = self.fk164.hom_list[i][t][l].onb[d]

                # 6 splitting into 2, 3; then tuck on identity on 1
                split236 = self.fk236.hom_list[j][k][t].onb[c]
                split236_extended = sp.kronecker_product(
                    sp.eye(self.irr1.bimods[i].dim_ambient),
                    split236
                )

                # combine
                tree = sp.simplify(split236_extended * split164)

                self.trees_bot_last[i][j][k][l][t][c][d] = tree

    def __compute_F_vals(self):
        for (i, j, k, l, s, t) in multi_range(self.__shape_irreds(1, 2, 3, 4, 5, 6)):
            # shape for a, b, c, d is variable
            shape_abcd = (self.fk125.N[i][j][s], self.fk534.N[s][k][l],
                          self.fk236.N[j][k][t], self.fk164.N[i][t][l])

            # store trees in a 4d list of the right shape
            self.F[i][j][k][l][s][t] = multidim_list_of_none(shape_abcd)
            for (a, b, c, d) in multi_range(shape_abcd):
                # TODO: fix "top first" and "bot first" being conflated
                tree_top_first = self.trees_top_last[i][j][k][l][s][a][b]
                tree_bot_first = self.trees_bot_last[i][j][k][l][t][c][d]

                # glue the trees along the leaves
                result = tree_bot_first.H * tree_top_first
                result = sp.simplify(result)

                # check that result is a scalar multiple of the identity
                scalar = result[0, 0]
                dim_ambient = self.irr4.bimods[l].dim_ambient
                assert (result - scalar * sp.eye(dim_ambient)).is_zero_matrix

                # this is (F^{ijk}_l)^{s, ab}_{t, cd}
                self.F[i][j][k][l][s][t][a][b][c][d] = scalar

    # print the F-symbols in a nice way
    def print(self):
        assert self.F is not None

        # enumerate splitting trees where splitting at the top occurs first.
        # here, the input and output irred bimod indexes ijkl are held fixed.
        def enumerate_top_first(i, j, k, l):
            out = []
            for s in range(self.irr5.n_bm):
                # shape for a, b is variable
                range_ab = (self.fk125.N[i][j][s], self.fk534.N[s][k][l])

                for (a, b) in multi_range(range_ab):
                    out.append((s, a, b))
            return out

        # enumerate splitting trees where splitting at the bottom occurs first.
        # here, the input and output irred bimod indexes ijkl are held fixed.
        def enumerate_bot_first(i, j, k, l):
            out = []
            for t in range(self.irr6.n_bm):
                # shape for c, d is variable
                range_cd = (self.fk236.N[j][k][t], self.fk164.N[i][t][l])

                for (c, d) in multi_range(range_cd):
                    out.append((t, c, d))
            return out

        def matrix_to_str(M):
            rows = [[str(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]
            # Pad each column to the widest entry in that column
            col_widths = [max(len(row[j]) for row in rows) for j in range(M.cols)]
            rows_str = []
            max_length = 0
            for row in rows:
                padded = "    ".join(entry.ljust(col_widths[j]) for j, entry in enumerate(row))
                row_str = f"[ {padded} ]"
                rows_str.append(row_str)
                max_length = max(max_length, len(row_str))

            return max_length, "\n".join(rows_str)

        for (i, j, k, l) in multi_range(self.__shape_irreds(1, 2, 3, 4)):
            bot_first = enumerate_bot_first(i, j, k, l)
            top_first = enumerate_top_first(i, j, k, l)

            n = len(bot_first)
            assert len(top_first) == n  # this should be true automatically, but check just in case

            if n > 0:  # if this splitting tree is allowed by fusion rules
                lst = [
                    [
                        self.get((i, j, k, l, s, t), (a, b, c, d))
                        for (t, c, d) in bot_first
                    ]
                    for (s, a, b) in top_first
                ]
                mat = sp.ImmutableMatrix(lst)

                max_length, mat_str = matrix_to_str(mat)
                print()
                print(f"F^({i}, {j}, {k})_{l}")
                print("-" * max_length)
                print(mat_str)
                print("-" * max_length)
                print("Top first (domain) basis:", top_first)
                print("Bottom first (codomain) basis:", bot_first)
                print()

    # return F value if indexes within range,
    # return 0 if a, b, c, d are out of range (if allowed)
    def get(self, irrep_ind, map_ind, allow_abcd_out_of_range=False):
        assert isinstance(irrep_ind, tuple) and len(irrep_ind) == 6
        assert isinstance(map_ind, tuple) and len(map_ind) == 4
        (i, j, k, l, s, t) = irrep_ind
        (a, b, c, d) = map_ind

        if allow_abcd_out_of_range:
            # enforce i, j, k, l, s, t being in range but allow a, b, c, d to be anything
            for (x, x_max) in zip(irrep_ind, self.shape_irreds):
                assert isinstance(x, int)
                assert 0 <= x < x_max

            # return 0 if a, b, c, d are out of range
            try:
                return self.F[i][j][k][l][s][t][a][b][c][d]
            except IndexError:
                return 0

        else:
            return self.F[i][j][k][l][s][t][a][b][c][d]

    # check that the F-symbols are unitary
    def check_unitary(self):
        for (i, j, k, l, s, spr) in multi_range(self.__shape_irreds(1, 2, 3, 4, 5, 5)):
            shape_abaprbpr = (
                self.fk125.hom_list[i][j][s].dim,
                self.fk534.hom_list[s][k][l].dim,
                self.fk125.hom_list[i][j][spr].dim,
                self.fk534.hom_list[spr][k][l].dim
            )
            for (a, b, apr, bpr) in multi_range(shape_abaprbpr):  # apr is a prime
                total = 0
                for t in range(self.irr6.n_bm):
                    temp = sum([
                        sp.conjugate(
                            self.get((i, j, k, l, spr, t), (apr, bpr, c, d))
                        )
                        * self.get((i, j, k, l, s, t), (a, b, c, d))
                        for (c, d) in multi_range((self.fk236.hom_list[j][k][t].dim,
                                                   self.fk164.hom_list[i][t][l].dim))], sp.sympify(0))
                    temp = sp.simplify(temp)
                    total += temp
                total = sp.simplify(total)

                if s != spr or a != apr or b != bpr:
                    assert total == 0
                else:
                    assert total == 1


class PentagonEquation:
    def __init__(self, F016, F423, F123, F053, F012):
        assert isinstance(F016, FSymb)
        assert isinstance(F423, FSymb)
        assert isinstance(F123, FSymb)
        assert isinstance(F053, FSymb)
        assert isinstance(F012, FSymb)

        # dict for individual F symbols
        self.F_NAMES = ("016", "423", "123", "053", "012")
        self.F_dict = {
            "016": F016,
            "423": F423,
            "123": F123,
            "053": F053,
            "012": F012
        }

        # important: this dict allows us to match the strands of F016, ..., F012 to strands of the pentagon equation.
        # the pent eq strands are labeled 0, ..., 9 as follows.
        #       0, 1, 2, 3 are the output strands, with 0 the top one and 3 the bottom one
        #       4 splits into 0, 1
        #       5 splits into 1, 2
        #       6 splits into 2, 3
        #       7 splits into 0, 5 or 2, 4 (so that it encompasses 0, 1, 2)
        #       8 splits into 5, 3 or 1, 6 (so that it encompasses 1, 2, 3)
        #       9 is the input; it splits into 0, 8 or 4, 6 or 7, 3, encompassing 0, 1, 2, 3
        # the F strands are labeled 0, ..., 5 as follows.
        #       0, 1, 2 are the output strands, with 0 the top one and 3 the bottom one
        #       4 splits into 0, 1; and 5 splits into 2, 3
        #       5 is the input strand
        self.F_SUPERIND = {
            "016": (0, 1, 6, 9, 4, 8),
            "423": (4, 2, 3, 9, 7, 6),
            "123": (1, 2, 3, 8, 5, 6),
            "053": (0, 5, 3, 9, 7, 8),
            "012": (0, 1, 2, 7, 4, 5)
        }

        # use self.F_SUPERIND to match the strands of F016, ..., F012 to pent eq strands,
        # checking along the way that the different F strands are compatible
        self.irr_list = [None] * 10  # the pent eq strands are numbered 0, ..., 9
        self.__assign_irr_list()

        # use self.F_SUPERIND to match the forks of F016, ..., F012 to pent eq forks,
        # checking along the way that the different F forks are compatible
        self.FORK_NAMES = ("014", "057", "089", "125", "168", "236", "427", "469", "538", "739")
        self.fk_dict = dict()
        self.__assign_fk_dict()

        # verify that the pentagon equation holds
        self.__check_pent_eq()

    # use self.F_SUPERIND to assign the value of self.irr_lust by matching pent eq strands to F016, ..., F012 strands
    def __assign_irr_list(self):
        for name in self.F_NAMES:
            F = self.F_dict[name]
            superind = self.F_SUPERIND[name]

            # match the irr from F to self.irr_list using self.F_SUPERIND
            for x in range(10):
                if x in superind:
                    # if self.F_SUPERIND says that F_name has something to say about the x-th strand

                    # use self.F_SUPERIND to convert index x = 0, ..., 9 of pent eq strands
                    # into an index 0, ..., 5 for the six strands of F
                    i = superind.index(x)
                    if self.irr_list[x] is None:
                        # if not given a value yet, assign
                        self.irr_list[x] = F.irr_tuple[i]
                    else:
                        # if a value already exists, check compatibility
                        if self.irr_list[x] != F.irr_tuple[i]:
                            raise ValueError("F" + name + "has unexpected irr", i + 1)

    # use self.F_SUPERIND to assign the value of self.fk_dict by matching pent eq forks to F016, ..., F012 forks
    def __assign_fk_dict(self):
        for name in self.F_NAMES:
            F = self.F_dict[name]
            superind = self.F_SUPERIND[name]

            # match the forks from F to self.fk_dict using self.F_SUPERIND
            for (fk_F, fk_spec_F) in zip(
                    (F.fk125, F.fk534, F.fk236, F.fk164),
                    ((0, 1, 4), (4, 2, 3), (1, 2, 5), (0, 5, 3))):  # 125, 534, 236, 164 in 0...5 format
                fk_spec_translated = "".join([str(superind[i]) for i in fk_spec_F])

                # if translating the fork spec from F to self does give a valid fork spec
                if fk_spec_translated in self.FORK_NAMES:
                    if fk_spec_F in self.fk_dict:
                        # must match existing value
                        if self.fk_dict[fk_spec_translated] != fk_F:
                            fk_spec_str = "".join([str(i) for i in fk_spec_F])
                            raise ValueError("F" + name + "has unexpected fork " + fk_spec_str)
                    else:
                        # if the fork spec is not already given a value, assign it
                        self.fk_dict[fk_spec_translated] = fk_F

    # given a list of indexes 0, ..., 9, return self.irr_i.n_bm for each index i
    def __shape_irreds(self, *indexes):
        def aux(i):
            assert isinstance(i, int)
            assert 0 <= i <= 9
            return self.irr_list[i].n_bm

        return tuple([aux(i) for i in indexes])

    def get_fk(self, x, y, z):
        return self.fk_dict[str(x) + str(y) + str(z)]

    # enumerate splitting trees of five different shapes.
    # here, the input strand irred bimod index is held fixed as m,
    # and such indexes on the outputs are fixed as i, j, k, l.
    # at the same time, intermediate irred bimod indexes as well as the indexes of splitting maps are allowed to vary.
    def __enumerate_spl_trees(self, i, j, k, l, m):
        tree_specs = (
            (((2, 3, 6), (1, 6, 8), (0, 8, 9)), ("23", "1u", "0v"), "bottom"),
            (((0, 1, 4), (2, 3, 6), (4, 6, 9)), ("01", "23", "uv"), "simultaneous"),
            (((0, 1, 4), (4, 2, 7), (7, 3, 9)), ("01", "u2", "v3"), "top"),
            (((1, 2, 5), (5, 3, 8), (0, 8, 9)), ("12", "u3", "0v"), "bottom, then top"),
            (((1, 2, 5), (0, 5, 7), (7, 3, 9)), ("12", "0u", "v3"), "top, then bottom")
        )
        # each element (order, xy_spec, name) of tree_specs is as follows.
        # order is a tuple ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3)) specifying the order of splitting for a tree.
        #   here (x1, y1, z1) are the (out_top, out_bot, input) superindexes of the fork that occurs the last,
        #   and (x3, y3, z3) likewise specify the first fork. each x is from 0 to 9 inclusive; these vars label
        #   strands of the pentagon equation (see comments above self.F_SUPERIND in init).
        #   for the simultaneous splitting tree, which fork to call last and which to call penultimate is chosen
        #   arbitrarily because the forks don't share any branches.
        #   WARNING: the convention for the order of (a, b, c) vs. (b, a, c) for "simultaneous"
        #   is assumed in __get_F_ext_val.
        # xy_spec is a tuple (xy1_spec, ..., xy3_spec).
        #   the first character of xy_i_spec tells whether x_i is (a) one of the input strands 0123
        #   or (b) one of "u", "v". here, "u" is the superindex of the input strand of the last fork,
        #   while "v" refers to the input strand of the middle fork.
        # name is simply a string that specifies how a given list of trees should be called in the output dict.

        trees = dict()
        for (order, xy_spec, name) in tree_specs:
            (x1, y1, z1) = order[0]
            (x2, y2, z2) = order[1]
            (x3, y3, z3) = order[2]
            tree_list = []
            for (u, v) in multi_range(self.__shape_irreds(z1, z2)):
                i_dict = {"0": i, "1": j, "2": k, "3": l, "u": u, "v": v}
                bds = [
                    (i_dict[temp[0]], i_dict[temp[1]])
                    for temp in xy_spec
                ]
                shape_abc = (
                    self.get_fk(x1, y1, z1).hom_list[bds[0][0]][bds[0][1]][u].dim,
                    self.get_fk(x2, y2, z2).hom_list[bds[1][0]][bds[1][1]][v].dim,
                    self.get_fk(x3, y3, z3).hom_list[bds[2][0]][bds[2][1]][m].dim
                )
                for (a, b, c) in multi_range(shape_abc):
                    tree_list.append((u, v, a, b, c))

            # superindexes of the strands with indexes u, v, m
            uv_superind = [z1, z2, 9]

            trees[name] = (uv_superind, tree_list)

        return trees

    def __get_F_ind_from_indexes_inp(self, name, indexes_inp,
                                     uv_super_inp, abc_inp):
        F_super = self.F_SUPERIND[name]

        # indexes i, j, k, l, t of F have pent eq strands with superindexes
        # number 0, 1, 2, 3, 5, respectively, from self.F_SUPERIND
        F_strand_ids = (0, 1, 2, 3, 5)
        ijklt = [indexes_inp[F_super[x]] for x in F_strand_ids]

        # superindexes 0, ..., 9 of strands that are inputs of the forks to which maps with indexes c_F, d_F attach
        F_cd_super = [F_super[5], F_super[3]]

        # uv_super_inp.index(i) tells us which of (ua, vb, mc) reproduces the desired superindex
        temp = [uv_super_inp.index(i) for i in F_cd_super]
        cd = [abc_inp[i] for i in temp]

        return ijklt, cd

    def __get_F_ind_from_indexes_out(self, name, indexes_out,
                                     uv_super_out, abc_out):
        F_super = self.F_SUPERIND[name]

        # indexes i, j, k, l, s of F have pent eq strands with superindexes
        # number 0, 1, 2, 3, 4, respectively, from self.F_SUPERIND
        F_strand_ids = (0, 1, 2, 3, 4)
        ijkls = [indexes_out[F_super[x]] for x in F_strand_ids]

        # superindexes 0, ..., 9 of strands that are inputs of the forks to which maps with indexes a_F, b_F attach
        F_ab_super = [F_super[4], F_super[3]]

        # uv_super_out.index(i) tells us which of (ua, vb, mc) reproduces the desired superindex
        temp = [uv_super_out.index(i) for i in F_ab_super]
        ab = [abc_out[i] for i in temp]

        return ijkls, ab

    # extend F016, ..., F012 from three output strands (their original defn)
    # to act on the four output strands of the pentagon equation.
    # use self.F_SUPERIND to so this.
    def __get_F_ext_val(self, name, ind_dom, ind_codom,
                        uv_super, uv_super_pr, ijklm):
        # WARNING: this data uses a convention for what "abc" and "uv" refers to from __enumerate_spl_trees
        F_FIXED_DATA = {
            "016": ((0, 0), (1, 1)),  # (u, a) on domain must match (v, b) on codomain
            "423": ((0, 0), (0, 0)),  # (u, a) matches (u, a)
            "123": ((1, 2), (1, 2)),  # (v, c) matches (v, c)
            "053": ((0, 0), (0, 0)),  # (u, a) matches (u, a)
            "012": ((1, 2), (1, 2))  # (v, c) matches (v, c)
        }

        # unpack indexes of trees
        (u, v), (a, b, c) = ind_dom
        (upr, vpr), (apr, bpr, cpr) = ind_codom

        # unpack indexes of irred bimods at the input and output
        (i, j, k, l, m) = ijklm

        (p1, q1), (p2, q2) = F_FIXED_DATA[name]
        if (u, v)[p1] == (upr, vpr)[p2] and (a, b, c)[q1] == (apr, bpr, cpr)[q2]:
            # the input splitting tree has one input with superindex 9 and index m,
            # three output with superinedexes 0, 1, 2, 3 and indexes i, j, k, l,
            # an intermediate strand with index u,
            # and an intermediate strand with index v. the superindexes of the intermediate strands
            # depend on the particular splitting tree as in question.
            # put this data into a dict {superindex: index}. not all superindexes 0, ..., 9 are represented.
            # that's because the splitting tree in question does not contain all superindexes.
            # the input splitting tree is bottom first.
            indexes_inp = {
                0: i, 1: j, 2: k, 3: l, 9: m,
                uv_super[0]: u, uv_super[1]: v
            }
            # extract indexes for F using the input tree data
            (i_F_inp, j_F_inp, k_F_inp, l_F_inp, t_F_inp), (c_F, d_F) = self.__get_F_ind_from_indexes_inp(name,
                                                                                                          indexes_inp,
                                                                                                          uv_super,
                                                                                                          (a, b, c))

            # same as above but the output splitting tree is top first.
            indexes_out = {
                0: i, 1: j, 2: k, 3: l, 9: m,
                uv_super_pr[0]: upr, uv_super_pr[1]: vpr
            }
            # extract indexes for F using the output tree data
            (i_F_out, j_F_out, k_F_out, l_F_out, s_F), (a_F, b_F) = self.__get_F_ind_from_indexes_out(name,
                                                                                                      indexes_out,
                                                                                                      uv_super_pr,
                                                                                                      (apr, bpr, cpr))

            # the ijkl indexes obtained in two different ways must agree. this should automatically be true because
            # the input and output splitting trees should have matching input/output indexes; check just in case.
            assert (i_F_out, j_F_out, k_F_out, l_F_out) == (i_F_inp, j_F_inp, k_F_inp, l_F_inp)

            ijklst_F = (i_F_inp, j_F_inp, k_F_inp, l_F_inp, s_F, t_F_inp)
            abcd_F = (a_F, b_F, c_F, d_F)
            return self.F_dict[name].get(ijklst_F, abcd_F)
        else:  # mismatch of indexes that are supposed to be equal
            return 0

    def __compute_F_extended(self, i, j, k, l, m):
        F_extended_data = (
            ("016", "bottom", "simultaneous"),
            ("423", "simultaneous", "top"),
            ("123", "bottom", "bottom, then top"),
            ("053", "bottom, then top", "top, then bottom"),
            ("012", "top, then bottom", "top")
        )
        trees = self.__enumerate_spl_trees(i, j, k, l, m)

        F_ext_dict = {}
        for (name, dom, codom) in F_extended_data:
            abc_super, tree_list = trees[dom]
            abc_super_pr, tree_list_pr = trees[codom]

            lst = [
                [
                    self.__get_F_ext_val(name, ((u, v), (a, b, c)), ((upr, vpr), (apr, bpr, cpr)),
                                         abc_super, abc_super_pr, (i, j, k, l, m))
                    for (u, v, a, b, c) in tree_list  # domain
                ]
                for (upr, vpr, apr, bpr, cpr) in tree_list_pr  # codomain
            ]
            F_ext = sp.ImmutableMatrix(lst)

            F_ext_dict[name] = F_ext

        return F_ext_dict

    def __check_pent_eq(self):
        for (i, j, k, l, m) in multi_range(self.__shape_irreds(0, 1, 2, 3, 9)):
            # dict whose entries specify F016, ..., F012 extended to a matrix that acts
            # on splitting trees with 4 outputs instead of 3 outputs
            F_extended = self.__compute_F_extended(i, j, k, l, m)

            # first path option for the pentagon equation
            path1 = sp.simplify(F_extended["423"] * F_extended["016"])

            # second path option for the pentagon equation
            path2 = sp.simplify(sp.simplify(F_extended["012"] * F_extended["053"]) * F_extended["123"])

            diff = sp.simplify(path1 - path2)
            if not diff.is_zero_matrix:
                for (name, F_mat) in F_extended.items():
                    # check if, perhaps, the error is already visible from F_mat not being unitary
                    if not sp.simplify(F_mat.H * F_mat - sp.eye(F_mat.rows)).is_zero_matrix:
                        raise RuntimeError(f"Extended F{name} with ijklm = {(i, j, k, l, m)} is not unitary, {F_mat}")

                    print("F" + name)
                    print(F_mat)
                    print()
                raise RuntimeError(f"Pentagon equation failed with ijklm = {(i, j, k, l, m)}, diff = {diff}")
