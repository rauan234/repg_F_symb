import sympy as sp
import itertools

from FSymbForRepG import RepG, RepSubgroupTwisted, FunctorCat, FSymb
from FSymbForRepG import Fork
from FSymbGeneral import is_tuple_of, IrredBimodsCompleteSet


def multi_range(shape):
    return itertools.product(*[range(n) for n in shape])


def list_duplicate_free(lst):
    return len(set(lst)) == len(lst)


# a multidimensional list of a specified shape whose entries are all None
def multidim_list_of_none(shape):
    if shape:  # recursion
        return [multidim_list_of_none(shape[1:]) for _ in range(shape[0])]
    else:  # base case
        return None


# a multidimensional list of a specified shape whose entries are different copies of the empty list
def multidim_list_base_empty(shape):
    if shape:  # recursion
        return [multidim_list_of_none(shape[1:]) for _ in range(shape[0])]
    else:  # base case
        return []


class GrayPatch:
    def __init__(self, fk, direction):
        assert isinstance(fk, Fork)
        self.fk = fk

        # there are four types of gray patches. decide which one we are dealing with.
        self.permutation = (None, None, None)
        self.assign_permutation(direction)

        # use self.permutation to determine what kind (Rep(G), functor, module) are the
        # left, middle, right strands (representing irreducible bimodules) of the gray patch
        self.irr_left, self.irr_mid, self.irr_right = None, None, None
        self.assign_irrs()

        # use self.permuation to make an array N such that N[i][j][k] is how high the index inside the gray patch
        # is allowed to go up to when the irred bimod indexes at the left, middle, right strands are i, j, k
        self.N = None
        self.assign_N()

    # gray patches can come from (1) Rep(G) right module action, or (2) functor action.
    # for each action type, the middle strand can come with either an incoming or an outgoing arrow.
    # each of these four gray patch types represents a fork of a corresponding type.
    # this correspondence is captured by self.permutation, which is a tuple
    # self.permutation = (pos_leftmost, pos_middle, pos_rightmost). here,
    # pos_leftmost \in (0, 1, 2) matches the leftmost gray patch strand to one of the fork strands.
    # value 0 means top output, 1 means bottom output, 2 means input. likewise for pos_middle and pos_rightmost.
    def assign_permutation(self, direction):
        self.direction = direction

        if self.fk.prod_spec[0] == "res_r":  # Rep(G) acts by restriction from the right
            # case 1: Rep(H, psi) as a right Rep(G)-module cat
            self.action_type = "repg"

            if direction == "incoming":
                # a Rep(G)-object arrow Y comes into the gray patch.
                # the strand labels on the gray patch are (M1, Y, M2), representing a fork (M1 < Y, M2).
                # (see the last case below for an explanation.)
                self.permutation = (0, 1, 2)

            elif direction == "outgoing":
                # a Rep(G)-object arrow Y comes out from the gray patch.
                # the strand labels on the gray patch are (M2, Y, M1), representing a fork (M1 < Y, M2).
                self.permutation = (2, 1, 0)

        elif self.fk.prod_spec == "module":  # functors are implemented using the module tensor product
            # case 2: a functor acting on Rep(H, psi)
            self.action_type = "functor"

            if direction == "incoming":
                # a functor arrow X comes into the gray patch.
                # the strand labels on the gray patch are (M, X, N), representing a fork
                # with outputs (X, M) and input N.
                self.permutation = (1, 0, 2)

            elif direction == "outgoing":
                # a functor arrow X comes outwards from the gray patch.
                # the strand labels on the gray patch are (N, X, M), representing a fork with
                # outputs (X, M) and input N. the fork strands are always listed
                # in the order (out_top, out_bot, inp), which in this case is (X, M, N).
                # since the gray patch strand labels are (N, X, M), the permutation
                # s.t. permutation[gray_patch_label] == fork_label is (1, 0, 2). for example, permutation[0] == 2
                # means that the zeroth (leftmost) strand of the grat patch, which in this case is N,
                # is the second (input) strand of the fork.
                self.permutation = (2, 0, 1)

            else:
                raise ValueError("Invalid direction")

        else:
            raise ValueError(f"Invalid gray patch fork type: {self.fk.prod_spec}")

    # what types (Rep(G), functor, module) are the left, middle, right strands of the gray patch
    def assign_irrs(self):
        # use self.permutation to reorder irrs_fork according to which gray patch strand
        # corresponds to which fork strand
        irrs_fork = (
            self.fk.irr_out_top,  # irreducible bimodules on the top output strand of the fork
            self.fk.irr_out_bot,
            self.fk.irr_in
        )
        (self.irr_left, self.irr_mid, self.irr_right) = [irrs_fork[self.permutation[a]] for a in range(3)]

    # the multiplicity to which the gray patch index is allowed to go
    def assign_N(self):
        shape = (self.irr_left.n_bm, self.irr_mid.n_bm, self.irr_right.n_bm)
        self.N = multidim_list_of_none(shape)

        # i, j, k index irreducible bimodules at the top output, bottom output, and input of the fork, respectively
        for (i, j, k) in multi_range(shape):
            i_new, j_new, k_new = None, None, None
            for a in range(2):
                # i_new is the index of the irreducible bimodule which is at the leftmost strand of the gray patch,
                # j_new is for the middle strand, and k_new is for the rightmost strand
                (i_new, j_new, k_new) = [(i, j, k)[self.permutation[a]] for a in range(3)]

            # the multiplicity is the same as for the fork which this gray patch represents.
            # in assign_N, we simply match the order in which i, j, k are specified for
            # forks and for gray patches.
            N_ijk = self.fk.N[i][j][k]
            self.N[i_new][j_new][k_new] = N_ijk

    # decide whether this gray patch can be contracted with other
    def can_be_contr_with(self, other):
        assert isinstance(other, GrayPatch)

        # directions must be opposite
        if not self.direction != other.direction:
            return False

        # the Fork objects must be the same. this ensures that
        # (1) the middle strands match, (2) the left/right strands match,
        # (3) the particular realization of the product operation (such as the orthonormal basis chosen for Hom) match.
        if not self.fk == other.fk:
            return False

        # if the directions are opposite and the forks match, then the patches are compatible
        return True


class CircleOfPatches:
    def __init__(self, patches):
        assert isinstance(patches, list)
        assert all([isinstance(pt, GrayPatch) for pt in patches])
        self.patches = patches
        self.L = len(patches)

        for x in range(self.L):
            assert patches[(x + 1) % self.L].irr_left == patches[x].irr_right
        self.shape_M = [pt.irr_right.n_bm for pt in patches]

        self.shape_XY = [pt.irr_mid.n_bm for pt in patches]

    # check whether what's supposed to be M_idx, XY_idx, or a_idx is valid
    def check_idx_valid(self, lst, bound):
        assert isinstance(lst, list) or isinstance(lst, tuple)
        assert len(lst) == self.L
        assert all([isinstance(i, int) and 0 <= i < i_max for (i, i_max) in zip(lst, bound)])

    def get_a_max(self, M_idx, XY_idx, x, skip_checks=False):
        if not skip_checks:
            self.check_idx_valid(M_idx, self.shape_M)
            self.check_idx_valid(XY_idx, self.shape_XY)
        assert isinstance(x, int)
        assert 0 <= x < self.L

        pt = self.patches[x]              # patch
        m_left = M_idx[(x - 1) % self.L]  # index on the module strand to the left
        xy = XY_idx[x]                    # index on the middle strand
        m_right = M_idx[x]                # index on the module strand to the right

        a_max = pt.N[m_left][xy][m_right]
        return a_max

    def get_shape_a(self, M_idx, XY_idx, sites=None):
        self.check_idx_valid(M_idx, self.shape_M)
        self.check_idx_valid(XY_idx, self.shape_XY)
        
        if sites is None:
            sites = range(0, self.L)

        shape_a = [self.get_a_max(M_idx, XY_idx, x, skip_checks=True) for x in sites]
        return shape_a


class GrayPatchBlob:
    def __init__(self, skeleton, names, vals):
        # information about patches
        assert isinstance(skeleton, CircleOfPatches)
        self.skeleton = skeleton
        self.L = skeleton.L  # number of patches

        # names[i] is the name of the i-th patch
        assert isinstance(names, list)
        assert all([isinstance(name, str) for name in names])
        assert len(names) == self.L
        assert list_duplicate_free(names)  # each tentacle must have a unique name
        self.names = names

        # what numerical value is assigned to every allowed combination of indexes
        assert isinstance(vals, dict)
        for (M_idx, XY_idx) in itertools.product(multi_range(skeleton.shape_M),
                                                 multi_range(skeleton.shape_XY)):
            for a_idx in multi_range(skeleton.get_shape_a(M_idx, XY_idx)):
                key = (M_idx, XY_idx, a_idx)
                if key not in vals:
                    raise RuntimeError(f"Key {key} not in vals")
        self.vals = vals
    
    def __get(self, M_idx, XY_idx, a_idx, skip_checks=False):
        if not skip_checks:
            # the validity of M_idx and XY_idx is checked inside get_shape_a
            shape_a = self.skeleton.get_shape_a(M_idx, XY_idx)
            self.skeleton.check_idx_valid(a_idx, shape_a)

        # M_idx, etc., are allowed to be list; convert to tuple
        key = (tuple(M_idx), tuple(XY_idx), tuple(a_idx))

        return self.vals[key]

    def contract(self, other, contr_names, names_self_to_new, names_other_to_new):
        assert isinstance(other, GrayPatchBlob)
        assert isinstance(names_self_to_new, dict)
        assert isinstance(names_other_to_new, dict)

        # convert the verbal contraction spec into a collection of three indexes,
        # checking validity along the way. here, k is the number of tentacles being contracted,
        # and i_start, j_start are such that tentacle i_start + s of self is contracted with j_start - s of other
        # for all 0 <= s < k.
        i_start, j_start, k = self.__contraction_verbal_spec_to_indexes(other, contr_names)

        # from i_start, etc., compute lists that directly state which tentacles and which module strands
        # of self, other, and new are identified/contracted. (see __get_idents_and_contr for explanation.)
        idents_XY, idents_M, contr_XY, contr_M = self.__get_idents_and_contr(other, i_start, j_start, k)
        
        # patches and names of the new blob. see __get_idents_and_contr for format of idents_XY.
        patches_new = []
        names_new = []
        for (owner, x) in idents_XY:  # the x_new-th element of idents_XY corresponds to x_new-th tentacle of new
            if owner == 0:  # self
                patches_new.append(self.skeleton.patches[x])
                names_new.append(names_self_to_new[self.names[x]])
            else:  # owner == 1; other
                patches_new.append(other.skeleton.patches[x])
                names_new.append(names_other_to_new[other.names[x]])
        
        # put patches into CircleOfPatches format
        skeleton_new = CircleOfPatches(patches_new)
        
        # fill in the new values
        vals_new = dict()
        for M_idx, XY_idx in itertools.product(multi_range(skeleton_new.shape_M),
                                               multi_range(skeleton_new.shape_XY)):
            shape_a = skeleton_new.get_shape_a(M_idx, XY_idx)
            for a_idx in multi_range(shape_a):
                contr_spec = (idents_XY, idents_M, contr_XY, contr_M)
                idx_new = (M_idx, XY_idx, a_idx)
                val = self.__get_val_contracted(other, contr_spec, idx_new)
                key = (M_idx, XY_idx, a_idx)
                vals_new[key] = val
        
        blob_new = GrayPatchBlob(skeleton_new, names_new, vals_new)
        return blob_new
    
    def __get_val_contracted(self, other, contr_spec, idx_new):
        idents_XY, idents_M, contr_XY, contr_M = contr_spec
        M_idx_new, XY_idx_new, a_idx_new = idx_new

        M_idx_self_other = [[None,] * self.L, [None,] * other.L]
        XY_idx_self_other = [[None,] * self.L, [None,] * other.L]
        a_idx_self_other = [[None,] * self.L, [None,] * other.L]
        
        # fill in non-contracted values from the new blob
        for (x_new, idents) in enumerate(idents_M):
            for (owner, x) in idents:
                M_idx_self_other[owner][x] = M_idx_new[x_new]
        for (x_new, (owner, x)) in enumerate(idents_XY):
            XY_idx_self_other[owner][x] = XY_idx_new[x_new]
            a_idx_self_other[owner][x] = a_idx_new[x_new]
        
        shape_M_sum = [self.skeleton.patches[x_self].irr_right.n_bm for (x_self, _) in contr_M]
        shape_XY_sum = [self.skeleton.patches[x_self].irr_mid.n_bm for (x_self, _) in contr_XY]
        
        result = sp.sympify(0)
        for XY_idx_summed in multi_range(shape_XY_sum):
            # fill in contracted XY indexes
            for (xy_new, (x_self, x_other)) in zip(XY_idx_summed, contr_XY):
                XY_idx_self_other[1][x_other] = XY_idx_self_other[0][x_self] = xy_new
            
            # by this point, XY indexes of self and other should be completely filled with numeric values
            # instead of None
            XY_idx_self, XY_idx_other = XY_idx_self_other
            
            for M_idx_summed in multi_range(shape_M_sum):
                # fill in contracted XY indexes
                for (m_new, (x_self, x_other)) in zip(M_idx_summed, contr_M):
                    M_idx_self_other[0][x_self] = M_idx_self_other[1][x_other] = m_new
                
                M_idx_self, M_idx_other = M_idx_self_other
                
                # having filled in M_idx and XY_idx, determine the range for a
                x_self_list = contr_XY[:][0]
                shape_a = self.skeleton.get_shape_a(M_idx_self, XY_idx_self, x_self_list)
                
                for a_idx_summed in multi_range(shape_a):
                    # fill in contracted a indexes
                    for (a_new, (x_self, x_other)) in zip(a_idx_summed, contr_XY):
                        a_idx_self_other[0][x_self] = a_idx_self_other[1][x_other] = a_new
                    
                    a_idx_self, a_idx_other = a_idx_self_other
                    
                    val_self = self.__get(M_idx_self, XY_idx_self, a_idx_self)
                    val_other = other.__get(M_idx_other, XY_idx_other, a_idx_other)
                    result += sp.simplify(val_self * val_other)
        
        result = sp.simplify(result)
        return result

    def __get_idents_and_contr(self, other, i_start, j_start, k):
        # contracting tentacle i_start + s of self with j_start - s of other for 0 <= s < k
        contr_XY = [((i_start + s) % self.L, (j_start - s) % other.L) for s in range(k)]

        # contracting module strand i_start + s of self with j_start - 1 - s of other for 0 <= s < k - 1.
        # NOTE: there are only k-1 contractions of module strands instead of k.
        assert k >= 1
        contr_M = [((i_start + s) % self.L, (j_start - 1 - s) % other.L) for s in range(k - 1)]

        # list with one element (owner, x) for each tentacle site 0 <= x_new < L of the new blob.
        # here owner == 0 means that x-th tentacle of self becomes the x_new-th tentacle of new,
        # and owner == 1 means that the x-th tentacle of other becomes the x_new-th tentacle of new.
        L_new = self.L + other.L - 2 * k
        idents_XY = [(0, (i_start + k + s) % self.L) for s in range(self.L - k)]  # 0 is for self
        idents_XY += [(1, (j_start + 1 + s) % other.L) for s in range(other.L - k)]  # 1 for other
        assert len(idents_XY) == L_new

        # list with one element [..., (owner, x), ...] for each module strand site 0 <= x_new < L of the new blob.
        # here the structure with (owner, x) is as for idents_XY but now each x_new can have multiple identifications.
        idents_M = [[] for _ in range(L_new)]
        for (x_new, (owner, x)) in enumerate(idents_XY):
            idents_M[x_new].append((owner, x))

            # if the x-th tentacle of self is identified with the x_new-th tentacle of new,
            # then also the (x-1)-th module strand of self is identified with the (x_new-1)-th module strand of new.
            L = (self.L, other.L)[owner]
            x_minus_1 = (x - 1) % L
            x_new_minus_1 = (x_new - 1) % L_new
            idents_M[x_new_minus_1].append((owner, x_minus_1))

        return idents_XY, idents_M, contr_XY, contr_M

    def patch_from_name(self, name):
        assert isinstance(name, str)

        try:
            i = self.names.index(name)
        except ValueError:
            raise ValueError(f"Tentacle name {name} not found in {self.names}")

        return self.skeleton.patches[i]

    def __contraction_verbal_spec_to_indexes(self, other, contr_names):
        # list [..., (name_self, name_other), ...] such that tentacle name_self of self
        # is contracted with tentacle name_other of other
        assert isinstance(contr_names, list)
        for pair in contr_names:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            name_self, name_other = pair
            assert name_self in self.names
            assert name_other in other.names

            # check that the patches have compatible types of module strands, middle strands, and arrow directions
            pt_self, pt_other = self.patch_from_name(pair[0]), other.patch_from_name(pair[1])
            if not pt_self.can_be_contr_with(pt_other):
                raise RuntimeError(f"Contracting {name_self} with {name_other} is not possible")

        # indexes of tentacles being contracted
        contr_ind = [
            (self.names.index(name_self), other.names.index(name_other))
            for (name_self, name_other) in contr_names
        ]
        i_list = [ij[0] for ij in contr_ind]

        # number of tentacles being contracted
        k = len(contr_names)
        assert k >= 1  # must contract at least one tentacle

        # set(i_list) should be of the form {i_start, i_start + 1 % self.L, ..., i_start + k - 1 % self.L}.
        # NOTE: i_start is not necessarily just min(i_list), for example if i_start == self.L - 2 and k = 5.
        i_start = None
        for i in i_list:
            if (i - 1) % self.L not in i_list:
                i_start = i
                break
        if i_start is None:  # contracting all tentacles of self
            assert k == self.L
            i_start = 0

        # sort according to increasing i. then contr_ind_sorted should be of the form
        # [(i_start, j_start), (i_start + 1 % self.n, j_start - 1 % other.n),
        # ..., (i_start + k - 1 % self.n, j_start - (k - 1) % self.n)].
        contr_ind_sorted = sorted(contr_ind, key=lambda x: x[0])
        j_start = contr_ind_sorted[0][1]
        for (s, (i, j)) in enumerate(contr_ind_sorted):
            if i != (i_start + s) % self.L:
                raise ValueError(f"From {self.names}, elements {contr_names[:][0]} are not contiguous")

            if j != (j_start - s) % other.L:
                raise ValueError(f"Pairs of indexes {contr_ind_sorted} are not oppositely contiguous")

        return i_start, j_start, k

    def get_val(self, ind_spec, module_strand_mismatch_allowed_and_gives_zero=True):
        assert isinstance(ind_spec, tuple) or isinstance(ind_spec, list)
        assert len(ind_spec) == self.L  # must specify every tentacle

        M_idx = [None,] * self.L
        XY_idx = [None,] * self.L
        a_idx = [None,] * self.L
        names_specified = set()
        for item in ind_spec:
            assert isinstance(item, tuple) and len(item) == 5
            name, m_left, xy, m_right, a = item

            assert isinstance(name, str)
            try:
                i = self.names.index(name)
            except ValueError:
                raise ValueError(f"Tentacle name {name} not found in {self.names}")
            assert name not in names_specified  # a tentacle can only be specified once
            names_specified.add(name)

            assert isinstance(m_left, int)
            assert isinstance(xy, int)
            assert isinstance(m_right, int)
            assert isinstance(a, int)
            patch = self.skeleton.patches[i]
            assert 0 <= m_left < patch.irr_left.n_bm
            assert 0 <= xy < patch.irr_mid.n_bm
            assert 0 <= m_right < patch.irr_right.n_bm
            assert 0 <= a < patch.N[m_left][xy][m_right]

            i_minus_1 = (i - 1) % self.L

            # if module strands of adjacent patches do not match, the value is zero
            if not ((M_idx[i_minus_1] is None) or (m_left == M_idx[i_minus_1])):
                if module_strand_mismatch_allowed_and_gives_zero:
                    return 0
                else:
                    raise ValueError(f"Mismatch of left module strand for element {name}")
            if not ((M_idx[i] is None) or (m_right == M_idx[i])):
                if module_strand_mismatch_allowed_and_gives_zero:
                    return 0
                else:
                    raise ValueError(f"Mismatch of right module strand for element {name}")

            M_idx[i_minus_1] = m_left
            XY_idx[i], a_idx[i] = xy, a
            M_idx[i] = m_right

        val = self.__get(M_idx, XY_idx, a_idx)
        return val


def module_pants(reph, Y1, Y2, Y3, l):
    assert isinstance(reph, RepSubgroupTwisted)
    assert reph.F_right is not None
    
    names = ["input", "out left", "out right"]
    patches = [
        GrayPatch(reph.fk_right, "outgoing"),
        GrayPatch(reph.fk_right, "incoming"),
        GrayPatch(reph.fk_right, "incoming")
    ]
    
    vals = dict()
    for (M1, M2, M3) in multi_range((reph.irr.n_bm,) * 3):
        for (Y1_dummy, Y2_dummy, Y3_dummy) in multi_range((reph.repg.irr.n_bm,) * 3):
            shape_ijk = (
                patches[1].N[M1][Y1_dummy][M3],
                patches[2].N[M3][Y2_dummy][M2],
                patches[0].N[M2][Y3_dummy][M1]
            )
            for (i, j, k) in multi_range(shape_ijk):
                key = ((M1, M3, M2), (Y3_dummy, Y1_dummy, Y2_dummy), (k, i, j))
                if Y1_dummy == Y1 and Y2_dummy == Y2 and Y3_dummy == Y3:
                    val = reph.F_right.get((M1, Y1, Y2, M2, M3, Y3), (i, j, l, k))
                else:
                    val = 0
                vals[key] = val
    
    blob = GrayPatchBlob(CircleOfPatches(patches), names, vals)
    return blob


def module_pants_flipped(reph, Y1, Y2, Y3, l):
    assert isinstance(reph, RepSubgroupTwisted)
    assert reph.F_right is not None

    names = ["output", "inp right", "inp left"]
    patches = [
        GrayPatch(reph.fk_right, "incoming"),
        GrayPatch(reph.fk_right, "outgoing"),
        GrayPatch(reph.fk_right, "outgoing")
    ]

    vals = dict()
    for (M1, M2, M3) in multi_range((reph.irr.n_bm,) * 3):
        for (Y1_dummy, Y2_dummy, Y3_dummy) in multi_range((reph.repg.irr.n_bm,) * 3):
            shape_ijk = (
                patches[2].N[M3][Y1_dummy][M1],
                patches[1].N[M2][Y2_dummy][M3],
                patches[0].N[M1][Y3_dummy][M2]
            )
            for (i, j, k) in multi_range(shape_ijk):
                key = ((M2, M3, M1), (Y3_dummy, Y2_dummy, Y1_dummy), (k, j, i))
                if Y1_dummy == Y1 and Y2_dummy == Y2 and Y3_dummy == Y3:
                    val = reph.F_right.get((M1, Y1, Y2, M2, M3, Y3), (i, j, l, k)).conjugate()
                else:
                    val = 0
                vals[key] = val

    blob = GrayPatchBlob(CircleOfPatches(patches), names, vals)
    return blob


def module_nn_int_blob(reph, Y1, Y2, Y3, Y4, Y5, l1, l2):
    pants_left = module_pants(reph, Y3, Y5, Y1, l1)
    pants_right_flipped = module_pants_flipped(reph, Y5, Y2, Y4, l2)

    contractions = [("out right", "inp left")]
    names_left_to_new = {"input": "inp left", "out left": "out left"}
    names_right_to_new = {"inp right": "inp right", "output": "out right"}

    interaction = pants_left.contract(pants_right_flipped, contractions, names_left_to_new, names_right_to_new)

    return interaction


def module_nn_int_mat(reph, Y1, Y2, Y3, Y4, Y5, l1, l2):
    blob = module_nn_int_blob(reph, Y1, Y2, Y3, Y4, Y5, l1, l2)

    patch_left = blob.patch_from_name("out left")
    patch_right = blob.patch_from_name("out right")

    basis = []
    for (M1, M2, M3) in multi_range((reph.irr.n_bm,) * 3):
        for (Y1, Y2) in multi_range((reph.repg.irr.n_bm,) * 2):
            shape_ij = (
                patch_left.N[M1][Y1][M2],
                patch_right.N[M2][Y2][M3]
            )
            for (i, j) in multi_range(shape_ij):
                basis.append((M1, M2, M3, Y1, Y2, i, j))

    d = len(basis)
    lst = [[None,] * d for _ in range(d)]
    for ((a, inp_spec), (b, out_spec)) in itertools.product(enumerate(basis), repeat=2):
        M1_inp, M2_inp, M3_inp, Y1_inp, Y2_inp, i_inp, j_inp = inp_spec
        M1_out, M2_out, M3_out, Y1_out, Y2_out, i_out, j_out = out_spec

        val = blob.get_val((
            ("inp left", M2_inp, Y1_inp, M1_inp, i_inp),
            ("inp right", M3_inp, Y2_inp, M2_inp, j_inp),
            ("out left", M1_out, Y1_out, M2_out, i_out),
            ("out right", M2_out, Y2_out, M3_out, j_out)
        ))

        lst[a][b] = val

    mat = sp.ImmutableMatrix(lst)
    return mat
