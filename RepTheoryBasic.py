import sympy as sp
import itertools


def multi_range(shape):
    return itertools.product(*[range(n) for n in shape])


class FiniteGroup:
    def __init__(self, mult):
        # multiplication table
        assert isinstance(mult, list)
        self.mult = mult
        self.n = len(mult)

        # check that the multiplication table is valid
        self.__check_multiplicativity()

        # inv[i] is such that mult[i][inv[i]] == 0
        self.inv = None
        self.__compute_inverses()

    # check that the multiplication table is valid
    def __check_multiplicativity(self):
        m = self.mult

        # check that each row of the multiplication table
        # is from 0 to n-1 and does not repeat indexes
        # (in particular, this implies that inverses exist)
        for row in m:
            assert set(row) == set(range(self.n))

        # check that the zeroth element is the identity
        for i in range(self.n):
            assert m[i][0] == m[0][i] == i

        # check that multiplication is associative
        for (i, j, k) in multi_range((self.n,) * 3):
            ij = m[i][j]
            jk = m[j][k]
            assert m[ij][k] == m[i][jk]

    def __compute_inverses(self):
        self.inv = [None] * self.n

        for i in range(self.n):
            success = False
            for j in range(self.n):
                if self.mult[i][j] == 0:
                    self.inv[i] = j
                    success = True
                    assert self.mult[j][i] == 0  # just in case, check that j * i is also trivial
                    break
            if not success:
                # should always succeed if __compute_F_extended passes, but check again just in case
                raise RuntimeError

        self.inv = tuple(self.inv)


# represent that H is a subgroup of G via an injection incl: H \to G
class SubgroupRelation:
    def __init__(self, G, H, incl):
        assert isinstance(G, FiniteGroup)
        assert isinstance(H, FiniteGroup)
        self.G = G
        self.H = H

        assert isinstance(incl, tuple) and len(incl) == H.n
        assert len(set(incl)) == H.n  # function must be injective
        self.incl = incl
        self.__check_incl_is_homomorphism()

    # check that the inclusion map is a group homomorphism
    def __check_incl_is_homomorphism(self):
        for (i, j) in multi_range((self.H.n, self.H.n)):
            ij = self.H.mult[i][j]
            phi_i, phi_j = self.incl[i], self.incl[j]
            assert self.incl[ij] == self.G.mult[phi_i][phi_j]


# view a group as a subgroup of itself
def subgroup_of_itself(G):
    assert isinstance(G, FiniteGroup)

    # the inclusion map is the identity map
    incl = tuple(range(G.n))

    return SubgroupRelation(G, G, incl)


# twisted group algebra
class GpAlg:
    def __init__(self, G, psi, claim_normalized=False, claim_untwisted=False):
        assert isinstance(G, FiniteGroup)
        self.G = G

        # cocycle: array of complex numbers of magnitude 1,
        # one number for each pair of elements of G
        assert isinstance(psi, tuple) and len(psi) == G.n
        for temp in psi:
            assert isinstance(temp, tuple) and len(temp) == G.n
            for x in temp:
                assert isinstance(x, sp.Basic)  # x is some kind of sympy thing
                assert x.is_complex and sp.Abs(x) == 1
        self.psi = psi

        self.__check_cocycle_cond()

        # whether psi(g, g^{-1}) = 1 for all g; this can be done by multiplying by a coboundary
        if claim_normalized:
            self.__check_normalization()  # assert that psi is indeed normalized
            self.cocycle_normalized = True
        else:
            self.cocycle_normalized = False

        # whether psi is trivial
        if claim_untwisted:
            self.claim_cocycle_trivial()
        else:
            self.cocycle_trivial = False

    def __check_cocycle_cond(self):
        for (i, j, k) in multi_range((self.G.n,) * 3):
            ij = self.G.mult[i][j]
            jk = self.G.mult[j][k]
            x = self.psi[i][j] * self.psi[ij][k]
            y = self.psi[i][jk] * self.psi[j][k]
            assert sp.simplify(x) == sp.simplify(y)

    # check that G is trivial, and that therefore self is a trivial group algebra
    def is_trivial(self):
        return self.G.n == 1

    # creates a new group algebra whose cocycle is given by psi(g2^{-1}, g1^{-1})
    def inv(self):
        psi_inv = tuple([tuple([self.psi[self.G.inv[j]][self.G.inv[i]]
                                for j in range(self.G.n)]) for i in range(self.G.n)])
        return GpAlg(self.G, psi_inv, claim_normalized=self.cocycle_normalized)

    # creates a NEW object with a different psi
    def copy_with_normalized_cocycle(self):
        sqrt_choices = {}
        for g in range(self.G.n):
            if g not in sqrt_choices:
                g_inv = self.G.inv[g]
                l = sp.simplify(sp.sqrt(self.psi[g][g_inv]))
                sqrt_choices[g] = sqrt_choices[g_inv] = l

        def psi_new_entry(g1, g2):
            return sp.simplify(self.psi[g1][g2]
                               * sqrt_choices[self.G.mult[g1][g2]] / sqrt_choices[g1] / sqrt_choices[g2])

        psi_new = tuple([tuple([
            psi_new_entry(g1, g2)
            for g2 in range(self.G.n)]) for g1 in range(self.G.n)])

        # make a new GpAlg object,
        # and also return how psi was altered
        lam = [sqrt_choices[g] for g in range(self.G.n)]  # put sqrt_choices into a list called lambda
        return GpAlg(self.G, psi_new, claim_normalized=True), lam

    # check that a certain optional without-loss-of-generality condition is satisfied
    def __check_normalization(self):
        for i in range(self.G.n):
            assert sp.simplify(self.psi[i][self.G.inv[i]]) == 1

    # check that the cocycle is trivial
    def __check_untwisted(self):
        for row in self.psi:
            for x in row:
                assert x == 1

    # check that the cocycle is trivial and flip self.cocycle_trivial to True
    def claim_cocycle_trivial(self):
        self.__check_untwisted()
        self.cocycle_trivial = True

    # given another group algebra of the same group, create a new GpAlg whose cocycle is the product
    def multiply_cocycle(self, other):
        assert isinstance(other, GpAlg)

        # the groups have to match
        assert self.G == other.G
        G = self.G

        # product of cocycles
        psi = tuple([
            tuple([
                sp.simplify(self.psi[g][h] * other.psi[g][h])
                for h in range(G.n)])
            for g in range(G.n)])

        return GpAlg(G, psi,
                     claim_normalized=self.cocycle_normalized and other.cocycle_normalized,
                     claim_untwisted=self.cocycle_trivial and other.cocycle_trivial)

    # restrict the cocycle to a subgroup
    def restrict(self, subg_rel):
        assert isinstance(subg_rel, SubgroupRelation)
        assert subg_rel.G == self.G  # the parent group of subg_rel has to be self.G
        H = subg_rel.H

        # use the inclusion map
        psi = tuple([tuple([
            self.psi[subg_rel.incl[i]][subg_rel.incl[j]]
            for j in range(H.n)
        ])
            for i in range(H.n)])

        return GpAlg(H, psi,
                     claim_normalized=self.cocycle_normalized,
                     claim_untwisted=self.cocycle_trivial)


# psi-twisted representation of G;
# equivalently, a left \C_\psi[G] module
class LeftRep:
    def __init__(self, alg, deg, rho):
        # group algebra (contains a group G and a cocycle psi)
        assert isinstance(alg, GpAlg)
        self.alg = alg

        # degree of the representation
        assert isinstance(deg, int) and deg >= 1
        self.deg = deg

        # map from G to Hom(V)
        assert isinstance(rho, tuple)
        assert len(rho) == self.alg.G.n  # rho(g) for every g in G
        for x in rho:
            assert isinstance(x, sp.ImmutableMatrix)
            assert x.shape == (deg, deg)
            assert all(entry.is_complex is True for entry in x)
            assert sp.Q.unitary(x)  # require that the rep is unitary
        self.rho = rho

        # test that rho is a psi-twisted rep of G
        self.__test_multiplicativity()

    def eval(self, i):
        return self.rho[i]

    # rho inverse
    def eval_inv(self, i):
        return self.rho[self.alg.G.inv[i]]

    def is_untwisted(self):
        return self.alg.cocycle_trivial

    # right rep given by rho(g^{-1})
    def inverse_right_rep(self):
        rho_inv = tuple([self.eval_inv(i) for i in range(self.alg.G.n)])
        return RightRep(self.alg.inv(), self.deg, rho_inv)

    def __test_multiplicativity(self):
        for (i, j) in multi_range((self.alg.G.n,) * 2):
            x = self.rho[i] * self.rho[j]
            k = self.alg.G.mult[i][j]
            y = self.alg.psi[i][j] * self.rho[k]  # include a phase from the cocycle
            if not sp.simplify(x - y).is_zero_matrix:
                print(sp.simplify(x))
                print(sp.simplify(y))
                raise RuntimeError(f"Candidate rep of degree {self.deg} of G with |G| = {self.alg.G.n}"
                                   f" fails multiplicativity with i, j = {i}, {j} and x, y = {x}, {y}")

    # make a copy of this rep but modifying rho by a U(1) second cohomology coboundary
    def copy_with_coboundary_change(self, alg_new, lam):
        assert isinstance(alg_new, GpAlg)
        assert isinstance(lam, list)

        # divide rho[g] by lam[g] to get a new representation
        rho_new = tuple([x / l for (l, x) in zip(lam, self.rho)])

        # check that the new representation does indeed match alg_new
        return LeftRep(alg_new, self.deg, rho_new)

    # restrict to a subgroup
    def restrict(self, subg_rel, alg=None):
        assert isinstance(subg_rel, SubgroupRelation)
        assert subg_rel.G == self.alg.G  # the parent group of subg_rel has to be self.alg.G

        if alg is None:
            alg = self.alg.restrict(subg_rel)
        else:
            assert isinstance(alg, GpAlg)
            assert subg_rel.H == alg.G

        # use inclusion map
        rho = tuple([self.rho[subg_rel.incl[h]] for h in range(subg_rel.H.n)])

        return LeftRep(alg, self.deg, rho)

    # tensor m copies of the trivial rep on the right
    def extend_right(self, m, alg=None):
        assert isinstance(m, int)
        assert m >= 1

        # group algebra
        if isinstance(alg, type(None)):
            alg = self.alg
        else:
            assert isinstance(alg, GpAlg)

        rho = tuple([
            sp.kronecker_product(self.rho[g], sp.eye(m)) for g in range(self.alg.G.n)
        ])

        return LeftRep(alg, self.deg * m, rho)

    # tensor product self \otimes other.
    # NOTE: here, we are taking alg as an optional input. this is because the "same" algebra data (group and cocycle)
    # can be realized by GpAlg objects with different pointers, and we might want to have a GpAlg object with a
    # desired pointer.
    def tprod(self, other, alg=None):
        assert isinstance(other, LeftRep)

        # the groups have to match (although the cocycles can be different)
        assert self.alg.G == other.alg.G
        G = self.alg.G

        # tensor product of actions
        rho = tuple([
            sp.kronecker_product(self.rho[g], other.rho[g]) for g in range(G.n)
        ])

        # group algebra
        if isinstance(alg, type(None)):
            # the group algebra of a tensor product of reps is always the one computed by multiply_cocycle.
            # if the user did not provide a GpAlg instance, we create a new one with this data.
            alg = self.alg.multiply_cocycle(other.alg)
        else:
            assert isinstance(alg, GpAlg)

        return LeftRep(alg, self.deg * other.deg, rho)


# a right \C_\psi[G] module
class RightRep:
    def __init__(self, alg, deg, rho):
        # group algebra (contains a group G and a cocycle psi)
        assert isinstance(alg, GpAlg)
        self.alg = alg

        # degree of the representation
        assert isinstance(deg, int) and deg >= 1
        self.deg = deg

        # map from G to Hom(V)
        assert isinstance(rho, tuple)
        assert len(rho) == self.alg.G.n  # rho(g) for every g in G
        for x in rho:
            assert isinstance(x, sp.ImmutableMatrix)
            assert x.shape == (deg, deg)
            assert all(entry.is_complex is True for entry in x)
            assert sp.Q.unitary(x)  # require that the rep is unitary
        self.rho = rho

        # test that rho is a psi-twisted rep of G
        self.__test_multiplicativity()

    def eval(self, i):
        return self.rho[i]

    # rho inverse
    def eval_inv(self, i):
        return self.rho[self.alg.G.inv[i]]

    # left rep given by rho(g^{-1})
    def inverse_left_rep(self):
        rho_inv = tuple([self.eval_inv(i) for i in range(self.alg.G.n)])
        return LeftRep(self.alg.inv(), self.deg, rho_inv)

    def __test_multiplicativity(self):
        for (i, j) in multi_range((self.alg.G.n,) * 2):
            x = self.rho[j] * self.rho[i]  # note: i and j are exchanged compared to left rep
            k = self.alg.G.mult[i][j]
            y = self.alg.psi[i][j] * self.rho[k]  # include a phase from the cocycle
            assert sp.simplify(x) == sp.simplify(y)

    def copy_with_coboundary_change(self, alg_new, lam):
        assert isinstance(alg_new, GpAlg)
        assert isinstance(lam, list)

        # divide rho[g] by lam[g] to get a new representation
        rho_new = tuple([x / l for (l, x) in zip(lam, self.rho)])

        # check that the new representation does indeed match alg_new
        return RightRep(alg_new, self.deg, rho_new)

    # tensor m copies of the trivial rep on the left
    def extend_left(self, m, alg=None):
        assert isinstance(m, int)
        assert m >= 1

        # group algebra
        if isinstance(alg, type(None)):
            alg = self.alg
        else:
            assert isinstance(alg, GpAlg)

        rho = tuple([
            sp.kronecker_product(sp.eye(m), self.rho[g]) for g in range(self.alg.G.n)
        ])

        return RightRep(alg, self.deg * m, rho)

    # tensor product self \otimes other.
    # NOTE: here, we are taking alg as an optional input. this is because the "same" algebra data (group and cocycle)
    # can be realized by GpAlg objects with different pointers, and we might want to have a GpAlg object with a
    # desired pointer.
    def tprod(self, other, alg=None):
        assert isinstance(other, RightRep)

        # the groups have to match (although the cocycles can be different)
        assert self.alg.G == other.alg.G
        G = self.alg.G

        # tensor product of actions
        rho = tuple([
            sp.kronecker_product(self.rho[g], other.rho[g]) for g in range(G.n)
        ])

        # group algebra
        if isinstance(alg, type(None)):
            # the group algebra of a tensor product of reps is always the one computed by multiply_cocycle.
            # if the user did not provide a GpAlg instance, we create a new one with this data.
            alg = self.alg.multiply_cocycle(other.alg)
        else:
            assert isinstance(alg, GpAlg)

        return RightRep(alg, self.deg * other.deg, rho)


# commuting left and right representations on the SAME vector space (but possibly of different group algebras).
# the vector space is the subspace of C^n which is the image of a projector P.
class Bimod:
    def __init__(self, left, right, P):
        assert isinstance(left, LeftRep) and isinstance(right, RightRep)
        assert left.alg.cocycle_normalized  # impose this WLOG condition for later use in module_tprod
        assert right.alg.cocycle_normalized  # impose this WLOG condition for later use in module_tprod
        self.left = left
        self.right = right

        # dimension of the ambient vector space
        assert left.deg == right.deg
        self.dim_ambient = left.deg

        # the projector whose image specifies the bimodule as a vector space
        assert isinstance(P, sp.ImmutableMatrix)
        assert P.shape == (self.dim_ambient, self.dim_ambient)
        assert sp.simplify(P * P - P).is_zero_matrix  # must be a projector
        self.P = P
        self.dim = sp.simplify(sp.Trace(P))
        assert self.dim.is_integer

        self.__test_reps_commute()

    # test that the left and right actions commute with each other and with P
    def __test_reps_commute(self):
        def commute(x, y):
            return sp.simplify(x * y - y * x).is_zero_matrix

        for (g, h) in multi_range((self.left.alg.G.n, self.right.alg.G.n)):
            x = self.left.rho[g]
            y = self.right.rho[h]
            assert commute(x, y)

        for g in range(self.left.alg.G.n):
            assert commute(self.left.rho[g], self.P)

        for h in range(self.right.alg.G.n):
            assert commute(self.right.rho[h], self.P)

    # if self and other are both bimodules over groups (G, H), return the bimodule
    # built similarly to the tensor product of representations.
    # NOTE: while the groups acting on the left and right must match, the cocycles can differ.
    def birep_tprod(self, other, alg_left=None, alg_right=None):
        assert isinstance(other, Bimod)

        # left action is the tprod of reps
        # (the groups have to match for this but the cocycles don't)
        left = self.left.tprod(other.left, alg=alg_left)

        # right action is the tprod of reps
        # (the groups have to match for this but the cocycles don't)
        right = self.right.tprod(other.right, alg=alg_right)

        # projection onto the subspace
        P = sp.kronecker_product(self.P, other.P)

        return Bimod(left, right, P)

    # if self is an (A, B)-bimod and other is a (B, C)-bimod, compute self \otimes_B other,
    # the tensor product of self and other as bimodules.
    # NOTE: alg_left is an optional input for the same reason as in birep_tprod above.
    def module_tprod(self, other, alg_left=None, alg_right=None):
        assert isinstance(other, Bimod)

        # the self.left and other.right group algebras have to match
        assert self.right.alg == other.left.alg

        # left action inherited from self
        # (it's just tensored with the identity map to act on the appropriate tensor product)
        left = self.left.extend_right(other.dim_ambient, alg=alg_left)

        # right action inherited from other
        right = other.right.extend_left(self.dim_ambient, alg=alg_right)

        # the projector now includes a factor from "contracting" self.right and other.left actions
        dim_ambient = self.dim_ambient * other.dim_ambient
        G = self.right.alg.G
        P = sum([sp.kronecker_product(self.right.eval_inv(g), other.left.rho[g])
                 for g in range(G.n)], sp.zeros(dim_ambient, dim_ambient)) / G.n
        P *= sp.kronecker_product(self.P, other.P)
        P = sp.simplify(P)

        return Bimod(left, right, P)


class HomSpace:
    def __init__(self, dom, codom):
        assert isinstance(dom, Bimod) and isinstance(codom, Bimod)
        assert dom.left.alg == codom.left.alg
        assert dom.right.alg == codom.right.alg
        self.dom = dom
        self.codom = codom
        self.alg_left = dom.left.alg
        self.alg_right = dom.right.alg
        self.dim_ambient_total = self.dom.dim_ambient * self.codom.dim_ambient

        # compute a matrix that will later be used in the "project" method
        # to project an arbitrary linear map onto the Hom space
        self.proj = None
        self.dim = None
        self.__compute_proj()

        # orthonormal basis (if/when computed)
        self.onb = None

    # compute a matrix that will be used in the "project" method.
    # the matrix encodes a "super-operator" that maps linear maps to other linear maps.
    # the calculation involving the kronecker product comes from the identity
    # vec(A X B) == kron(A, B^T) vec(X),
    # where vec(X) = X.reshape(X.shape[0] * X.shape[1], 1) flattens X into a column vector.
    def __compute_proj(self):
        # projector coming from restricting an arbitrary map to the subspaces specified by P
        P_subsp = sp.kronecker_product(self.codom.P, self.dom.P.T)

        # projector coming from the left action
        P_left = sum([sp.kronecker_product(
            self.codom.left.eval_inv(g),
            self.dom.left.rho[g].T
        )
            for g in range(self.alg_left.G.n)],
            sp.zeros(self.dim_ambient_total, self.dim_ambient_total)) / self.alg_left.G.n
        P_left = sp.simplify(P_left)

        # projector coming from the right action
        P_right = sum([sp.kronecker_product(
            self.codom.right.eval_inv(g),
            self.dom.right.rho[g].T
        )
            for g in range(self.alg_right.G.n)],
            sp.zeros(self.dim_ambient_total, self.dim_ambient_total)) / self.alg_right.G.n
        P_right = sp.simplify(P_right)

        # product of individual projectors
        self.proj = sp.simplify(P_subsp * P_left * P_right)

        # the dimension of the Hom space is given by the degree of the projector
        self.dim = sp.simplify(sp.Trace(self.proj))
        try:
            assert self.dim.is_integer
        except AssertionError:
            raise RuntimeError("Unable to compute the dimension of the Hom space; getting dim =", self.dim)

    # given an arbitrary linear map from C^{dom.dim_ambient} to C^{codom.dim_ambient}, project it onto the Hom space
    def project(self, mat):
        assert isinstance(mat, sp.ImmutableMatrix)
        assert mat.shape == (self.codom.dim_ambient, self.dom.dim_ambient)

        # act with the projection matrix, reshaping between matrices / vectors as necessary
        vec = mat.reshape(self.dim_ambient_total, 1)
        vec = sp.simplify(self.proj * vec)
        mat_new = vec.reshape(self.codom.dim_ambient, self.dom.dim_ambient)
        return mat_new

    # given an arbitrary linear map from C^{dom.dim_ambient} to C^{codom.dim_ambient}, determine if it is equivariant
    def test_membership(self, mat):
        assert isinstance(mat, sp.ImmutableMatrix)
        assert mat.shape == (self.codom.dim_ambient, self.dom.dim_ambient)

        # whether mat maps between the subspaces which are the domain and codomain
        P, Q = self.dom.P, self.codom.P
        if not sp.simplify(Q * mat * P - mat).is_zero_matrix:
            return False

        # whether mat commutes with the left action
        for g in range(self.alg_left.G.n):
            x = mat * self.dom.left.rho[g]
            y = self.codom.left.rho[g] * mat
            if not sp.simplify(x - y).is_zero_matrix:
                return False

        # whether mat commutes with the right action
        for g in range(self.alg_right.G.n):
            x = mat * self.dom.right.rho[g]
            y = self.codom.right.rho[g] * mat
            if not sp.simplify(x - y).is_zero_matrix:
                return False

        return True

    # compute an orthonormal basis for the Hom space (orthonormal under trace inn prod divided by dim of domain).
    # each element of the basis is an isometry from the domain bimod to the codomain bimod.
    def compute_onb(self):
        # self.proj is the orthogonal projector onto the Hom space represented by self,
        # in the format where a rectangular matrix is flattened into a vector
        hom_space_vectorized = self.proj.columnspace()

        # compute an orthogonal (but not orthonormal) basis
        basis_vectorized = sp.GramSchmidt(hom_space_vectorized, orthonormal=False)

        # normalize according to the usual inner product divided by dim_domain.
        # NOTE: divide by dom.dim instead of dom.dim_ambient because an isometry A from dom to codom
        # would have A.H * A being a PROJECTION onto dom, so that its rank is dom.dim.
        dim_domain = self.dom.dim
        def normalize(vec):
            scale = sp.simplify(sp.sqrt(dim_domain) / vec.norm())
            return sp.simplify(vec * scale)
        basis_vectorized = [normalize(vec) for vec in basis_vectorized]

        # convert from flattened vectors back into matrices
        def to_mat(vec):
            mat = vec.reshape(self.codom.dim_ambient, self.dom.dim_ambient)
            return sp.ImmutableMatrix(mat)
        basis = [to_mat(vec) for vec in basis_vectorized]

        # check that an orthonormal basis has been successfully computed.
        # (this is necessary because the Gram-Schmidt code is written by ChatGPT
        # and not carefully examined by me.)
        assert self.dim == len(basis)
        if True:  # debug mode
            assert all([self.test_membership(mat) for mat in basis])
            assert are_matrices_orthonormal_symbolic(basis, dim_domain)

        self.onb = basis


def triv_group():
    mult = [[0]]
    return FiniteGroup(mult)


def untwisted_gp_alg(G: FiniteGroup):
    psi = tuple(tuple([sp.Integer(1) for _ in range(G.n)]) for __ in range(G.n))
    return GpAlg(G, psi, claim_untwisted=True, claim_normalized=True)


def triv_alg():
    G = triv_group()
    psi = ((sp.sympify(1),),)  # trivial cocycle
    return GpAlg(G, psi, claim_normalized=True)


# trivial right representation of the trivial group
def triv_right_rep(deg, alg=None):
    assert isinstance(deg, int) and deg >= 1

    # by default, alg is a fresh instance of the trivial group algebra.
    # optionally, allow the user to input an existing instance instead.
    if isinstance(alg, type(None)):
        alg = triv_alg()
    else:
        assert isinstance(alg, GpAlg)

    rho = (sp.ImmutableMatrix(sp.eye(deg)),)

    return RightRep(alg, deg, rho)


# recast a left rep into a bimodule by equipping it with a trivial right action.
# here, we let P be the identity because the entire C^{rep.deg} is allowed.
def left_rep_to_bimod(rep, alg_right=None):
    assert isinstance(rep, LeftRep)

    I = sp.ImmutableMatrix(sp.eye(rep.deg))
    return Bimod(rep, triv_right_rep(rep.deg, alg=alg_right), I)


# determine whether a list of sympy matrices is orthonormal under a modified trace (Hilbert-Schmidt) inner product
def are_matrices_orthonormal_symbolic(lst, scale):
    assert isinstance(lst, list)

    for (i, j) in itertools.product(range(len(lst)), repeat=2):
        A, B = lst[i], lst[j]
        dim_domain = A.cols  # dimension of the vector space from which A, B map
        prod = sp.simplify(sp.trace(A.H * B)) / scale
        if i == j:
            if prod != 1:
                return False
        else:
            if prod != 0:
                return False

    return True
