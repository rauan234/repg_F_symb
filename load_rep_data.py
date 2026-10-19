import sympy as sp

from RepTheoryBasic import FiniteGroup, SubgroupRelation, GpAlg, LeftRep, untwisted_gp_alg
from FSymbForRepG import RepG, RepSubgroupTwisted

# given the output of read_write_GPT.py, convert it into FiniteGroup, LeftRep, etc. objects.
# note: the __init__ of FiniteGroup, LeftRep, etc., automatically test the input for whether it
# gives a valid finite group, left representation, etc.


def make_cocycle_normalized(alg, irreps):
    assert isinstance(alg, GpAlg)
    assert isinstance(irreps, list)

    alg_new, lam = alg.copy_with_normalized_cocycle()
    reps_new = []
    for rep in irreps:
        assert isinstance(rep, LeftRep)
        assert rep.alg == alg  # require that the algebras are the SAME GpAlg object

        reps_new.append(rep.copy_with_coboundary_change(alg_new, lam))

    return alg_new, tuple(reps_new)


# load a group from a list format into FiniteFroup
def load_group(mult_lst):
    assert isinstance(mult_lst, list)

    G = FiniteGroup(mult_lst)
    return G


# load an irrep from a list format into LeftRep
def load_irrep(alg, irrep_data):
    # irrep_data[i] stores rho(g_i) for each 0 <= i < n indexing elements of G
    assert isinstance(irrep_data, list)
    assert len(irrep_data) == alg.G.n

    # each irrep_data[i] is supposed to be a deg-by-deg list
    deg = len(irrep_data[0])

    # convert into a tuple of ImmutableMatrix
    rho = tuple([
        sp.ImmutableMatrix(mat_spec)
        for mat_spec in irrep_data])

    # cast into LeftRep, checking here that rho does indeed satisfy multiplicativity
    rep = LeftRep(alg, deg, rho)

    return rep


# load a subgroup relationship by converting the inclusion map data from a list into SubgroupRelation
def load_subg_rel(G, H, incl_data):
    assert isinstance(G, FiniteGroup)
    assert isinstance(H, FiniteGroup)
    assert isinstance(incl_data, list)

    subg_rel = SubgroupRelation(G, H, tuple(incl_data))

    return subg_rel


# load the data of untwisted representations of G
def load_repg(G_data):
    assert isinstance(G_data, dict)
    assert set(G_data.keys()) == set(["mult", "irreps"])

    # group
    G = load_group(G_data["mult"])

    # untwisted group algebra
    alg = untwisted_gp_alg(G)

    # irreps
    irreps = tuple([load_irrep(alg, irrep_data) for irrep_data in G_data["irreps"]])

    # cast into RepG
    repg = RepG(G, irreps)

    return repg


# load a cohomology class and the corresponding twisted irreps
def load_coh_class(H, cls_data):
    assert isinstance(H, FiniteGroup)
    assert isinstance(cls_data, dict)
    assert set(cls_data.keys()) == set(["cocycle", "irreps"])

    # cocycle
    assert isinstance(cls_data["cocycle"], list)
    psi_raw = tuple([
        tuple([x for x in temp])
        for temp in cls_data["cocycle"]])  # convert from nested list into nested tuple

    # test if the cocycle is valid and cast into GpAlg
    alg_raw = GpAlg(H, psi_raw)

    # irreps
    irreps_raw = [load_irrep(alg_raw, irrep_data) for irrep_data in cls_data["irreps"]]

    # modify alg and irreps by a U(1) second cohomology coboundary
    # to ensure that psi(g, g^{-1}) = 1 for all g
    alg, irreps = make_cocycle_normalized(alg_raw, irreps_raw)

    return alg, irreps


# load the cohomology classes of a subgroup H. for each class, load the irreps and make a RepSubgroupTwisted object.
def load_subgroup(repg, subgroup_data):
    assert isinstance(repg, RepG)  # parent group
    assert isinstance(subgroup_data, dict)
    assert set(subgroup_data.keys()) == set(["mult", "incl_map", "coh_and_irreps"])

    # subgroup
    H = load_group(subgroup_data["mult"])

    # inclusion map gives rise to a subgroup relation
    subg_rel = load_subg_rel(repg.G, H, subgroup_data["incl_map"])

    # a list where each entry specifies a cohomology class of H and the corresponding irreps
    assert isinstance(subgroup_data["coh_and_irreps"], list)
    coh_and_irreps = [load_coh_class(H, cls_data) for cls_data in subgroup_data["coh_and_irreps"]]

    # cast the data into RepSubgroupTwisted form
    subg_rep_list = [
        RepSubgroupTwisted(repg, subg_rel, alg, irreps)
        for (alg, irreps) in coh_and_irreps  # iterate over cohomology classes and the corresponding irreps
    ]

    return subg_rep_list