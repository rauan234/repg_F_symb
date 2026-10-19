#!/usr/bin/env python3

# read representation theoretic data from a json file into a sympy format.

# WARNING: this code was written by ChatGPT and not carefully examined by me.
# in my code, it is used in such a way that the validity of the output is verified by code that I directly wrote.
# the classes FiniteGroup, LeftRep, etc., are such that their __init__ method automatically checks for validity,
# so anything that comes out of this code is checked. what's NOT verified is whether the output of this code
# does correctly represent the content of the json file being read.

"""Construct, serialize, deserialize, and verify the requested S_3 data.

Enumeration conventions
-----------------------
Write

    S_3 = <r, s | r^3 = s^2 = 1, s r s = r^(-1)>.

The parent-group element r^a s^b has label a + 3*b, where 0 <= a < 3
and b is 0 or 1. Thus the labels are

    0: 1,  1: r,  2: r^2,  3: s,  4: r s,  5: r^2 s.

The subgroup isomorphism types are stored in this order:

    0: {1},  1: C_2=<s>,  2: C_3=<r>,  3: S_3.

Every one of these groups has trivial H^2(-, U(1)), so each subgroup has
exactly one cohomology sector, represented by the cocycle identically 1.

JSON encoding
-------------
Python integers, lists, and dictionaries are encoded normally. Every SymPy
expression x is encoded as the tagged JSON object

    {"__sympy_srepr__": sympy.srepr(x)}.

On reading, the srepr string is reconstructed with sympy.sympify. Consequently
read_from_file should only be used on trusted JSON files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import sympy as sp


SYMPY_TAG = "__sympy_srepr__"


# ---------------------------------------------------------------------------
# Exact SymPy-aware JSON serialization
# ---------------------------------------------------------------------------


def _encode_for_json(obj: Any) -> Any:
    if isinstance(obj, sp.Basic):
        return {SYMPY_TAG: sp.srepr(obj)}
    if isinstance(obj, dict):
        return {key: _encode_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_for_json(value) for value in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"Cannot encode object of type {type(obj).__name__}")


def _decode_from_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {SYMPY_TAG}:
            payload = obj[SYMPY_TAG]
            if not isinstance(payload, str):
                raise TypeError(f"{SYMPY_TAG} payload must be a string")
            return sp.sympify(payload)
        return {key: _decode_from_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode_from_json(value) for value in obj]
    return obj


def write_to_file(data: dict[str, Any], filename: str | os.PathLike[str]) -> None:
    """Atomically write data to JSON, preserving exact SymPy expressions."""
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_for_json(data)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(encoded, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def read_from_file(filename: str | os.PathLike[str]) -> dict[str, Any]:
    """Read JSON written by write_to_file and restore all SymPy expressions."""
    with Path(filename).open("r", encoding="utf-8") as source:
        encoded = json.load(source)
    decoded = _decode_from_json(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("The top-level JSON value must decode to a dictionary")
    return decoded


# ---------------------------------------------------------------------------
# Groups and exact representations
# ---------------------------------------------------------------------------


def semidirect_mult_table(rotation_order: int) -> list[list[int]]:
    """Multiplication for <r,s | r^n=s^2=1, srs=r^(-1)>.

    Elements r^a s^b are labeled a + rotation_order*b.
    """

    def multiply(g: int, h: int) -> int:
        a, b = g % rotation_order, g // rotation_order
        c, d = h % rotation_order, h // rotation_order
        exponent = (a + (c if b == 0 else -c)) % rotation_order
        return exponent + rotation_order * ((b + d) % 2)

    order = 2 * rotation_order
    return [[multiply(g, h) for h in range(order)] for g in range(order)]


def cyclic_mult_table(order: int) -> list[list[int]]:
    return [[(i + j) % order for j in range(order)] for i in range(order)]


def matrix_entries(matrix: sp.MatrixBase) -> list[list[sp.Expr]]:
    """Convert a SymPy matrix to the requested nested list of exact scalars."""
    return [
        [sp.simplify(sp.expand_complex(matrix[i, j])) for j in range(matrix.cols)]
        for i in range(matrix.rows)
    ]


def one_dimensional_rep(values: list[sp.Expr]) -> list[list[list[sp.Expr]]]:
    return [[[sp.simplify(value)]] for value in values]


def trivial_cocycle(order: int) -> list[list[sp.Expr]]:
    return [[sp.Integer(1) for _j in range(order)] for _i in range(order)]


def trivial_group_irreps() -> list[list[list[list[sp.Expr]]]]:
    return [one_dimensional_rep([sp.Integer(1)])]


def c2_irreps() -> list[list[list[list[sp.Expr]]]]:
    return [
        one_dimensional_rep([sp.Integer(1), sp.Integer(1)]),
        one_dimensional_rep([sp.Integer(1), sp.Integer(-1)]),
    ]


def c3_irreps() -> list[list[list[list[sp.Expr]]]]:
    omega = -sp.Rational(1, 2) + sp.I * sp.sqrt(3) / 2
    return [
        one_dimensional_rep(
            [sp.simplify(sp.expand_complex(omega ** (character * exponent)))
             for exponent in range(3)]
        )
        for character in range(3)
    ]


def s3_irreps() -> list[list[list[list[sp.Expr]]]]:
    # Trivial and sign representations.
    trivial = one_dimensional_rep([sp.Integer(1)] * 6)
    sign = one_dimensional_rep(
        [sp.Integer(1)] * 3 + [sp.Integer(-1)] * 3
    )

    # A real orthogonal realization of the standard representation.
    rotation = sp.ImmutableMatrix(
        [
            [-sp.Rational(1, 2), -sp.sqrt(3) / 2],
            [sp.sqrt(3) / 2, -sp.Rational(1, 2)],
        ]
    )
    reflection = sp.ImmutableMatrix([[1, 0], [0, -1]])
    standard = []
    for label in range(6):
        a, b = label % 3, label // 3
        standard.append(matrix_entries(rotation ** a * reflection ** b))

    return [trivial, sign, standard]


def one_cohomology_sector(
    order: int,
    irreps: list[list[list[list[sp.Expr]]]],
) -> list[dict[str, Any]]:
    return [{"cocycle": trivial_cocycle(order), "irreps": irreps}]


def build_data() -> dict[str, Any]:
    s3_mult = semidirect_mult_table(3)

    subgroup_data = [
        {
            "mult": cyclic_mult_table(1),
            "incl_map": [0],
            "coh_and_irreps": one_cohomology_sector(1, trivial_group_irreps()),
        },
        {
            "mult": cyclic_mult_table(2),
            "incl_map": [0, 3],  # 1, s
            "coh_and_irreps": one_cohomology_sector(2, c2_irreps()),
        },
        {
            "mult": cyclic_mult_table(3),
            "incl_map": [0, 1, 2],  # 1, r, r^2
            "coh_and_irreps": one_cohomology_sector(3, c3_irreps()),
        },
        {
            "mult": s3_mult,
            "incl_map": list(range(6)),
            "coh_and_irreps": one_cohomology_sector(6, s3_irreps()),
        },
    ]

    return {
        "G": {"mult": s3_mult, "irreps": s3_irreps()},
        "subgroups": subgroup_data,
    }


data = build_data()


# ---------------------------------------------------------------------------
# Exact verification
# ---------------------------------------------------------------------------


def _is_zero(expression: sp.Expr) -> bool:
    return sp.simplify(sp.expand_complex(expression)) == 0


def _is_zero_matrix(matrix: sp.MatrixBase) -> bool:
    return all(_is_zero(entry) for entry in matrix)


def verify_group(mult: list[list[int]]) -> None:
    order = len(mult)
    assert order > 0
    assert all(len(row) == order for row in mult)
    assert all(mult[0][g] == g and mult[g][0] == g for g in range(order))
    for g in range(order):
        assert any(mult[g][h] == 0 and mult[h][g] == 0 for h in range(order))
        for h in range(order):
            assert 0 <= mult[g][h] < order
            for k in range(order):
                assert mult[mult[g][h]][k] == mult[g][mult[h][k]]


def verify_cocycle(mult: list[list[int]], cocycle: list[list[sp.Expr]]) -> None:
    order = len(mult)
    assert len(cocycle) == order and all(len(row) == order for row in cocycle)
    assert all(_is_zero(cocycle[0][g] - 1) for g in range(order))
    assert all(_is_zero(cocycle[g][0] - 1) for g in range(order))
    for g in range(order):
        for h in range(order):
            for k in range(order):
                lhs = cocycle[g][h] * cocycle[mult[g][h]][k]
                rhs = cocycle[h][k] * cocycle[g][mult[h][k]]
                assert _is_zero(lhs - rhs)


def verify_irrep_sector(
    mult: list[list[int]],
    cocycle: list[list[sp.Expr]],
    irreps: list[list[list[list[sp.Expr]]]],
) -> None:
    order = len(mult)
    dimensions: list[int] = []
    matrices: list[list[sp.ImmutableMatrix]] = []

    for representation in irreps:
        assert len(representation) == order
        dimension = len(representation[0])
        assert dimension > 0
        converted = []
        for entries in representation:
            matrix = sp.ImmutableMatrix(entries)
            assert matrix.shape == (dimension, dimension)
            converted.append(matrix)
        dimensions.append(dimension)
        matrices.append(converted)

        identity = sp.eye(dimension)
        assert converted[0] == identity
        for g in range(order):
            assert _is_zero_matrix(converted[g].conjugate().T * converted[g] - identity)
            for h in range(order):
                lhs = converted[g] * converted[h]
                rhs = cocycle[g][h] * converted[mult[g][h]]
                assert _is_zero_matrix(lhs - rhs)

    # Exact (projective) character inner products prove irreducibility and
    # pairwise inequivalence within this fixed cocycle sector.
    for i, representation_i in enumerate(matrices):
        for j, representation_j in enumerate(matrices):
            inner_product = sum(
                sp.conjugate(sp.trace(representation_i[g]))
                * sp.trace(representation_j[g])
                for g in range(order)
            ) / order
            expected = 1 if i == j else 0
            assert _is_zero(inner_product - expected)

    # Together with irreducibility, this proves completeness.
    assert sum(dimension ** 2 for dimension in dimensions) == order


def verify_data(candidate: dict[str, Any]) -> None:
    assert set(candidate) == {"G", "subgroups"}
    parent = candidate["G"]
    assert set(parent) == {"mult", "irreps"}
    verify_group(parent["mult"])
    verify_irrep_sector(
        parent["mult"], trivial_cocycle(len(parent["mult"])), parent["irreps"]
    )

    subgroups = candidate["subgroups"]
    assert [len(subgroup["mult"]) for subgroup in subgroups] == [1, 2, 3, 6]
    parent_order = len(parent["mult"])
    for subgroup in subgroups:
        assert set(subgroup) == {"mult", "incl_map", "coh_and_irreps"}
        mult = subgroup["mult"]
        inclusion = subgroup["incl_map"]
        order = len(mult)
        verify_group(mult)

        assert len(inclusion) == order
        assert len(set(inclusion)) == order
        assert inclusion[0] == 0
        assert all(0 <= image < parent_order for image in inclusion)
        for i in range(order):
            for j in range(order):
                assert (
                    parent["mult"][inclusion[i]][inclusion[j]]
                    == inclusion[mult[i][j]]
                )

        # H^2(H_s,U(1))=0 for all four subgroup types, hence one sector each.
        sectors = subgroup["coh_and_irreps"]
        assert len(sectors) == 1
        for sector in sectors:
            assert set(sector) == {"cocycle", "irreps"}
            verify_cocycle(mult, sector["cocycle"])
            verify_irrep_sector(mult, sector["cocycle"], sector["irreps"])


def main() -> None:
    output = Path(__file__).with_name("S3_data.json")
    verify_data(data)
    write_to_file(data, output)
    recovered = read_from_file(output)
    assert recovered == data
    verify_data(recovered)

    print(f"Wrote {output}")
    print("Verified exact JSON round trip.")
    print("Subgroup orders: [1, 2, 3, 6]")
    print("Cohomology-sector counts: [1, 1, 1, 1]")
    print("Irrep degree patterns: [[1], [1,1], [1,1,1], [1,1,2]]")


if __name__ == "__main__":
    main()
