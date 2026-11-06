"""RfamGen Figure 2 sequence/structure preprocessing primitives.

The RfamGen paper filters family sequences by whether their consensus-guided
secondary structure is parsable by the paper's G6 grammar.  The public
repository contains the reference implementations in ``src/preprocess.py``
and ``src/grammar.py``, but those modules also import the model training stack.
This module keeps the benchmark preprocessing dependency-light while
preserving the relevant semantics:

* normalize ``SS_cons`` exactly like ``preprocess.normalize_SS_cons``;
* fold each aligned sequence exactly like ``preprocess.fold_by_SS_cons``;
* recognize exactly the language of ``grammar.grammar_g6``.

The linear G6 recognizer is suitable for the long Figure 2 families.  A small
NLTK parity suite is provided so worker runs can check it against the grammar
used by the official repository before processing a family.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable


G6_GRAMMAR = """
S -> L S | L
L -> 'A(' F 'U)' | 'U(' F 'A)' | 'G(' F 'C)' | 'C(' F 'G)' | 'G(' F 'U)' | 'U(' F 'G)' | 'A.' | 'U.' | 'G.' | 'C.'
F -> 'A(' F 'U)' | 'U(' F 'A)' | 'G(' F 'C)' | 'C(' F 'G)' | 'G(' F 'U)' | 'U(' F 'G)' | L S
Nothing -> None
"""

RNA_ALPHABET = frozenset("ACGU")
CANONICAL_PAIRS = frozenset(
    {
        ("A", "U"),
        ("U", "A"),
        ("G", "C"),
        ("C", "G"),
        ("G", "U"),
        ("U", "G"),
    }
)


@dataclass(frozen=True)
class G6ParseResult:
    """Result of recognizing one sequence/structure pair with G6."""

    parsable: bool
    reason: str
    derivation_rule_count: int | None


@dataclass(frozen=True)
class _G6Element:
    kind: str
    l_parsable: bool
    l_rule_count: int | None
    f_parsable: bool | None = None
    f_rule_count: int | None = None


@dataclass(frozen=True)
class StockholmAlignment:
    """The sequence rows and consensus structure from one Stockholm block."""

    sequences: dict[str, str]
    sequence_order: tuple[str, ...]
    ss_cons: str
    declared_sequence_count: int | None


def normalize_ss_cons(ss_cons: str) -> str:
    """Map WUSS consensus structure to the repository's dot-bracket subset."""

    normalized: list[str] = []
    for character in ss_cons:
        if character == "<":
            normalized.append("(")
        elif character == ">":
            normalized.append(")")
        elif character in {"(", ")"}:
            normalized.append(character)
        else:
            normalized.append(".")
    return "".join(normalized)


def fold_by_ss_cons(aligned_sequence: str, normalized_ss_cons: str) -> tuple[str, str]:
    """Recover a sequence-specific structure using the official fold rules.

    Cmalign may use ``.`` as a sequence gap in Stockholm output whereas the
    official repository first round-trips the alignment through Biopython,
    which writes ``-``.  Treating both as a gap reproduces that round trip.
    Invalid consensus nesting is an upstream alignment error and therefore
    raises instead of being classified as a normal G6 rejection.
    """

    if len(aligned_sequence) != len(normalized_ss_cons):
        raise ValueError(
            "aligned sequence and normalized SS_cons have different lengths: "
            f"{len(aligned_sequence)} != {len(normalized_ss_cons)}"
        )

    aligned = aligned_sequence.upper().replace(".", "-")
    invalid = set(aligned).difference(RNA_ALPHABET | {"-"})
    if invalid:
        raise ValueError(f"aligned sequence contains unsupported symbols: {sorted(invalid)}")

    open_pairs: list[tuple[int, str]] = []
    sequence_columns: list[str] = []
    structure_columns: list[str] = []
    for position, (base, consensus) in enumerate(zip(aligned, normalized_ss_cons)):
        if consensus == "(":
            open_pairs.append((position, base))
            sequence_base, structure_base = base, "("
        elif consensus == ")":
            if not open_pairs:
                raise ValueError(
                    f"SS_cons has an unmatched closing parenthesis at column {position}"
                )
            opening_position, opening_base = open_pairs.pop()
            if base != "-" and opening_base != "-":
                if (opening_base, base) in CANONICAL_PAIRS:
                    sequence_base, structure_base = base, ")"
                else:
                    sequence_base, structure_base = base, "."
                    structure_columns[opening_position] = "."
            elif base == "-" and opening_base != "-":
                sequence_base, structure_base = "-", " "
                structure_columns[opening_position] = "."
            elif base != "-" and opening_base == "-":
                sequence_base, structure_base = base, "."
                structure_columns[opening_position] = " "
            else:
                sequence_base, structure_base = "-", " "
                structure_columns[opening_position] = " "
        elif consensus == "." and base != "-":
            sequence_base, structure_base = base, "."
        elif consensus == ".":
            sequence_base, structure_base = "-", " "
        else:
            raise ValueError(f"unsupported normalized SS_cons symbol: {consensus!r}")

        sequence_columns.append(sequence_base)
        structure_columns.append(structure_base)

    if open_pairs:
        positions = [position for position, _base in open_pairs]
        raise ValueError(f"SS_cons has unmatched opening parentheses at columns {positions}")

    sequence = "".join(sequence_columns).replace("-", "")
    structure = "".join(structure_columns).replace(" ", "")
    if len(sequence) != len(structure):
        raise AssertionError(
            "official consensus fold produced unequal sequence/structure lengths: "
            f"{len(sequence)} != {len(structure)}"
        )
    return sequence, structure


def recognize_g6(sequence: str, structure: str) -> G6ParseResult:
    """Recognize the exact language of the RfamGen G6 grammar in linear time."""

    if len(sequence) != len(structure):
        raise ValueError(
            f"sequence and structure lengths differ: {len(sequence)} != {len(structure)}"
        )
    if not sequence:
        return G6ParseResult(False, "empty_sequence", None)
    invalid_bases = set(sequence).difference(RNA_ALPHABET)
    if invalid_bases:
        raise ValueError(f"sequence contains non-RNA symbols: {sorted(invalid_bases)}")
    invalid_structure = set(structure).difference({".", "(", ")"})
    if invalid_structure:
        raise ValueError(
            f"structure contains non-dot-bracket symbols: {sorted(invalid_structure)}"
        )

    element_stacks: list[list[_G6Element]] = [[]]
    opening_bases: list[tuple[int, str]] = []
    first_invalid_reason: str | None = None

    for position, (base, symbol) in enumerate(zip(sequence, structure)):
        if symbol == ".":
            element_stacks[-1].append(
                _G6Element(kind="unpaired", l_parsable=True, l_rule_count=1)
            )
            continue
        if symbol == "(":
            opening_bases.append((position, base))
            element_stacks.append([])
            continue
        if len(element_stacks) == 1:
            return G6ParseResult(False, f"unmatched_close_at_{position}", None)

        children = element_stacks.pop()
        opening_position, opening_base = opening_bases.pop()
        canonical = (opening_base, base) in CANONICAL_PAIRS

        f_parsable = False
        f_rule_count: int | None = None
        if len(children) == 1 and children[0].kind == "pair":
            child = children[0]
            if child.f_parsable:
                f_parsable = True
                if child.f_rule_count is None:
                    raise AssertionError("parsable nested F has no rule count")
                f_rule_count = 1 + child.f_rule_count
        elif len(children) >= 2 and all(child.l_parsable for child in children):
            child_rule_counts = [child.l_rule_count for child in children]
            if any(count is None for count in child_rule_counts):
                raise AssertionError("parsable L has no rule count")
            # F -> L S contributes one rule.  S over the remaining k-1 L
            # elements contributes k-1 rules.
            f_rule_count = len(children) + sum(
                int(count) for count in child_rule_counts
            )
            f_parsable = True

        l_parsable = canonical and f_parsable
        l_rule_count = 1 + f_rule_count if l_parsable and f_rule_count is not None else None
        if not canonical and first_invalid_reason is None:
            first_invalid_reason = (
                f"noncanonical_pair_{opening_base}{base}_at_"
                f"{opening_position}_{position}"
            )
        elif canonical and not f_parsable and first_invalid_reason is None:
            first_invalid_reason = (
                f"g6_invalid_pair_interior_at_{opening_position}_{position}"
            )
        element_stacks[-1].append(
            _G6Element(
                kind="pair",
                l_parsable=l_parsable,
                l_rule_count=l_rule_count,
                f_parsable=f_parsable,
                f_rule_count=f_rule_count,
            )
        )

    if len(element_stacks) != 1:
        positions = [position for position, _base in opening_bases]
        return G6ParseResult(False, f"unmatched_open_at_{positions}", None)

    roots = element_stacks[0]
    if not roots:
        return G6ParseResult(False, "empty_sequence", None)
    if not all(element.l_parsable for element in roots):
        return G6ParseResult(False, first_invalid_reason or "g6_unparsable", None)
    root_rule_counts = [element.l_rule_count for element in roots]
    if any(count is None for count in root_rule_counts):
        raise AssertionError("parsable root L has no rule count")
    rule_count = len(roots) + sum(int(count) for count in root_rule_counts)
    if rule_count > 2 * len(sequence):
        raise AssertionError(
            f"G6 derivation has {rule_count} rules for {len(sequence)} residues"
        )
    return G6ParseResult(True, "parsable", rule_count)


def build_nltk_g6_parser() -> Any:
    """Build the same NLTK ``ChartParser`` used by the public repository."""

    try:
        import nltk
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "NLTK is required for the official G6 parity check; install it in the "
            "worker environment or pass --skip-nltk-parity explicitly"
        ) from error

    return nltk.ChartParser(nltk.CFG.fromstring(G6_GRAMMAR))


def nltk_g6_parse_count(
    sequence: str,
    structure: str,
    *,
    parser: Any | None = None,
) -> tuple[int, int | None]:
    """Return the reference NLTK G6 parse count and derivation length."""

    if len(sequence) != len(structure):
        raise ValueError(
            f"sequence and structure lengths differ: {len(sequence)} != {len(structure)}"
        )
    if parser is None:
        parser = build_nltk_g6_parser()
    tokens = [base + symbol for base, symbol in zip(sequence, structure)]
    trees = list(parser.parse(tokens))
    derivation_length = len(trees[0].productions()) if len(trees) == 1 else None
    return len(trees), derivation_length


def _assert_nltk_parity(sequence: str, structure: str, parser: Any) -> None:
    direct = recognize_g6(sequence, structure)
    parse_count, derivation_length = nltk_g6_parse_count(
        sequence, structure, parser=parser
    )
    if direct.parsable != (parse_count == 1):
        raise AssertionError(
            "direct/NLTK G6 parsability mismatch for "
            f"{sequence}/{structure}: direct={direct}, NLTK={parse_count}"
        )
    if direct.parsable and direct.derivation_rule_count != derivation_length:
        raise AssertionError(
            "direct/NLTK G6 derivation-length mismatch for "
            f"{sequence}/{structure}: direct={direct.derivation_rule_count}, "
            f"NLTK={derivation_length}"
        )


def verify_direct_g6_against_nltk(max_exhaustive_length: int = 4) -> int:
    """Exhaustively compare all short RNA/dot-bracket strings with NLTK.

    Length four is the smallest exhaustive suite that includes a valid G6
    hairpin (a canonical pair around two unpaired residues).  It comprises
    22,621 cases including the empty string and catches both language and
    derivation-length disagreements.
    """

    if max_exhaustive_length < 0:
        raise ValueError("max_exhaustive_length must be non-negative")
    parser = build_nltk_g6_parser()
    checked = 0
    _assert_nltk_parity("", "", parser)
    checked += 1
    for length in range(1, max_exhaustive_length + 1):
        for sequence_tuple in product("ACGU", repeat=length):
            sequence = "".join(sequence_tuple)
            for structure_tuple in product(".()", repeat=length):
                _assert_nltk_parity(sequence, "".join(structure_tuple), parser)
                checked += 1
    return checked


def verify_real_g6_case_against_nltk(
    sequence: str,
    structure: str,
    *,
    parser: Any,
) -> None:
    """Fail loudly if one real folded record differs from official NLTK G6."""

    _assert_nltk_parity(sequence, structure, parser)


def read_stockholm_alignment(path: Path) -> StockholmAlignment:
    """Read the single Stockholm/Pfam block emitted by one cmalign call."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty Stockholm alignment: {path}")

    sequence_parts: dict[str, list[str]] = {}
    sequence_order: list[str] = []
    ss_cons_parts: list[str] = []
    declared_sequence_count: int | None = None
    saw_header = False
    saw_terminator = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line == "# STOCKHOLM 1.0":
                if saw_header:
                    raise ValueError(f"{path}:{line_number}: multiple Stockholm blocks")
                saw_header = True
                continue
            if line == "//":
                if not saw_header:
                    raise ValueError(f"{path}:{line_number}: terminator before header")
                saw_terminator = True
                continue
            if saw_terminator:
                raise ValueError(f"{path}:{line_number}: content after Stockholm terminator")
            if line.startswith("#=GF SQ"):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError(f"{path}:{line_number}: malformed #=GF SQ line")
                value = int(fields[2])
                if declared_sequence_count is not None and value != declared_sequence_count:
                    raise ValueError(f"{path}:{line_number}: conflicting #=GF SQ values")
                declared_sequence_count = value
                continue
            if line.startswith("#=GC SS_cons"):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError(f"{path}:{line_number}: malformed SS_cons line")
                ss_cons_parts.append(fields[2])
                continue
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: malformed sequence row")
            identifier, aligned_piece = fields
            if identifier not in sequence_parts:
                sequence_parts[identifier] = []
                sequence_order.append(identifier)
            sequence_parts[identifier].append(aligned_piece)

    if not saw_header or not saw_terminator:
        raise ValueError(f"{path}: incomplete Stockholm block")
    if not sequence_parts:
        raise ValueError(f"{path}: alignment contains no sequences")
    if not ss_cons_parts:
        raise ValueError(f"{path}: alignment contains no #=GC SS_cons annotation")

    sequences = {
        identifier: "".join(sequence_parts[identifier]) for identifier in sequence_order
    }
    ss_cons = "".join(ss_cons_parts)
    lengths = {len(sequence) for sequence in sequences.values()}
    lengths.add(len(ss_cons))
    if len(lengths) != 1:
        raise ValueError(
            f"{path}: aligned sequence/SS_cons lengths disagree: {sorted(lengths)}"
        )
    if declared_sequence_count is not None and declared_sequence_count != len(sequences):
        raise ValueError(
            f"{path}: #=GF SQ declares {declared_sequence_count}, found {len(sequences)}"
        )
    return StockholmAlignment(
        sequences=sequences,
        sequence_order=tuple(sequence_order),
        ss_cons=ss_cons,
        declared_sequence_count=declared_sequence_count,
    )


def count_fasta_records(lines: Iterable[str]) -> int:
    """Count FASTA records in a text stream, rejecting sequence-before-header."""

    count = 0
    saw_header = False
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            count += 1
            saw_header = True
        elif not saw_header:
            raise ValueError(f"line {line_number}: sequence before FASTA header")
    return count


__all__ = [
    "CANONICAL_PAIRS",
    "G6_GRAMMAR",
    "G6ParseResult",
    "StockholmAlignment",
    "build_nltk_g6_parser",
    "count_fasta_records",
    "fold_by_ss_cons",
    "nltk_g6_parse_count",
    "normalize_ss_cons",
    "read_stockholm_alignment",
    "recognize_g6",
    "verify_direct_g6_against_nltk",
    "verify_real_g6_case_against_nltk",
]
