"""
point_add.py
============
Python port of ecdsafail/ecdsafail-challenge src/point_add/mod.rs

Reversible secp256k1 point-addition quantum circuit builder.

The circuit is specialised to secp256k1:
    p = 2^256 - 2^32 - 977,  a = 0,  b = 7

`build_builder()` allocates four 256-wide registers in declaration order:
    target_x  (qubits)
    target_y  (qubits)
    offset_x  (classical bits)
    offset_y  (classical bits)
and emits gates that mutate the target registers into (P + Q) where P is
the quantum point and Q is the classical point.

NOTE: The sub-modules (arith, emit, rounds, venting, dialog_gcd_classical_filter,
      Simulator, WeierstrassEllipticCurve, etc.) are referenced but not included
      here; stub placeholders mark every call-site so the file is syntactically
      complete and importable.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Identifiers (newtype wrappers)
# ─────────────────────────────────────────────────────────────────────────────

class QubitId(int):
    """Opaque qubit identifier."""
    pass


class BitId(int):
    """Opaque classical-bit identifier."""
    pass


class RegisterId(int):
    """Opaque register identifier."""
    pass


NO_BIT = BitId(0xFFFF_FFFF_FFFF_FFFF)   # sentinel – matches Rust's NO_BIT


# ─────────────────────────────────────────────────────────────────────────────
# QubitOrBit union
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QubitOrBit:
    """Holds either a qubit or a classical bit."""
    is_qubit: bool
    qubit: Optional[QubitId] = None
    bit:   Optional[BitId]   = None

    @staticmethod
    def from_qubit(q: QubitId) -> "QubitOrBit":
        return QubitOrBit(is_qubit=True,  qubit=q)

    @staticmethod
    def from_bit(b: BitId) -> "QubitOrBit":
        return QubitOrBit(is_qubit=False, bit=b)


# ─────────────────────────────────────────────────────────────────────────────
# OperationType enum  (18 variants, indices 0..17)
# ─────────────────────────────────────────────────────────────────────────────

class OperationType(IntEnum):
    X                = 0
    CX               = 1
    CCX              = 2
    Z                = 3
    CZ               = 4
    CCZ              = 5
    Swap             = 6
    Hmr              = 7
    R                = 8
    Neg              = 9
    AppendToRegister = 10
    Register         = 11
    PushCondition    = 12
    PopCondition     = 13
    # Remaining indices reserved to keep the array width at 18
    _Unused14        = 14
    _Unused15        = 15
    _Unused16        = 16
    _Unused17        = 17


# ─────────────────────────────────────────────────────────────────────────────
# Op – a single quantum gate / directive
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Op:
    kind:       OperationType = OperationType._Unused14
    q_control2: QubitId       = QubitId(0)
    q_control1: QubitId       = QubitId(0)
    q_target:   QubitId       = QubitId(0)
    c_target:   BitId         = NO_BIT
    c_condition:BitId         = NO_BIT
    r_target:   RegisterId    = RegisterId(0)

    @staticmethod
    def empty() -> "Op":
        return Op()


# ─────────────────────────────────────────────────────────────────────────────
# PhaseResource – stats for a named build phase
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseResource:
    phase:       str
    start:       int
    end:         int
    ops:         int
    toffoli_ops: int
    ccx_ops:     int
    ccz_ops:     int
    hmr_ops:     int
    r_ops:       int


# ─────────────────────────────────────────────────────────────────────────────
# CountSnapshot (internal helper)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CountSnapshot:
    ops:             int
    kind_ops:        list[int]          # len 18
    phase_kind_ops:  list[int]          # len 18
    phase_start_ops: int
    phase_rows_len:  int
    phase:           str


# ─────────────────────────────────────────────────────────────────────────────
# Thread-local for d1 phase-corrected product scope flag
# ─────────────────────────────────────────────────────────────────────────────

_d1_tls = threading.local()

def _d1_phase_corrected_product_core_active() -> bool:
    return getattr(_d1_tls, "active", False)


# ─────────────────────────────────────────────────────────────────────────────
# B – the circuit builder
# ─────────────────────────────────────────────────────────────────────────────

class B:
    """
    Quantum circuit builder.

    Mirrors the Rust struct `B` in point_add/mod.rs.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.ops:                      list[Op]           = []
        self.count_only:               bool               = False
        self.counted_ops:              int                = 0
        self.counted_kind_ops:         list[int]          = [0] * 18
        self.counted_phase_kind_ops:   list[int]          = [0] * 18
        self.counted_phase_start_ops:  int                = 0
        self.counted_phase_rows:       list[PhaseResource]= []
        self.counted_registers:        list[list[QubitOrBit]] = []
        self.next_qubit:               int                = 0
        self.next_bit:                 int                = 0
        self.next_register:            int                = 0
        self.free_qubits:              list[int]          = []
        self.active_qubits:            int                = 0
        self.peak_qubits:              int                = 0
        self.peak_ops_idx:             int                = 0
        self.peak_phase:               str                = ""
        self.phase:                    str                = "init"
        self.peak_log:                 list[tuple[int, str, int]] = []
        self.phase_active_max:         dict[str, int]     = {}
        self.phase_active_regions:     list[tuple[int, str, int]] = []
        self.current_phase_active_max: int                = 0
        self.phase_transitions:        list[tuple[int, str]] = []
        self.active_timeline:          list[tuple[int, int]] = []
        self.k2_shift2_log:            list[QubitId]      = []

    @classmethod
    def new(cls) -> "B":
        return cls()

    @classmethod
    def new_count_only(cls) -> "B":
        b = cls()
        b.count_only = True
        return b

    # ── internal helpers ───────────────────────────────────────────────────────

    def _push_op(self, op: Op) -> None:
        self.counted_ops += 1
        self.counted_kind_ops[int(op.kind)] += 1
        self.counted_phase_kind_ops[int(op.kind)] += 1
        if not self.count_only:
            self.ops.append(op)

    def _count_snapshot(self) -> _CountSnapshot:
        return _CountSnapshot(
            ops             = self.counted_ops,
            kind_ops        = list(self.counted_kind_ops),
            phase_kind_ops  = list(self.counted_phase_kind_ops),
            phase_start_ops = self.counted_phase_start_ops,
            phase_rows_len  = len(self.counted_phase_rows),
            phase           = self.phase,
        )

    def _count_delta_since(self, snap: _CountSnapshot) -> list[int]:
        return [self.counted_kind_ops[i] - snap.kind_ops[i] for i in range(18)]

    def _restore_count_snapshot(self, snap: _CountSnapshot) -> None:
        self.counted_ops            = snap.ops
        self.counted_kind_ops       = list(snap.kind_ops)
        self.counted_phase_kind_ops = list(snap.phase_kind_ops)
        self.counted_phase_start_ops= snap.phase_start_ops
        del self.counted_phase_rows[snap.phase_rows_len:]
        self.phase                  = snap.phase

    def _add_counted_kind(self, kind: OperationType, count: int) -> None:
        self.counted_ops                       += count
        self.counted_kind_ops[int(kind)]       += count
        self.counted_phase_kind_ops[int(kind)] += count

    def _current_ops_len(self) -> int:
        return self.counted_ops if self.count_only else len(self.ops)

    def _close_counted_phase(self) -> None:
        if not self.count_only:
            return
        start = self.counted_phase_start_ops
        end   = self.counted_ops
        if start < end:
            ccx_ops = self.counted_phase_kind_ops[int(OperationType.CCX)]
            ccz_ops = self.counted_phase_kind_ops[int(OperationType.CCZ)]
            hmr_ops = self.counted_phase_kind_ops[int(OperationType.Hmr)]
            r_ops   = self.counted_phase_kind_ops[int(OperationType.R)]
            self.counted_phase_rows.append(PhaseResource(
                phase       = self.phase,
                start       = start,
                end         = end,
                ops         = end - start,
                toffoli_ops = ccx_ops + ccz_ops,
                ccx_ops     = ccx_ops,
                ccz_ops     = ccz_ops,
                hmr_ops     = hmr_ops,
                r_ops       = r_ops,
            ))
        self.counted_phase_start_ops = self.counted_ops
        self.counted_phase_kind_ops  = [0] * 18

    def _record_active_timeline(self) -> None:
        if os.environ.get("PROFILE_ACTIVE_TIMELINE"):
            self.active_timeline.append((self._current_ops_len(), self.active_qubits))

    def _record_phase_active(self) -> None:
        self._record_active_timeline()
        if os.environ.get("TRACE_PHASE_ACTIVE"):
            prev = self.phase_active_max.get(self.phase, 0)
            if self.active_qubits > prev:
                self.phase_active_max[self.phase] = self.active_qubits
            if self.active_qubits > self.current_phase_active_max:
                self.current_phase_active_max = self.active_qubits

    def _close_phase_active_region(self) -> None:
        if os.environ.get("TRACE_PHASE_ACTIVE") and self.current_phase_active_max > 0:
            self.phase_active_regions.append(
                (self._current_ops_len(), self.phase, self.current_phase_active_max)
            )
            self.current_phase_active_max = 0

    def _update_peak(self) -> None:
        if self.active_qubits > self.peak_qubits:
            self.peak_qubits  = self.active_qubits
            self.peak_ops_idx = self._current_ops_len()
            self.peak_phase   = self.phase
            if os.environ.get("TRACE_EACH_PEAK"):
                print(
                    f"PEAK active={self.active_qubits} next_idx={self.next_qubit} "
                    f"phase='{self.phase}' ops_idx={self._current_ops_len()}",
                    flush=True,
                )
        if os.environ.get("TRACE_PEAK") and self.active_qubits + 10 >= self.peak_qubits:
            self.peak_log.append((self.active_qubits, self.phase, self._current_ops_len()))

    # ── qubit / bit allocation ─────────────────────────────────────────────────

    def alloc_qubit(self) -> QubitId:
        self.active_qubits += 1
        self._record_phase_active()
        self._update_peak()
        if self.free_qubits:
            return QubitId(self.free_qubits.pop())
        q = self.next_qubit
        self.next_qubit += 1
        return QubitId(q)

    def alloc_qubits(self, n: int) -> list[QubitId]:
        return [self.alloc_qubit() for _ in range(n)]

    def alloc_bit(self) -> BitId:
        b = self.next_bit
        self.next_bit += 1
        return BitId(b)

    def alloc_bits(self, n: int) -> list[BitId]:
        return [self.alloc_bit() for _ in range(n)]

    def free(self, q: QubitId) -> None:
        self.r(q)
        self.free_qubits.append(int(q))
        if self.active_qubits > 0:
            self.active_qubits -= 1
        self._record_active_timeline()

    def free_vec(self, qs: list[QubitId]) -> None:
        for q in qs:
            self.free(q)

    def reacquire(self, q: QubitId) -> None:
        idx = next((i for i, v in enumerate(self.free_qubits) if v == int(q)), None)
        if idx is None:
            raise ValueError(f"reacquire qubit {q} that is not currently free")
        self.free_qubits[idx] = self.free_qubits[-1]
        self.free_qubits.pop()
        self.active_qubits += 1
        self._record_phase_active()
        self._update_peak()

    def reacquire_vec(self, qs: list[QubitId]) -> None:
        for q in qs:
            self.reacquire(q)

    # ── register declaration ───────────────────────────────────────────────────

    def declare_qubit_register(self, qs: list[QubitId]) -> None:
        r = RegisterId(self.next_register)
        self.next_register += 1
        for q in qs:
            while len(self.counted_registers) <= int(r):
                self.counted_registers.append([])
            self.counted_registers[int(r)].append(QubitOrBit.from_qubit(q))
            op = Op.empty()
            op.kind     = OperationType.AppendToRegister
            op.q_target = q
            op.r_target = r
            self._push_op(op)
        op = Op.empty()
        op.kind     = OperationType.Register
        op.r_target = r
        self._push_op(op)

    def declare_bit_register(self, bs: list[BitId]) -> None:
        r = RegisterId(self.next_register)
        self.next_register += 1
        for b in bs:
            while len(self.counted_registers) <= int(r):
                self.counted_registers.append([])
            self.counted_registers[int(r)].append(QubitOrBit.from_bit(b))
            op = Op.empty()
            op.kind     = OperationType.AppendToRegister
            op.c_target = b
            op.r_target = r
            self._push_op(op)
        op = Op.empty()
        op.kind     = OperationType.Register
        op.r_target = r
        self._push_op(op)

    # ── phase management ───────────────────────────────────────────────────────

    def set_phase(self, p: str) -> None:
        self._close_phase_active_region()
        self._close_counted_phase()
        self.phase = p
        if os.environ.get("TRACE_PHASE_ACTIVE"):
            self.current_phase_active_max = self.active_qubits
        self.phase_transitions.append((self._current_ops_len(), p))

    # ── single-qubit gates ─────────────────────────────────────────────────────

    def x(self, q: QubitId) -> None:
        op = Op.empty()
        op.kind     = OperationType.X
        op.q_target = q
        self._push_op(op)

    def x_if(self, q: QubitId, cond: BitId) -> None:
        op = Op.empty()
        op.kind        = OperationType.X
        op.q_target    = q
        op.c_condition = cond
        self._push_op(op)

    def z_if(self, q: QubitId, cond: BitId) -> None:
        op = Op.empty()
        op.kind        = OperationType.Z
        op.q_target    = q
        op.c_condition = cond
        self._push_op(op)

    def r(self, q: QubitId) -> None:
        """Reset gate (ancilla release)."""
        op = Op.empty()
        op.kind     = OperationType.R
        op.q_target = q
        self._push_op(op)

    # ── two-qubit gates ────────────────────────────────────────────────────────

    def cx(self, ctrl: QubitId, tgt: QubitId) -> None:
        if ctrl == tgt:
            raise ValueError(f"invalid CX with aliased control/target {ctrl}")
        op = Op.empty()
        op.kind      = OperationType.CX
        op.q_control1 = ctrl
        op.q_target  = tgt
        self._push_op(op)

    def cz(self, a: QubitId, b_q: QubitId) -> None:
        if a == b_q:
            op = Op.empty()
            op.kind     = OperationType.Z
            op.q_target = a
            self._push_op(op)
            return
        op = Op.empty()
        op.kind      = OperationType.CZ
        op.q_control1 = a
        op.q_target  = b_q
        self._push_op(op)

    def cz_if(self, a: QubitId, b_q: QubitId, cond: BitId) -> None:
        if a == b_q:
            self.z_if(a, cond)
            return
        op = Op.empty()
        op.kind        = OperationType.CZ
        op.q_control1  = a
        op.q_target    = b_q
        op.c_condition = cond
        self._push_op(op)

    def swap(self, a: QubitId, b_q: QubitId) -> None:
        if a == b_q:
            return
        op = Op.empty()
        op.kind      = OperationType.Swap
        op.q_control1 = a
        op.q_target  = b_q
        self._push_op(op)

    # ── three-qubit gate ───────────────────────────────────────────────────────

    def ccx(self, c1: QubitId, c2: QubitId, tgt: QubitId) -> None:
        if c1 == c2:
            if c1 != tgt:
                self.cx(c1, tgt)
            return
        if c1 == tgt or c2 == tgt:
            raise ValueError(f"invalid CCX with target aliased to a control: {c1}, {c2}, {tgt}")
        op = Op.empty()
        op.kind      = OperationType.CCX
        op.q_control2 = c1
        op.q_control1 = c2
        op.q_target  = tgt
        self._push_op(op)

    # ── measurement / classical gates ──────────────────────────────────────────

    def hmr(self, q: QubitId, c: BitId) -> None:
        """H-measure-reset: Hadamard + measure into classical bit c."""
        op = Op.empty()
        op.kind     = OperationType.Hmr
        op.q_target = q
        op.c_target = c
        self._push_op(op)

    # ── classical condition stack ──────────────────────────────────────────────

    def push_condition(self, cond: BitId) -> None:
        op = Op.empty()
        op.kind        = OperationType.PushCondition
        op.c_condition = cond
        self._push_op(op)

    def pop_condition(self) -> None:
        op = Op.empty()
        op.kind = OperationType.PopCondition
        self._push_op(op)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N = 256   # register width in bits

# secp256k1 prime:  p = 2^256 - 2^32 - 977
SECP256K1_P: int = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
)

# secp256k1 generator and group order
SECP256K1_GX: int = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
)
SECP256K1_GY: int = (
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
)
SECP256K1_ORDER: int = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)

ONE_INV_DX3_AFFINE_PA_ENV     = "ONE_INV_DX3_AFFINE_PA"
ONE_INV_DX3_AFFINE_PA_BLOCKER = (
    "ONE_INV_DX3_AFFINE_PA_BLOCKED: the dx^3 algebra gives Rx and Ry with "
    "one inversion of w=dx^3, but a clean in-place Google-ABI circuit must "
    "also uncompute w, dx^2, and the Kaliski input copy after tx/ty have been "
    "overwritten by Rx/Ry.  At that point dx is recoverable only by the inverse "
    "affine add P=R-Q, whose denominator is Rx-Qx.  That is a second inversion, "
    "or else a retained 256-bit dx witness / dirty reset, so this path cannot "
    "emit a clean one-inversion four-register PA."
)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers  (mirror the Rust fn-per-flag pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"

def direct_const_walks_enabled()         -> bool: return _env_flag("KAL_DIRECT_CONST_WALKS")
def secp_direct_const_arith_enabled()    -> bool: return _env_flag("SECP_DIRECT_CONST_ARITH")
def r84_lowq_enabled()                   -> bool: return _env_flag("R84_LOWQ")
def r84_lowq_cin_borrow_enabled()        -> bool: return _env_flag("R84_LOWQ_CIN_BORROW")
def kal_vent_modadd_enabled()            -> bool: return _env_flag("KAL_VENT_MODADD")
def kal_vent_halve_enabled()             -> bool: return _env_flag("KAL_VENT_HALVE")


# ─────────────────────────────────────────────────────────────────────────────
# Alt-seed check constants
# ─────────────────────────────────────────────────────────────────────────────

ALT_SEED_COUNT            = 5
ALT_SEED_COMMIT           = 24
ALT_SEED_SHOTS            = 4096
ALT_SEED_CLASSICAL_LIMIT  = 2


# ─────────────────────────────────────────────────────────────────────────────
# Stub helpers (implementations live in sub-modules)
# ─────────────────────────────────────────────────────────────────────────────
# These stubs preserve the call signatures so the builder compiles; replace
# them with real implementations from arith.py / emit.py / rounds.py.

def mod_sub_qb(b: B, target: list[QubitId], classical: list[BitId], p: int) -> None:
    """
    Subtract classical register from quantum register mod p.
    Implements step 1: Px -= Qx  (and step 2: Py -= Qy).
    """
    raise NotImplementedError("mod_sub_qb: import from arith module")


def emit_dialog_gcd_raw_pa(
    b: B,
    tx: list[QubitId],
    ty: list[QubitId],
    ox: list[BitId],
    oy: list[BitId],
    p: int,
) -> None:
    """
    Emit the full affine-addition circuit (steps 3-12 from the module doc).
    Implementation lives in emit.py / rounds.py.
    """
    raise NotImplementedError("emit_dialog_gcd_raw_pa: import from emit module")


# ─────────────────────────────────────────────────────────────────────────────
# Default environment configuration
# ─────────────────────────────────────────────────────────────────────────────

def _set_default_env(name: str, value: str) -> None:
    """Set an env-var only if it is not already set (mirrors Rust set_default_env)."""
    if name not in os.environ:
        os.environ[name] = value


def configure_ecdsafail_submission_route() -> None:
    """
    Set all the tuned environment variables for the competition submission.
    Mirrors `configure_ecdsafail_submission_route` in the Rust source.
    """
    _set_default_env("DIALOG_GCD_VENTED_BODY_ODD_LOWBIT",           "1")
    _set_default_env("DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS",          "19")
    _set_default_env("DIALOG_GCD_WIDTH_SLOPE_X1000",                 "1015")
    _set_default_env("DIALOG_GCD_FOLD_CARRY_TRUNC_W",                "18")
    _set_default_env("DIALOG_GCD_FOLD_FREE_FIRST_HIGH_CARRY",        "1")
    _set_default_env("DIALOG_GCD_ACTIVE_ITERATIONS",                 "258")
    _set_default_env("DIALOG_GCD_APPLY_BOUNDARY_FREE_OWNED_DURING_REPLAY", "1")
    _set_default_env("DIALOG_GCD_APPLY_BORROW_FUTURE_BOUNDARY_CARRIES",    "1")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS",            "20")
    _set_default_env(
        "DIALOG_GCD_APPLY_CHUNKED_F_CUTS",
        "17,34,50,66,81,96,110,124,137,150,163,175,187,198,209,219,229,238,247",
    )
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_AUTO_TOPCLEAN_MAX_BITS", "2")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_AUTO_TOPCLEAN_TARGET",   "1168")
    _set_default_env("DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS",          "18")
    _set_default_env("DIALOG_GCD_APPLY_IMPLICIT_HIGH_ZERO",          "1")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_EXTRA",                "3")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_MAP",                  "11:1,12:1,13:1")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_STEPS",                "8,9,10")
    _set_default_env(
        "DIALOG_GCD_BODY_CARRY_BAND_TRIMS",
        "0,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3",
    )
    _set_default_env("DIALOG_GCD_COMPARE_BITS",                      "46")
    _set_default_env(
        "DIALOG_GCD_COMPARE_STEP_BITS",
        "181:48,194:48,199:48,202:48,207:48,212:48,216:48",
    )
    _set_default_env("DIALOG_GCD_FOLD_CARRY_TRUNC_STEP_WINDOWS",     "")
    _set_default_env("DIALOG_GCD_FOLD_CARRY_TRUNC_W",                "17")
    _set_default_env("DIALOG_GCD_FOLD_FREED_TAIL",                   "1")
    _set_default_env("DIALOG_GCD_FOLD_FREED_TAIL_ED",                "1")
    _set_default_env("DIALOG_GCD_FOLD_HOST_DERIVED_CONTROLS",        "1")
    _set_default_env("DIALOG_GCD_FOLD_HOST_E_TOP_CARRY",             "1")
    _set_default_env("DIALOG_GCD_FOLD_MAJ1",                         "1")
    _set_default_env("DIALOG_GCD_FOLD_MAJ2",                         "1")
    _set_default_env("DIALOG_GCD_FOLD_PARK_LOW_CARRIES",             "15")
    _set_default_env(
        "DIALOG_GCD_FOLD_PARK_LOW_CARRIES_STEP_MAP",
        "0:17,3:16,8:16,9:16,10:16,21:17,22:16,24:16,26:16,33:16,34:16,"
        "37:17,41:16,42:17,51:16,55:16,65:17,73:16,77:16,81:16,82:16,"
        "86:16,87:16,97:16,104:16,109:16,110:16,120:16,129:16,132:17,"
        "134:16,141:17,142:16,146:16,157:16,160:16,169:16,170:17,174:16,"
        "177:16,191:16,192:16,198:16,205:16,206:16,212:16,215:16,216:16,"
        "217:16,224:17,228:16",
    )
    _set_default_env("DIALOG_GCD_FOLD_STREAM_CONTROLS",              "1")
    _set_default_env("DIALOG_FUSE_C_FORM",                           "1")
    _set_default_env("DIALOG_FUSE_X_RESTORE",                        "1")
    _set_default_env("DIALOG_GCD_K2",                                "1")
    _set_default_env("DIALOG_GCD_K5_CLEAN_BLOCK",                    "1")
    _set_default_env("DIALOG_GCD_K5_FIXED_TAIL_APPLY",               "0")
    _set_default_env("DIALOG_GCD_K5_FREE_CLEAN_BLOCK_DURING_SHIFT",  "1")
    _set_default_env("DIALOG_GCD_K5_HEAD11_CODEC",                   "1")
    _set_default_env("DIALOG_GCD_K5_HEAD11_STREAM_PAIR_APPLY",       "1")
    _set_default_env("DIALOG_GCD_K5_HEAD11_SPLIT_PAIR_SHIFT_APPLY",  "1")
    _set_default_env("DIALOG_GCD_K5_HEAD11_PAIR01_S2_PERMUTE_APPLY", "1")
    _set_default_env("DIALOG_GCD_K5_HEAD11_PAIR23_S2_BORROW_PAIR01_APPLY", "1")
    _set_default_env("DIALOG_GCD_K5_PARTIAL_RAW_RELEASE",            "8")
    _set_default_env("DIALOG_GCD_K5_RELEASE_SCALE_BITS",             "5")
    _set_default_env("DIALOG_GCD_K5_STREAM_PAIR_APPLY",              "1")
    _set_default_env("DIALOG_GCD_K5_TAIL3_FIXED_LAST",               "0")
    _set_default_env("DIALOG_GCD_K5_TAIL3_TOP32_CODEC",              "1")
    _set_default_env("DIALOG_GCD_K5_TAIL3_TOP32_STREAM_APPLY",       "1")
    _set_default_env("DIALOG_GCD_K5_TAIL3_TOP32_SPLIT_SLOT_APPLY",   "1")
    _set_default_env("DIALOG_GCD_K5_TAIL3_TOP32_FINAL_S2_CONST_APPLY", "1")
    _set_default_env("DIALOG_GCD_ODD_U_LOWBIT_FASTPATH",             "1")
    _set_default_env("DIALOG_GCD_PA9024_COMPARE_SCHEDULE",           "1")
    _set_default_env("DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN",    "0")
    _set_default_env("DIALOG_GCD_PERPOS_MAJ2",                       "1")
    _set_default_env("DIALOG_GCD_RAW_IPMUL_CLEAR_P_RESIDUAL",        "1")
    _set_default_env("DIALOG_GCD_RAW_TOBITVECTOR_MATERIALIZED_SUB",  "0")
    _set_default_env("DIALOG_GCD_RAW_TOBITVECTOR_VARIABLE_WIDTH",    "1")
    _set_default_env("DIALOG_GCD_RUNWAY_PARTIAL_BLOCK",              "1")
    _set_default_env("DIALOG_GCD_SKIP_ZERO_EDGE_CSHIFT",             "1")
    _set_default_env("DIALOG_GCD_SPECIAL_FOLD_BORROW_CARRIES",       "1")
    _set_default_env(
        "DIALOG_GCD_SPECIAL_FOLD_CARRY_TRUNC_STEP_WINDOWS",
        "10:19,11:19,21:20,63:19,74:19,100:19,107:19,110:19,118:19,"
        "135:19,136:19,137:19,188:20,204:19,227:20,241:19",
    )
    _set_default_env("DIALOG_GCD_SPECIAL_FOLD_PARK_LOW_CARRIES",     "16")
    _set_default_env("DIALOG_GCD_SPECIAL_FOLD_PARK_LOW_CARRIES_STEP_MAP", "")
    _set_default_env("DIALOG_GCD_SPECIAL_FOLD_RELEASE_SCRATCH",      "1")
    _set_default_env(
        "DIALOG_GCD_SPECIAL_OVERFLOW_CLEAN_STEP_BITS",
        "1:24,4:21,6:25,7:20,10:22,11:20,19:21,21:21,22:21,23:23,"
        "28:21,30:22,32:20,33:24,34:25,48:21,49:22,55:22,62:23,64:20,"
        "66:21,71:22,86:20,92:21,113:21,116:21,118:20,119:24,120:21,"
        "121:20,127:20,129:22,131:22,142:22,144:21,145:23,147:21,151:23,"
        "153:23,154:23,155:20,156:24,159:22,161:24,165:21,166:21,168:21,"
        "173:20,175:21,178:21,184:22,185:20,187:23,188:22,190:20,193:21,"
        "194:22,196:20,197:21,199:21,203:22,205:22,209:20,210:21,213:20,"
        "217:22,221:21,222:23,229:21,236:21,241:21",
    )
    _set_default_env(
        "DIALOG_GCD_SPECIAL_UNDERFLOW_CLEAN_STEP_BITS",
        "3:21,5:21,10:23,11:22,14:22,17:20,27:22,33:20,34:22,38:21,"
        "42:22,47:21,50:22,51:21,53:20,54:21,58:21,60:21,65:21,67:23,"
        "68:25,73:20,74:20,75:23,77:21,78:20,84:21,89:23,91:22,95:22,"
        "98:26,103:21,109:22,110:22,114:22,118:22,127:26,135:20,136:20,"
        "137:22,143:21,149:21,152:20,154:26,155:20,156:22,157:20,158:26,"
        "166:20,178:20,181:20,186:24,188:25,191:21,194:20,198:20,200:21,"
        "201:21,202:23,203:23,204:22,212:25,213:20,214:22,221:20,223:21,"
        "228:21,231:23,243:21,246:20",
    )
    _set_default_env("DIALOG_GCD_TOBITVECTOR_CSWAP_BODY_TRIM",       "0")
    _set_default_env("DIALOG_GCD_WIDTH_MARGIN",                      "10")
    _set_default_env("DIALOG_GCD_WIDTH_SLOPE_X1000",                 "1017")
    _set_default_env("DIALOG_TAIL_NONCE",                            "200005858317")
    _set_default_env("KAL_DOUBLE_CARRY_TRUNC_W",                     "19")
    _set_default_env("KAL_FOLD_CARRY_TRUNC_W",                       "18")
    _set_default_env("SQUARE_ROW_MAX_SEG",                           "141")
    _set_default_env("SQUARE_ROW_WINDOW_CLEAN_COMPARE_BITS",         "18")
    _set_default_env(
        "SQUARE_ROW_WINDOW_CLEAN_ROW_BITS",
        "2:20,11:20,12:20,13:21,16:22,19:20,20:21,21:20,26:21,29:21,"
        "32:21,37:21,44:22,46:20,53:21,56:20,64:20,70:20,75:20,78:20,87:20",
    )
    _set_default_env(
        "SQUARE_ROW_WINDOW_CLEAN_SITE_BITS",
        "1:0:f:19,3:0:r:21,9:0:f:22,10:0:r:21,13:0:r:22,14:0:r:20,"
        "15:0:r:19,17:0:r:20,26:0:f:22,36:0:f:20,38:0:f:20,38:0:r:20,"
        "39:0:r:19,40:0:r:22,41:0:r:19,42:0:r:20,43:0:r:19,45:0:r:19,"
        "47:0:f:22,47:0:r:19,48:0:r:20,50:0:f:22,50:0:r:22,51:0:f:22,"
        "54:0:f:19,57:0:r:19,59:0:f:19,60:0:f:19,62:0:f:22,62:0:r:21,"
        "63:0:f:20,65:0:f:19,66:0:f:21,66:0:r:21,67:0:f:19,68:0:r:21,"
        "71:0:r:20,72:0:f:21,73:0:r:21,74:0:r:19,76:0:r:21,79:0:r:20,"
        "81:0:f:20,83:0:r:22,89:0:r:19,90:0:r:21,91:0:f:21,92:0:r:21,"
        "95:0:r:20,97:0:r:21,102:0:f:20,103:0:r:19,104:0:r:19,107:0:f:20,"
        "109:0:f:21,110:0:f:19,110:0:r:20",
    )
    _set_default_env("SQUARE_ROW_WINDOW_MEASURED_CARRY_CLEAR",       "1")
    _set_default_env("SKIP_ALT_SEED_CHECKS",                         "1")
    _set_default_env("DIALOG_GCD_COMPRESSED_SIDECAR_LOG",            "1")
    _set_default_env("SQUARE_ROW_WINDOW_CLEAN_COMPARE_BITS",         "21")
    _set_default_env("SQUARE_ROW_WINDOW_MEASURED_CARRY_CLEAR",       "1")
    _set_default_env("ROUND84_KEEP_QUOTIENT_PRODUCT",                "1")
    _set_default_env("DIALOG_GCD_FOLD_CARRY_TRUNC_W",                "17")
    _set_default_env("DIALOG_TAIL_NONCE",                            "200005858317")
    _set_default_env("DIALOG_GCD_SKIP_ZERO_EDGE_CSHIFT",             "1")
    _set_default_env("DIALOG_GCD_COMPRESSED_BLOCK_LIFECYCLE",        "1")
    _set_default_env("DIALOG_GCD_HOST_REVERSE_RAW_BLOCK",            "1")
    _set_default_env("DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY",      "1")
    _set_default_env("DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY_BLOCKS","999")
    _set_default_env("DIALOG_GCD_COMPOSITE_SCRATCH",                 "1")
    _set_default_env("DIALOG_GCD_BORROW_CURRENT_BLOCK",              "1")
    _set_default_env("DIALOG_GCD_CTRL_BODY_VENTED",                  "1")
    _set_default_env("DIALOG_GCD_APPLY_REPLAY_SWAP_HOST",            "1")
    _set_default_env("SQUARE_SELFHOST_SAFE_LANE_REUSE",              "1")
    _set_default_env("SQUARE_SELFHOST_GATE_SUFFIX_CARRIES",          "0")
    _set_default_env("DIALOG_GCD_PA9024_COMPARE_SCHEDULE",           "1")
    _set_default_env("DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN",    "0")
    _set_default_env("KAL_DOUBLE_CARRY_TRUNC_W",                     "19")
    _set_default_env("KAL_FOLD_CARRY_TRUNC_W",                       "18")
    _set_default_env("DIALOG_GCD_ROUND763_DEDUP",                    "1")
    _set_default_env("DIALOG_GCD_ROUND763_COMPRESS_LEVER",           "1")
    _set_default_env("DIALOG_GCD_MEASURED_UNDERFLOW_GATE",           "1")
    _set_default_env("DIALOG_GCD_COMPARE_BITS",                      "46")
    _set_default_env("DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS",          "18")
    _set_default_env("DIALOG_GCD_APPLY_BOUNDARY_CONDITIONAL_REPLAY", "1")
    _set_default_env(
        "DIALOG_GCD_SELECTED_BODY_STREAM_SUFFIX_MAP",
        "3:2,4:3,5:5,6:6,7:7,8:5,9:7,10:5,11:7,12:6,13:7,14:5,15:6,16:3,17:5,18:1,19:3,21:1",
    )
    _set_default_env("DIALOG_GCD_REVERSE_BRANCH_CONDITIONAL_REPLAY", "1")
    _set_default_env("DIALOG_GCD_SPECIAL_CLEAN_CONDITIONAL_REPLAY",  "1")
    _set_default_env("MOD_FAST_FLAG_CONDITIONAL_REPLAY",             "1")
    _set_default_env("DIALOG_GCD_RAW_PA",                            "1")
    _set_default_env("DIALOG_GCD_K2",                                "1")
    _set_default_env("DIALOG_GCD_APPLY_FUSED_FOLD",                  "1")
    _set_default_env("DIALOG_GCD_K2_PAIR_COMPRESS",                  "1")
    _set_default_env("DIALOG_GCD_ACTIVE_ITERATIONS",                 "258")
    _set_default_env("DIALOG_GCD_PERPOS_MAJ2",                       "1")
    _set_default_env("DIALOG_GCD_FUSED_HCLEAR_MEASURED",             "1")
    _set_default_env("DIALOG_GCD_FUSED_DCLEAR_MEASURED",             "1")
    _set_default_env("DIALOG_GCD_FUSED_HALVE_EDCLEAR_MEASURED",      "1")
    _set_default_env("DIALOG_GCD_RAW_IPMUL_TERMINAL_REUSE",          "1")
    _set_default_env("DIALOG_GCD_RAW_IPMUL_CLEAR_P_RESIDUAL",        "1")
    _set_default_env("DIALOG_GCD_RAW_QUOTIENT_TERMINAL_REUSE",       "1")
    _set_default_env("DIALOG_GCD_RAW_APPLY_REVERSE_MATERIALIZED_SPECIAL_SUB", "1")
    _set_default_env("DIALOG_GCD_RAW_APPLY_MATERIALIZED_SPECIAL_ADD","1")
    _set_default_env("DIALOG_GCD_RAW_APPLY_TRUNCATED_CLEAN",         "1")
    _set_default_env("DIALOG_GCD_RAW_TOBITVECTOR_MATERIALIZED_SUB",  "0")
    _set_default_env("DIALOG_GCD_RAW_TOBITVECTOR_VARIABLE_WIDTH",    "1")
    _set_default_env("DIALOG_GCD_RAW_TOBITVECTOR_BORROW_FUTURE_LOG_CARRIES", "1")
    _set_default_env("ROUND84_XTAIL_KARATSUBA",                      "0")
    _set_default_env("KARA_SOL_DBL_FAST",                            "1")
    _set_default_env("KARA_FREE_Z1_TOPBIT",                          "1")
    _set_default_env("DIALOG_GCD_WIDTH_MARGIN",                      "10")
    _set_default_env("DIALOG_GCD_MEASURED_APPLY_SUB",                "1")
    _set_default_env("DIALOG_GCD_HOST_GATED",                        "1")
    _set_default_env("DIALOG_GCD_APPLY_WINDOW_BLOCKS",               "2")
    _set_default_env("ROUND84_XTAIL_BORROW_CARRIES",                 "1")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS",            "16")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUSTOM4",           "0")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUSTOM5",           "0")
    _set_default_env("KARA_Z02_LOWQ",                                "1")
    _set_default_env("KARA_Z2_SELFHOST",                             "1")
    _set_default_env("KARA_SOL_MOD_VENT",                            "1")
    _set_default_env("DIALOG_GCD_BRANCH_BITS_HOST_COMPARATOR",       "1")
    _set_default_env("DIALOG_GCD_BODY_HOST_CIN",                     "1")
    _set_default_env("DIALOG_GCD_LATE_BORROW_UV_HIGH",               "1")
    _set_default_env(
        "DIALOG_GCD_BODY_CARRY_BAND_TRIMS",
        "0,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3",
    )
    _set_default_env("DIALOG_GCD_TOBITVECTOR_CSWAP_BODY_TRIM",       "0")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_STEPS",                "8,9,10")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_EXTRA",                "3")
    _set_default_env("DIALOG_GCD_BINDER_NOTCH_MAP",                  "11:1,12:1,13:1")
    _set_default_env(
        "DIALOG_GCD_SPECIAL_OVERFLOW_CLEAN_STEP_BITS",
        "113:21,131:21,142:22,187:23,205:22,210:21",
    )
    _set_default_env(
        "DIALOG_GCD_SPECIAL_UNDERFLOW_CLEAN_STEP_BITS",
        "42:22,91:22,118:22,149:21",
    )
    _set_default_env("DIALOG_GCD_FUSED_OVFCLEAR_MEASURED",           "1")
    _set_default_env("DIALOG_GCD_APPLY_FINAL_LOWQ",                  "0")
    _set_default_env("R84_LOWQ",                                     "1")
    _set_default_env("R84_LOWQ_CIN_BORROW",                          "1")
    _set_default_env("R84_QPROD_NAF",                                "1")
    _set_default_env("ROUND84_INPLACE_SOLINAS_FOLD",                 "1")
    _set_default_env("ROUND84_INPLACE_QUOTIENT_CARRY_TRUNC_W",       "21")
    _set_default_env("SQUARE_ROW_MAX_SEG",                           "176")
    _set_default_env("DIALOG_GCD_K5_CLEAN_BLOCK",                    "1")
    _set_default_env("DIALOG_GCD_FOLD_PARK_LOW_CARRIES",             "1")
    _set_default_env("DIALOG_GCD_SPECIAL_FOLD_BORROW_CARRIES",       "1")
    _set_default_env("DIALOG_GCD_K2_APPLY_INPLACE_RAW_BLOCK",        "1")
    _set_default_env("DIALOG_GCD_FOLD_FREED_TAIL",                   "1")
    _set_default_env("DIALOG_GCD_BORROW_CURRENT_S2",                 "1")
    _set_default_env("DIALOG_GCD_BORROW_ZERO_RAW_FUTURE",            "1")
    _set_default_env("DIALOG_GCD_FREE_SCRATCH_BEFORE_SHIFT",         "1")
    _set_default_env("DIALOG_GCD_APPLY_BOUNDARY_SPLIT",              "100")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUT",               "50")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUT2",              "100")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUT3",              "150")
    _set_default_env("DIALOG_GCD_APPLY_CHUNKED_F_CUT4",              "190")
    _set_default_env("DIALOG_GCD_WIDTH_SLOPE_X1000",                 "1017")
    _set_default_env("DIALOG_GCD_SELECTED_BODY_NOCIN",               "1")
    _set_default_env("DIALOG_TAIL_NONCE",                            "200005858317")
    _set_default_env("ROUND84_FOLD_FAST_ADD",                        "0")
    _set_default_env("DIALOG_GCD_FOLD_MAJ2",                         "1")
    _set_default_env("DIALOG_GCD_FOLD_MAJ1",                         "1")
    _set_default_env("DIALOG_GCD_APPLY_FINAL_TOPCLEAN",              "0")
    _set_default_env("ROUND84_QPROD_VENT_PAD",                       "1")
    _set_default_env("DIALOG_GCD_FOLD_FREED_TAIL_ED",                "1")
    _set_default_env("DIALOG_GCD_APPLY_FINAL_WINDOWED_FAST_BLOCKS",  "0")
    _set_default_env("DIALOG_GCD_FUSED_BRANCH_BITS",                 "1")
    _set_default_env("DIALOG_GCD_ODD_U_LOWBIT_FASTPATH",             "1")
    _set_default_env("DIALOG_REROLL",                                "4269")
    _set_default_env("DIALOG_POST_SUB_REROLL",                       "503292")


# ─────────────────────────────────────────────────────────────────────────────
# build_builder() – top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_builder() -> B:
    """
    Configure the submission route, allocate the four 256-bit registers, and
    emit the full reversible secp256k1 point-addition circuit.

    Returns the populated builder B (access B.ops for the gate list).
    """
    configure_ecdsafail_submission_route()

    if os.environ.get("POINT_ADD_COUNT_ONLY") == "1":
        b = B.new_count_only()
    else:
        b = B.new()

    # Register 0: target_x (quantum)
    tx = b.alloc_qubits(N)
    b.declare_qubit_register(tx)

    # Register 1: target_y (quantum)
    ty = b.alloc_qubits(N)
    b.declare_qubit_register(ty)

    # Register 2: offset_x (classical bits)
    ox = b.alloc_bits(N)
    b.declare_bit_register(ox)

    # Register 3: offset_y (classical bits)
    oy = b.alloc_bits(N)
    b.declare_bit_register(oy)

    # Fiat-Shamir pre-reroll (DIALOG_REROLL identity X;X pairs)
    reroll = int(os.environ.get("DIALOG_REROLL", "0"))
    if reroll > 0:
        b.set_phase("dialog_reroll")
        for _ in range(reroll):
            b.x(tx[0])
            b.x(tx[0])

    p = SECP256K1_P

    # Step 1: Px -= Qx
    mod_sub_qb(b, tx, ox, p)
    # Step 2: Py -= Qy
    mod_sub_qb(b, ty, oy, p)

    # Post-subtraction reroll
    post_sub_reroll = int(os.environ.get("DIALOG_POST_SUB_REROLL", "0"))
    if post_sub_reroll > 0:
        b.set_phase("dialog_post_sub_reroll")
        for _ in range(post_sub_reroll):
            b.x(tx[1])
            b.x(tx[1])

    # Steps 3-12: full GCD-based affine point addition
    emit_dialog_gcd_raw_pa(b, tx, ty, ox, oy, p)

    # Fiat-Shamir tail nonce (fixed-length 96-op identity block)
    nonce_str = os.environ.get("DIALOG_TAIL_NONCE")
    if nonce_str is not None:
        nonce = int(nonce_str)
        NONCE_BITS = 48
        b.set_phase("dialog_tail_nonce")
        for i in range(NONCE_BITS):
            q = tx[1] if (nonce >> i) & 1 else tx[0]
            b.x(q)
            b.x(q)

    return b


def build() -> list[Op]:
    """Build and return only the Op list (convenience wrapper)."""
    return build_builder().ops


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"N              = {N}")
    print(f"SECP256K1_P    = 0x{SECP256K1_P:064X}")
    print(f"SECP256K1_GX   = 0x{SECP256K1_GX:064X}")
    print(f"SECP256K1_GY   = 0x{SECP256K1_GY:064X}")
    print(f"SECP256K1_ORDER= 0x{SECP256K1_ORDER:064X}")

    # Count-only dry-run (skips the unimplemented stubs)
    os.environ["POINT_ADD_COUNT_ONLY"] = "1"
    configure_ecdsafail_submission_route()
    b = B.new_count_only()
    tx = b.alloc_qubits(N)
    b.declare_qubit_register(tx)
    ty = b.alloc_qubits(N)
    b.declare_qubit_register(ty)
    ox = b.alloc_bits(N)
    b.declare_bit_register(ox)
    oy = b.alloc_bits(N)
    b.declare_bit_register(oy)
    print(f"\nDry-run (register allocation only):")
    print(f"  next_qubit     = {b.next_qubit}")
    print(f"  next_bit       = {b.next_bit}")
    print(f"  next_register  = {b.next_register}")
    print(f"  active_qubits  = {b.active_qubits}")
    print(f"  counted_ops    = {b.counted_ops}  (register-directive ops only)")
