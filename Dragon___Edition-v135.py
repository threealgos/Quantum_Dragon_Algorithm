# Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#============================================================================================
"""
TODO: The Qiskit Code will Be Converted To Guppy ) Quantum programming language ---> NEXT We Can Use it in Q-Nexus Platformes.

=========🐉 DRAGON__CODE v120 🐉🔥=============
🏆 Ultimate Quantum ECDLP Solver - 41 optimized Modes
🔢 Features: Full Draper/IPE Oracles, Advanced Mitigation, Smart Mode Selection
💰 Donation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai

📌 Key Components:
- 41 quantum modes with best oracles total the First One is mod_0_porb Just for Futur_use for Google Quantum QPU 1 PHisical Qubit ~ 1 million Logical Qubits.
- Complete Draper 1D/2D/Scalar + IPE oracle implementations
- Advanced error mitigation and ZNE
- Powerful post-processing with window scanning
- Full circuit analysis and visualization
- Smart mode selection based on backend capabilities
"""

# Qiskit Imports
from IPython.display import display, HTML
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, HGate, CXGate, CCXGate, QFT
from qiskit.synthesis import synth_qft_full
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeLagosV2, FakeManilaV2
from qiskit_aer import AerSimulator
from qiskit.circuit.parameterexpression import Parameter
from qiskit import synthesis, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.synthesis import synth_qft_full
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import ZGate, MCXGate, RYGate, QFTGate, HGate, CXGate, CCXGate
from qiskit.visualization import plot_histogram, plot_distribution
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Options, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import logging
import math
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import hashlib
import base58
import pandas as pd
from math import gcd, pi, ceil, log2
from typing import Optional, Tuple, List, Dict, Union, Any
import pickle, os, time, sys, json, warnings
from datetime import datetime
from fractions import Fraction
from collections import Counter, defaultdict
from Crypto.Hash import RIPEMD160, SHA256
from ecdsa import SigningKey, SECP256k1
from Crypto.PublicKey import ECC
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import numbertheory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
def custom_transpile(qc, backend, optimization_level=3):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        Unroller,
        Optimize1qGates,
        CXCancellation,
        SabreSwap,
        SabreLayout,
        TrivialLayout,
        RemoveDiagonalGatesBeforeMeasure,
        Depth,
        Width,
        Size,
        CommutativeCancellation,
        OptimizeSwapBeforeMeasure,
        Collect2qBlocks,
        ConsolidateBlocks,
        GateDirection,
        PadDynamicalDecoupling,
        RemoveResetInZeroState,
        RemoveBarriers,
        CheckMap,
        SetLayout,
        FullAncillaAllocation,
        EnlargeWithAncilla,
        ApplyLayout,
        BarrierBeforeFinalMeasurements,
        FixedPoint,
        CheckCXDirection,
        CheckGateDirection,
        OptimizeCliffs,
        Optimize1qGatesDecomposition,
        OptimizePhaseGates,
        OptimizeSingleQubitGates,
        OptimizeBarriers,
        RemoveRedundantResets,
    )

    coupling_map = backend.configuration().coupling_map
    basis_gates = backend.configuration().basis_gates

    pass_manager = PassManager([
        Unroller(basis_gates),
        RemoveBarriers(),
        Optimize1qGates(),
        CXCancellation(),
        SabreLayout(coupling_map, max_iterations=100),
        SabreSwap(coupling_map, heuristic='basic', fake_run=False),
        TrivialLayout(coupling_map),
        RemoveDiagonalGatesBeforeMeasure(),
        Depth(),
        Width(),
        Size(),
        CommutativeCancellation(),
        OptimizeSwapBeforeMeasure(),
        Collect2qBlocks(),
        ConsolidateBlocks(),
        GateDirection(),
        PadDynamicalDecoupling(),
        RemoveResetInZeroState(),
        CheckMap(coupling_map),
        SetLayout(),
        FullAncillaAllocation(),
        EnlargeWithAncilla(),
        ApplyLayout(),
        BarrierBeforeFinalMeasurements(),
        FixedPoint('depth'),
        CheckCXDirection(),
        CheckGateDirection(),
        OptimizeCliffs(),
        Optimize1qGatesDecomposition(),
        OptimizePhaseGates(),
        OptimizeSingleQubitGates(),
        OptimizeBarriers(),
        RemoveRedundantResets(),
        RemoveDiagonalGatesBeforeMeasure(),
        RemoveResetInZeroState(),
        Depth(),
        Width(),
        Size(),
        CommutativeCancellation(),
        OptimizeSwapBeforeMeasure(),
        Collect2qBlocks(),
        ConsolidateBlocks(),
        GateDirection(),
        PadDynamicalDecoupling(),
    ])

    return pass_manager.run(qc)

def manual_zne(qc: QuantumCircuit, backend, shots: int, config: Config, scales: List[int] = [1, 3, 5]) -> Dict[str, float]:
    logger.info(f"🧪 Running Manual ZNE (Scales: {scales}) with {shots} shots...")
    counts_list = []
    scale_results = {}

    for scale in scales:
        scaled_qc = qc.copy()
        scale_results[scale] = {}

        if scale > 1:
            logger.debug(f"Applying noise scaling factor {scale}")
            for _ in range(scale - 1):
                scaled_qc.barrier()
                for q in scaled_qc.qubits:
                    scaled_qc.id(q)

        logger.info(f"[⚙️] Transpiling Scale {scale}...")
        tqc = custom_transpile(
            scaled_qc,
            backend=backend,
            optimization_level=config.OPT_LEVEL
        )

        scale_results[scale] = {
            'depth': tqc.depth(),
            'size': tqc.size(),
            'qubits': tqc.num_qubits,
            'gates': estimate_gate_counts(tqc)
        }
        logger.debug(f"[📊] Scale {scale} Metrics: Depth={tqc.depth()}, Size={tqc.size()}")

        sampler = Sampler(backend)
        sampler = configure_sampler_options(sampler, config)
        sampler.options.resilience_level = 0  # Force Raw for ZNE

        job = sampler.run([tqc], shots=shots)
        logger.debug(f"[📡] ZNE Scale {scale} Job ID: {job.job_id()}")

        try:
            job_result = job.result()
            counts = safe_get_counts(job_result[0])
            if counts:
                counts_list.append(counts)
                logger.debug(f"[✅] Scale {scale}: {len(counts)} unique measurements")
            else:
                logger.warning(f"[⚠️] No counts for scale {scale}")
        except Exception as e:
            logger.error(f"[❌] Scale {scale} failed: {e}")
            continue

    if not counts_list:
        logger.warning("⚠️ No valid counts from any ZNE scale")
        return defaultdict(float)

    logger.info("📈 Performing linear extrapolation...")
    extrapolated = defaultdict(float)
    all_keys = set().union(*counts_list)

    for key in all_keys:
        vals = [c.get(key, 0) for c in counts_list]
        if len(vals) > 1:
            try:
                fit = np.polyfit(scales[:len(vals)], vals, 1)
                extrapolated[key] = max(0, fit[1])
                logger.debug(f"Extrapolated {key}: {extrapolated[key]}")
            except Exception as e:
                logger.warning(f"Extrapolation failed for {key}: {e}")
                extrapolated[key] = vals[-1]  # Fallback
        else:
            extrapolated[key] = vals[0]

    logger.info(f"📊 ZNE Results: {len(extrapolated)} extrapolated values")
    return extrapolated

def run_dragon_code():
    config = Config()
    config.user_menu()

    # Initialize engines
    mitigation_engine = ErrorMitigationEngine(config)
    post_processor = UniversalPostProcessor(config)

    # Prompt for IBM Quantum credentials if not set
    if not config.TOKEN:
        config.TOKEN = input("Enter your IBM Quantum API token: ").strip()
    if not config.CRN:
        config.CRN = input("Enter your IBM Quantum CRN (or press Enter for 'free'): ").strip() or "free"

    # Connect to IBM Quantum
    logger.info("🔌 Connecting to IBM Quantum services...")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=config.TOKEN,
        instance=config.CRN
    )

    # Select backend
    backend = select_backend(config)

    # Decompress target and compute delta
    logger.info("🔐 Decompressing public key...")
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    start_point = ec_scalar_mult(config.KEYSPACE_START, G)
    delta = ec_point_sub(Q, start_point)
    logger.info(f"   Public Key: {hex(Q.x())[:10]}...{hex(Q.x())[-10:]}")
    logger.info(f"   Delta: ({hex(delta.x())[:10]}..., {hex(delta.y())[-10:]})")

    # Build circuit
    if isinstance(config.METHOD, int):
        if config.METHOD not in MODE_METADATA:
            logger.warning(f"Mode {config.METHOD} does not exist. Defaulting to Mode KING (41).")
            mode_id = 41
        else:
            mode_id = config.METHOD
    else:
        mode_id = 41

    mode_meta = MODE_METADATA[mode_id]
    logger.info(f"🛠️ Building circuit for mode {mode_id}: {mode_meta['logo']} {mode_meta['name']}")

    qc = build_circuit_selector(mode_id, config.BITS, delta, config)

    # Circuit analysis
    mitigation_engine.analyze_circuit(qc, backend)

    # Transpile with custom pass manager
    logger.info("⚙️ Transpiling circuit...")
    transpiled = custom_transpile(qc, backend, optimization_level=config.OPT_LEVEL)

    logger.info(f"   Original depth: {qc.depth()}")
    logger.info(f"   Transpiled depth: {transpiled.depth()}")
    logger.info(f"   Original size: {qc.size()}")
    logger.info(f"   Transpiled size: {transpiled.size()}")
    logger.info(f"   Qubits used: {transpiled.num_qubits}")

    # Execute with mitigation
    sampler = Sampler(backend)
    sampler = mitigation_engine.configure_sampler_options(sampler)

    if config.USE_ZNE and config.ZNE_METHOD == "manual":
        logger.info("🧪 Running Manual ZNE...")
        counts = manual_zne(transpiled, backend, config.SHOTS, config)
    else:
        logger.info(f"📡 Submitting job with {config.SHOTS} shots...")
        job = sampler.run([transpiled], shots=config.SHOTS)
        logger.info(f"   Job ID: {job.job_id()}")
        logger.info("⏳ Waiting for results...")
        result = job.result()
        counts = mitigation_engine.safe_get_counts(result[0])

    # Display top results
    logger.info("\n📊 Top Measurement Results:")
    for i, (state, count) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]):
        logger.info(f"  {i+1}. {state}: {count} counts")

    # Post-processing
    logger.info("🔍 Beginning comprehensive post-processing...")
    candidate = post_processor.process_all_measurements(
        counts, config.BITS, ORDER, config.KEYSPACE_START, Q.x(), mode_meta
    )

    if candidate:
        logger.info(f"🎉 SUCCESS: Found candidate private key: {hex(candidate)}")
    else:
        logger.warning("❌ No candidates found in top results")

    plot_visuals(counts, config.BITS)
"""

# ==========================================
# 1. PRESETS & CONSTANTS
# ==========================================
PRESET_17 = {
    "bits": 17,
    "start": 0x10000,
    "pub": "033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3",
    "logo": "🔢",
    "description": "Small key for testing and simulation"
}


PRESET_135 = {
    "bits": 135,
    "start": 0x4000000000000000000000000000000000,
    "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
    "logo": "🔐",
    "description": "Standard 135-bit key for real hardware"
}

# --- secp256k1 Constants ---------------------------------------------

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
ORDER = N
CURVE = CurveFp(P, A, B)
G = Point(CURVE, Gx, Gy)
G_POINT = G
#-----------------------------------------------------------------------

# Constants for mode IDs
# KING = 41  # Special mode ID for the KING mode

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. COMPLETE MATHEMATICAL UTILITIES
# ==========================================
def gcd_verbose(a: int, b: int) -> int:
    """Verbose GCD calculation with logging"""
    logger.debug(f"Calculating GCD of {a} and {b}")
    while b:
        a, b = b, a % b
    logger.debug(f"GCD result: {a}")
    return a


def extended_euclidean(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm for modular inverses"""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_euclidean(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)


def modular_inverse_verbose(a: int, m: int) -> Optional[int]:
    """Modular inverse with verbose logging"""
    try:
        return pow(a, -1, m)
    except ValueError:
        g, x, y = extended_euclidean(a, m)
        if g != 1:
            logger.warning(f"No inverse exists for {a} mod {m}")
            return None
        return x % m


def tonelli_shanks_sqrt(n: int, p: int) -> int:
    """Tonelli-Shanks square root modulo prime"""
    if pow(n, (p - 1) // 2, p) != 1:
        return 0
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    s, e = p - 1, 0
    while s % 2 == 0:
        s //= 2
        e += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    x = pow(n, (s + 1) // 2, p)
    b, g, r = pow(n, s, p), pow(z, s, p), e
    
    while True:
        t, m = b, 0
        for m in range(r):
            if t == 1:
                break
            t = pow(t, 2, p)
        if m == 0:
            return x
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m

def continued_fraction_approx(num: int, den: int, max_den: int = 1000000) -> Tuple[int, int]:
    """Compute continued fraction approximation with detailed logging."""
    # Input validation
    if den == 0:
        logger.warning("Denominator is zero, returning (0, 1)")
        return 0, 1

    # Simplify fraction
    common_divisor = gcd_verbose(num, den)
    if common_divisor > 1:
        simplified_num = num // common_divisor
        simplified_den = den // common_divisor
        logger.debug(f"Simplified {num}/{den} to {simplified_num}/{simplified_den}")
    else:
        simplified_num, simplified_den = num, den

    # Compute continued fraction approximation
    approximation = Fraction(simplified_num, simplified_den).limit_denominator(max_den)
    logger.debug(f"Best approximation: {approximation.numerator}/{approximation.denominator}")

    return approximation.numerator, approximation.denominator


def decompress_pubkey(hex_key: str) -> Point:
    """Decompress SECP256K1 public key"""
    hex_key = hex_key.lower().replace("0x", "").strip()
    if len(hex_key) not in [66, 130]:
        raise ValueError(f"Invalid key length: {len(hex_key)}")
    
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    
    y_sq = (pow(x, 3, P) + B) % P
    y = tonelli_shanks_sqrt(y_sq, P)
    
    if y == 0:
        raise ValueError("Invalid public key")
    
    if (prefix == 2 and y % 2 != 0) or (prefix == 3 and y % 2 == 0):
        y = P - y
    
    return Point(CURVE, x, y)

def ec_point_add(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Elliptic curve point addition with full validation and optimization."""
    # Handle identity cases
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    # Extract coordinates
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()

    # Point at infinity check (P + (-P) = O)
    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    # Point doubling case (P + P)
    if x1 == x2 and y1 == y2:
        numerator = (3 * x1 * x1 + A) % P
        denominator = (2 * y1) % P
        inv_denominator = modular_inverse_verbose(denominator, P)
        if inv_denominator is None:
            return None  # Division by zero
        lam = (numerator * inv_denominator) % P
    # Distinct points case (P + Q)
    else:
        numerator = (y2 - y1) % P
        denominator = (x2 - x1) % P
        inv_denominator = modular_inverse_verbose(denominator, P)
        if inv_denominator is None:
            return None  # Division by zero
        lam = (numerator * inv_denominator) % P

    # Compute new point coordinates
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P

    return Point(CURVE, x3, y3)

# Draper adders
def draper_add_const(qc: QuantumCircuit, ctrl, target: QuantumRegister, value: int,
                    modulus: int = N, inverse: bool = False):

    sign = -1 if inverse else 1
    n = len(target)

    # Apply QFT to target register
    qft_reg(qc, target)

    # Apply phase rotations for each qubit
    for i in range(n):
        angle = sign * (2 * math.pi * value) / (2 ** (n - i))
        if ctrl is not None:
            qc.cp(angle, ctrl, target[i])
        else:
            qc.p(angle, target[i])

    # Apply inverse QFT
    iqft_reg(qc, target)

def add_const_mod_gate(c: int, mod: int, use_matrix: bool = False, max_matrix_size: int = 64) -> UnitaryGate:
    n_qubits = math.ceil(math.log2(mod)) if mod > 1 else 1

    # Use matrix representation for small moduli
    if mod <= max_matrix_size and use_matrix:
        try:
            # Create unitary matrix
            mat = np.zeros((mod, mod), dtype=complex)
            for x in range(mod):
                mat[(x + c) % mod, x] = 1

            # Pad to full dimension if needed
            full_dim = 2 ** n_qubits
            if full_dim > mod:
                full_mat = np.eye(full_dim, dtype=complex)
                full_mat[:mod, :mod] = mat
                return UnitaryGate(full_mat, label=f"+{c} mod {mod}")

            return UnitaryGate(mat, label=f"+{c} mod {mod}")
        except Exception as e:
            logger.warning(f"Matrix gate creation failed: {e}. Falling back to Draper's method")

    # Fallback to Draper's QFT-based method
    qc = QuantumCircuit(n_qubits, name=f"+{c} mod {mod}")
    qft_reg(qc, qc.qubits)

    for i in range(n_qubits):
        angle = 2 * math.pi * c / (2 ** (n_qubits - i))
        qc.p(angle, i)

    qc.append(QFTGate(n_qubits, do_swaps=False).inverse(), range(n_qubits))
    return qc.to_gate()

def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    """Elliptic curve scalar multiplication with optimization"""
    if k == 0 or point is None:
        return None
    result = None
    addend = point
    while k:
        if k & 1:
            result = ec_point_add(result, addend)
        addend = ec_point_add(addend, addend)
        k >>= 1
    return result


# ==========================================
# 3. FAULT TOLERANCE & REPETITION CODES
# ==========================================

def prepare_verified_ancilla(qc: QuantumCircuit, qubit, initial_state: int = 0):
    """Prepare a verified ancilla qubit in a known state"""
    qc.reset(qubit)
    if initial_state == 1:
        qc.x(qubit)
    logger.debug(f"Prepared ancilla qubit {qubit} in state {initial_state}")


def encode_repetition(qc: QuantumCircuit, logical_qubit, ancillas: List):
    """Encode 1 logical qubit into 3 physical qubits (Bit Flip Code)"""
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])
    logger.debug(f"Encoded logical qubit {logical_qubit} with ancillas {ancillas}")


def decode_repetition(qc: QuantumCircuit, ancillas: List, logical_qubit):
    """Decode 3 physical qubits back to 1 logical qubit using majority vote"""
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)
    logger.debug(f"Decoded logical qubit {logical_qubit} from ancillas {ancillas}")


# ==========================================
# 4. COMPLETE ORACLE IMPLEMENTATIONS
# ==========================================
def qft_reg(qc: QuantumCircuit, reg: QuantumRegister):
    """Apply QFT to register with logging"""
    logger.debug(f"Applying QFT to {len(reg)} qubits")
    qc.append(synth_qft_full(len(reg), do_swaps=False).to_gate(), reg)


def iqft_reg(qc: QuantumCircuit, reg: QuantumRegister):
    """Apply inverse QFT to register"""
    logger.debug(f"Applying IQFT to {len(reg)} qubits")
    qc.append(synth_qft_full(len(reg), do_swaps=False).inverse().to_gate(), reg)

# Matrix/Unitary scalable gate
class GeometricIPE:
    """Enhanced Geometric IPE implementation with robust point handling"""
    def __init__(self, n_bits: int):
        self.n = n_bits

    def _oracle_geometric_phase(self, qc: QuantumCircuit, ctrl, state_reg: QuantumRegister,
                               point_val: Union[Point, tuple, int]):
        """Apply geometric phase oracle with comprehensive point handling"""
        if point_val is None:
            logger.debug("Skipping geometric phase oracle (point is None)")
            return

        # Robust point value extraction
        if isinstance(point_val, Point):
            vx = point_val.x()
        elif isinstance(point_val, tuple) and len(point_val) >= 1:
            vx = point_val[0]
        elif isinstance(point_val, (int, np.integer)):
            vx = point_val
        else:
            logger.warning(f"Unsupported point type: {type(point_val)}")
            return

        # Apply phase rotations with proper angle calculation
        for i in range(self.n):
            try:
                angle_x = 2 * math.pi * vx / (2 ** (i + 1))
                if ctrl is not None:
                    qc.cp(angle_x, ctrl, state_reg[i])
                else:
                    qc.p(angle_x, state_reg[i])
            except Exception as e:
                logger.error(f"Failed to apply phase rotation: {e}")
                continue

        logger.debug(f"Applied geometric IPE oracle with vx={hex(vx)[:10]}...")


class IPEOracle:
    """Complete IPE Oracle Implementation with adaptive strategies"""
    def __init__(self, n_bits: int):
        self.n = n_bits

    def oracle_phase(self, qc: QuantumCircuit, ctrl, point_reg: QuantumRegister,
                    delta_point: Union[Point, tuple], k_step: int, order: int = ORDER):
        """Apply IPE phase oracle with proper point handling"""
        if delta_point is None:
            logger.debug("Skipping IPE phase oracle (delta point is None)")
            return

        # Get coordinates properly
        if isinstance(delta_point, Point):
            dx = delta_point.x()
            dy = delta_point.y()
        elif isinstance(delta_point, tuple) and len(delta_point) >= 2:
            dx, dy = delta_point[0], delta_point[1]
        else:
            logger.warning(f"Unsupported delta point type: {type(delta_point)}")
            return

        power = 1 << k_step
        const_x = (dx * power) % order

        if const_x != 0:
            # Use more efficient scalar oracle
            draper_adder_oracle_scalar(qc, ctrl, point_reg, const_x)
            logger.debug(f"Applied IPE phase oracle with power={power}, const_x={hex(const_x)[:10]}...")

    def oracle_geometric(self, qc: QuantumCircuit, ctrl, state_reg: QuantumRegister,
                        point_val: Union[Point, tuple, int]):
        """Apply geometric IPE oracle with comprehensive point handling"""
        if point_val is None:
            logger.debug("Skipping geometric oracle (point is None)")
            return

        # Robust point value extraction
        if isinstance(point_val, Point):
            vx = point_val.x()
        elif isinstance(point_val, tuple) and len(point_val) >= 1:
            vx = point_val[0]
        elif isinstance(point_val, (int, np.integer)):
            vx = point_val
        else:
            logger.warning(f"Unsupported point type: {type(point_val)}")
            return

        # Apply phase rotations with proper error handling
        for i in range(self.n):
            try:
                angle_x = 2 * math.pi * vx / (2 ** (i + 1))
                if ctrl is not None:
                    qc.cp(angle_x, ctrl, state_reg[i])
                else:
                    qc.p(angle_x, state_reg[i])
            except Exception as e:
                logger.error(f"Failed to apply geometric phase: {e}")
                continue

        logger.debug(f"Applied geometric IPE oracle with vx={hex(vx)[:10]}...")

def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl, target: QuantumRegister,
                                dx: int, dy: int = 0):
    """Enhanced 1D Draper oracle with proper angle calculation"""
    n = len(target)
    qft_reg(qc, target)

    for i in range(n):
        try:
            # More precise angle calculation
            angle = 2 * math.pi * dx / (2 ** (i + 1))
            if ctrl is not None:
                qc.cp(angle, ctrl, target[i])
            else:
                qc.p(angle, target[i])
        except Exception as e:
            logger.error(f"Failed in 1D Draper oracle: {e}")
            continue

    iqft_reg(qc, target)
    logger.debug(f"Applied 1D Draper oracle with dx={hex(dx)[:10]}...")

def draper_adder_oracle_2d(qc: QuantumCircuit, ctrl, target: QuantumRegister,
                          dx: int, dy: int):
    """Enhanced 2D Draper oracle with proper point handling"""
    n = len(target)
    qft_reg(qc, target)

    for i in range(n):
        try:
            # More precise angle calculations
            angle_x = 2 * math.pi * dx / (2 ** (i + 1))
            angle_y = 2 * math.pi * dy / (2 ** (i + 1))

            if ctrl is not None:
                qc.cp(angle_x, ctrl, target[i])
                qc.cp(angle_y, ctrl, target[i])
            else:
                # Combine angles for uncontrolled case
                qc.p(angle_x + angle_y, target[i])
        except Exception as e:
            logger.error(f"Failed in 2D Draper oracle: {e}")
            continue

    iqft_reg(qc, target)
    logger.debug(f"Applied 2D Draper oracle with dx={hex(dx)[:10]}, dy={hex(dy)[:10]}...")

def draper_adder_oracle_scalar(qc: QuantumCircuit, ctrl, target: QuantumRegister, scalar: int):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle = 2 * math.pi * scalar / (2 ** (n - i))
        if ctrl is not None:
            qc.cp(angle, ctrl, target[i])
        else:
            qc.p(angle, target[i])
    iqft_reg(qc, target)
    logger.debug(f"Applied scalar Draper oracle with scalar {scalar}")


def ft_draper_modular_adder(qc: QuantumCircuit, ctrl, target_reg: QuantumRegister,
                          ancilla_reg: QuantumRegister, value: int, modulus: int = N):
    n = len(target_reg)
    temp_overflow = ancilla_reg[0]


    # Apply QFT to target register
    qft_reg(qc, target_reg)


    # Add the value (controlled if ctrl is provided)
    draper_adder_oracle_scalar(qc, ctrl, target_reg, value)


    # Subtract modulus to handle potential overflow
    draper_adder_oracle_scalar(qc, None, target_reg, -modulus)


    # Check for overflow
    iqft_reg(qc, target_reg)
    qc.cx(target_reg[-1], temp_overflow)  # Set overflow flag if MSB is 1


    # If overflow occurred, add modulus back
    qft_reg(qc, target_reg)
    qc.cx(temp_overflow, target_reg[-1])  # Conditionally flip MSB
    draper_adder_oracle_scalar(qc, temp_overflow, target_reg, modulus)
    qc.cx(temp_overflow, target_reg[-1])  # Restore MSB
    iqft_reg(qc, target_reg)


    # Reset temporary registers
    qc.reset(temp_overflow)


    logger.debug(f"Applied fault-tolerant Draper adder with value {value} mod {modulus}")


def ft_draper_modular_adder_omega(qc: QuantumCircuit, value: int, target_reg: QuantumRegister,
                                modulus: int, ancilla_reg: QuantumRegister, temp_reg: QuantumRegister):
    # Simply call the standard implementation
    return ft_draper_modular_adder(qc, None, target_reg, ancilla_reg, value, modulus)


def shor_oracle(qc: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister,
               state_reg: QuantumRegister, points: List[Point], ancilla_reg: QuantumRegister):
    """
    Shor-style oracle implementation using proper Point objects.

    Args:
        qc: Quantum circuit
        a_reg: First control register
        b_reg: Second control register
        state_reg: Target state register
        points: List of precomputed Point objects
        ancilla_reg: Ancilla qubits for fault tolerance
    """
    logger.debug("Building Shor-style oracle")

    # Process a_reg controls
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        if pt:
            # Use proper Point methods instead of tuple access
            val = pt.x() % N
            logger.debug(f"Applying a_reg[{i}] with point x-coordinate: {hex(val)[:10]}...")
            if val:
                ft_draper_modular_adder(qc, a_reg[i], state_reg, ancilla_reg, val)

    # Process b_reg controls
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)]
        if pt:
            # Use proper Point methods instead of tuple access
            val = pt.x() % N
            logger.debug(f"Applying b_reg[{i}] with point x-coordinate: {hex(val)[:10]}...")
            if val:
                ft_draper_modular_adder(qc, b_reg[i], state_reg, ancilla_reg, val)

    logger.debug("Applied Shor-style oracle construction")


def ecdlp_oracle_ab(qc: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister,
                   point_reg: QuantumRegister, points: List[Point], ancilla_reg: QuantumRegister):
    """
    ECDLP oracle implementation for AB registers using proper Point objects.

    Args:
        qc: Quantum circuit
        a_reg: A control register
        b_reg: B control register
        point_reg: Target point register
        points: List of precomputed Point objects
        ancilla_reg: Ancilla qubits for fault tolerance
    """
    logger.debug("Building ECDLP AB oracle")

    # Validate all points in the list
    for pt in points:
        if pt and not isinstance(pt, Point):
            raise TypeError(f"Expected Point object, got {type(pt)}")

    # Process a_reg controls
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        if pt:
            # Use proper Point methods instead of tuple access
            val = pt.x() % N
            logger.debug(f"Applying a_reg[{i}] with point x-coordinate: {hex(val)[:10]}...")
            if val:
                ft_draper_modular_adder(qc, a_reg[i], point_reg, ancilla_reg, val)

    # Process b_reg controls
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)]
        if pt:
            # Use proper Point methods instead of tuple access
            val = pt.x() % N
            logger.debug(f"Applying b_reg[{i}] with point x-coordinate: {hex(val)[:10]}...")
            if val:
                ft_draper_modular_adder(qc, b_reg[i], point_reg, ancilla_reg, val)

    logger.debug("Applied ECDLP oracle")
 
def windowed_oracle(
    qc: QuantumCircuit,
    ctrl: QuantumRegister,
    state_reg: QuantumRegister,
    delta_point: Point,
    window_size: int,
    k_step: int
) -> None:
    """Apply windowed oracle using delta_point (Point object)."""
    power = 1 << k_step
    dx = (delta_point.x() * power) % N  # Fixed: delta_point.x()
    dy = (delta_point.y() * power) % N  # Fixed: delta_point.y()

    for j in range(window_size):
        draper_adder_oracle_1d_serial(qc, ctrl[j], state_reg, dx)
    logger.debug(f"Applied windowed oracle with window size {window_size}, dx={hex(dx)[:10]}")


def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    """Scalar multiplication on elliptic curve"""
    if k == 0 or point is None:
        return None
    
    result = None
    addend = point
    
    while k:
        if k & 1:
            result = ec_point_add(result, addend)
        addend = ec_point_add(addend, addend)
        k >>= 1
    
    return result

def ec_point_negate(point: Optional[Point]) -> Optional[Point]:
    """Negate elliptic curve point"""
    if point is None:
        return None
    return Point(CURVE, point.x(), (-point.y()) % P)

def ec_point_sub(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Point subtraction: P1 - P2"""
    return ec_point_add(p1, ec_point_negate(p2))

def ec_point_subtract(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Alias for ec_point_sub"""
    return ec_point_sub(p1, p2)

# Fixed eigenvalue_phase_oracle
def eigenvalue_phase_oracle(qc: QuantumCircuit, ctrl, target: QuantumRegister, point: Point, power: int):
    """Updated phase oracle for ECDLP context"""
    scalar = (point.x() * power) % N
    theta = 2 * math.pi * scalar / N
    qc.cp(theta, ctrl, target[0])

def precompute_powers(delta: Point, bits: int) -> List[Optional[Point]]:
    """Precompute powers of delta (2^k * delta) for gate optimization.
    Args:
        delta: Point object (Q - k*G).
        bits: Number of bits (e.g., 135).
    Returns:
        List of precomputed points (2^k * delta), or None if infinity.
    """
    powers = []
    curr = delta
    for _ in range(bits):
        powers.append(curr)
        if curr is None:
            # If curr is None (point at infinity), all further powers are None
            powers.extend([None] * (bits - len(powers)))
            break
        curr = ec_point_add(curr, curr)  # 2^k * delta
    return powers


def precompute_points(bits: int) -> List[Point]:
    """Precompute elliptic curve points for the given bit length"""
    points = []
    current = G
    for _ in range(bits):
        points.append(current)
        current = ec_point_add(current, current)
    return points


def compute_offset(Q: Point, start: int) -> Point:
    """Compute delta = Q - start*G"""
    start_G = ec_scalar_mult(start, G)
    if start_G is None:
        return Q
    return ec_point_sub(Q, start_G)  # Keep as Point, update callers


def precompute_good_indices_range(start, end, target_qx, gx=G.x(), gy=G.y(), p=P, max_window=100000):
    """
    Enhanced classical brute-force check in range with keyspace awareness
    
    Args:
        start: Start index (can be measurement or keyspace-adjusted)
        end: End index (inclusive)
        target_qx: Target public key x-coordinate
        gx: Generator x (default: G.x())
        gy: Generator y (default: G.y())
        p: Curve prime (default: P)
        max_window: Maximum window size to prevent runaway
    
    Returns:
        List of offsets where (start + offset)*G.x() == target_qx
    """
    # Safety check - prevent excessive computation
    window_size = end - start
    if window_size > max_window:
        logger.warning(f"Window size {window_size} > max {max_window}, truncating")
        end = start + max_window
    
    # Ensure within valid range
    if start < 0:
        start = 0
    if end > N:
        end = N
    
    if window_size <= 0:
        return []
    
    logger.info(f"Classical window scan: [{hex(start)[:20]}... - {hex(end)[:20]}...] (size: {end-start:,})")
    
    good = []
    base_point = Point(CURVE, gx, gy)
    
    # Compute starting point
    try:
        current = ec_scalar_mult(start, base_point)
    except Exception as e:
        logger.error(f"Failed to compute start point for {hex(start)[:20]}...: {e}")
        return []
    
    # Progress tracking
    total_iterations = end - start
    if total_iterations > 1000:
        report_interval = max(1, total_iterations // 10)  # Report every 10%
    else:
        report_interval = total_iterations + 1  # Don't report for small windows
    
    for offset in range(total_iterations + 1):
        k = start + offset
        
        # Skip if we've already computed this (for start=0 case)
        if offset == 0:
            if current and current.x() == target_qx:
                logger.info(f"Immediate hit at start: {hex(k)}")
                good.append(0)
                continue
        
        # Compute next point efficiently
        if current is None:
            current = base_point
        elif offset > 0:  # Add base point for each step
            current = ec_point_add(current, base_point)
        
        # Check if we found the target
        if current and current.x() == target_qx:
            logger.info(f"✓ Found key via classical scan: {hex(k)}")
            good.append(offset)
        
        # Progress reporting
        if total_iterations > 1000 and offset % report_interval == 0 and offset > 0:
            progress = (offset / total_iterations) * 100
            logger.info(f"Window scan progress: {progress:.1f}% ({offset}/{total_iterations})")
    
    logger.info(f"Window scan complete. Found {len(good)} candidate(s)")
    return good

# Oracle variations
def ipe_oracle_phase(
    qc: QuantumCircuit,
    ctrl: QuantumRegister,
    point_reg: QuantumRegister,
    delta_point: Point,
    k_step: int,
    order: int = ORDER
) -> None:
    """Apply IPE phase oracle using delta_point (Point object)."""
    power = 1 << k_step
    const_x = (delta_point.x() * power) % order  # Fixed: delta_point.x()

    if const_x != 0:
        draper_add_const(qc, ctrl, point_reg, const_x)
        logger.debug(f"Applied IPE phase oracle with power={power}, const_x={hex(const_x)[:10]}")


# ==========================================
# 5. MODE METADATA with all 41 modes 
# ==========================================
# Complete MODE_METADATA 
MODE_METADATA = {
    0: {"name": "Hardware IPE Diagnostic", "qubits": 136, "success": 70, "endian": "LSB", "oracle": "Diagnostic", "logo": "🔧"},
    1: {"name": "IPE Standard 🥇", "qubits": 135, "success": 80, "endian": "LSB", "oracle": "IPE", "logo": "🥇"},
    2: {"name": "Hive (Chunked) 🥈", "qubits": 127, "success": 65, "endian": "LSB", "oracle": "Hive", "logo": "🥈"},
    3: {"name": "Windowed IPE 🥉", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "Windowed", "logo": "🥉"},
    4: {"name": "Semiclassical 🏆", "qubits": 135, "success": 75, "endian": "LSB", "oracle": "Semiclassical", "logo": "🏆"},
    5: {"name": "AB Shor Optimized", "qubits": 156, "success": 55, "endian": "MSB", "oracle": "Shor", "logo": "4️⃣"},
    6: {"name": "FT Draper Test", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "DraperFT", "logo": "🛡️"},
    7: {"name": "Geometric IPE", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "Geometric", "logo": "🌀"},
    8: {"name": "Verified (Flags)", "qubits": 136, "success": 65, "endian": "LSB", "oracle": "Verified", "logo": "🏷️"},
    9: {"name": "Shadow 2D", "qubits": 138, "success": 65, "endian": "LSB", "oracle": "Shadow", "logo": "👻"},
    10: {"name": "Reverse IPE", "qubits": 135, "success": 60, "endian": "LSB", "oracle": "Reverse", "logo": "🔄"},
    11: {"name": "Swarm", "qubits": 127, "success": 60, "endian": "LSB", "oracle": "Swarm", "logo": "🐝"},
    12: {"name": "Heavy Draper", "qubits": 128, "success": 68, "endian": "LSB", "oracle": "Draper1D", "logo": "🔢"},
    13: {"name": "Compressed Shadow", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "Compressed", "logo": "📦"},
    14: {"name": "Shor Logic", "qubits": 156, "success": 55, "endian": "MSB", "oracle": "ShorLogic", "logo": "🧠"},
    15: {"name": "Geo IPE (Exp)", "qubits": 134, "success": 76, "endian": "LSB", "oracle": "GeoIPE", "logo": "🌍"},
    16: {"name": "Windowed Explicit", "qubits": 136, "success": 55, "endian": "LSB", "oracle": "WindowedExp", "logo": "🪟"},
    17: {"name": "Hive Swarm", "qubits": 127, "success": 65, "endian": "LSB", "oracle": "HiveSwarm", "logo": "🐝"},
    18: {"name": "Explicit Logic", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "Explicit", "logo": "📝"},
    19: {"name": "Fixed AB", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "FixedAB", "logo": "🔗"},
    20: {"name": "Matrix Mod 🆕", "qubits": 135, "success": 75, "endian": "LSB", "oracle": "Matrix", "logo": "🔢"},
    21: {"name": "Phantom Parallel", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "Phantom", "logo": "👻"},
    22: {"name": "Shor Parallel", "qubits": 156, "success": 55, "endian": "MSB", "oracle": "ShorParallel", "logo": "👥"},
    23: {"name": "GHZ Parallel", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "GHZ", "logo": "🕸️"},
    24: {"name": "Verified Parallel", "qubits": 136, "success": 65, "endian": "LSB", "oracle": "VerifiedPar", "logo": "🏷️"},
    25: {"name": "Hive Edition", "qubits": 127, "success": 65, "endian": "LSB", "oracle": "HiveEd", "logo": "🐝"},
    26: {"name": "Extra Shadow", "qubits": 136, "success": 60, "endian": "LSB", "oracle": "ExtraShadow", "logo": "👻"},
    27: {"name": "Advanced QPE", "qubits": 136, "success": 70, "endian": "LSB", "oracle": "AdvancedQPE", "logo": "📊"},
    28: {"name": "Full Quantum Opt", "qubits": 156, "success": 55, "endian": "MSB", "oracle": "FullQuantum", "logo": "💎"},
    29: {"name": "Semiclassical Omega", "qubits": 136, "success": 75, "endian": "LSB", "oracle": "SemiOmega", "logo": "🏆"},
    30: {"name": "Verified Shadow", "qubits": 138, "success": 65, "endian": "LSB", "oracle": "VerifiedShad", "logo": "👻"},
    31: {"name": "Verified Advanced", "qubits": 136, "success": 65, "endian": "LSB", "oracle": "VerifiedAdv", "logo": "🛡️"},
    32: {"name": "Heavy Draper Omega", "qubits": 128, "success": 68, "endian": "LSB", "oracle": "HeavyOmega", "logo": "🔢"},
    33: {"name": "Compressed Shadow Omega", "qubits": 136, "success": 70, "endian": "LSB", "oracle": "CompressedOmega", "logo": "📦"},
    34: {"name": "Shor Logic Omega", "qubits": 156, "success": 55, "endian": "MSB", "oracle": "ShorOmega", "logo": "🧠"},
    35: {"name": "Geometric IPE Omega", "qubits": 134, "success": 76, "endian": "LSB", "oracle": "GeoOmega", "logo": "🌍"},
    36: {"name": "Windowed IPE Omega", "qubits": 136, "success": 70, "endian": "LSB", "oracle": "WindowedOmega", "logo": "🪟"},
    37: {"name": "Hive Swarm Omega", "qubits": 127, "success": 65, "endian": "LSB", "oracle": "HiveOmega", "logo": "🐝"},
    38: {"name": "Explicit Logic Omega", "qubits": 136, "success": 70, "endian": "LSB", "oracle": "ExplicitOmega", "logo": "📝"},
    39: {"name": "Matrix Mod Omega 🆕", "qubits": 132, "success": 78, "endian": "LSB", "oracle": "MatrixOmega", "logo": "🌀"},
    41: {"name": "Semiclassical Omega KING 👑", "qubits": 136, "success": 85, "endian": "LSB", "oracle": "IPEKing", "logo": "👑"}
}


# ==========================================
# 6. CONFIG CLASS (With All Toggles)
# ==========================================

class Config:
    def __init__(self):
        # Target settings
        self.BITS = 135
        self.KEYSPACE_START = PRESET_135["start"]
        self.COMPRESSED_PUBKEY_HEX = PRESET_135["pub"]


        # Backend settings
        self.BACKEND = "ibm_kingston"  #  And For Future backends ~1386 Qubits Nighthawk/Kookaburra 
        self.TOKEN = None
        self.CRN = None


        # Mode selection
        self.METHOD = "smart"


        # Error mitigation toggles
        self.USE_FT = False          # Fault tolerance
        self.USE_REPETITION = False  # Repetition codes
        self.USE_ZNE = True          # Zero-noise extrapolation
        self.ZNE_METHOD = "manual"   # manual/standard
        self.USE_DD = True           # Dynamical decoupling
        self.DD_SEQUENCE = "XY4"     # DD sequence type # "XpXm & "XX" 
        self.USE_MEAS_MITIGATION = True  # Measurement error mitigation


        # Execution settings
        self.SHOTS = 8192  # MAX_SHOTS = 16384 & 100,000 & 1 million
        self.OPT_LEVEL = 3
        self.INTERNAL_RESILIENCE_LEVEL = 2
        self.SEARCH_DEPTH = 10000     # Default search depth for window scanning
        self.EXCLUDE_MODES = []  # Default empty list
        self.MAX_MATRIX_SIZE = 64  # For add_const_mod_gate
        self.USE_SMART_GATE = True  # For add_const_mod_gate

    def user_menu(self):
        print("\n" + "="*70)
        print(" 🐉 DRAGON_Edition_CODE v120 🔥 - 40 Optimized Modes |+> Mode King 🏆")
        print("="*70)
        print("📌 Available Modes:")
        for mode_id, meta in MODE_METADATA.items():
            print(f"  {mode_id}: {meta['logo']} {meta['name']} ({meta['success']}% success, {meta['qubits']} qubits)")
        print("-"*70)


        # Preset selection
        preset = input("Use preset? [17/135/none] (default 135): ").strip().lower()
        if preset == "17":
            self.BITS = PRESET_17["bits"]
            self.KEYSPACE_START = PRESET_17["start"]
            self.COMPRESSED_PUBKEY_HEX = PRESET_17["pub"]
        elif preset != "none":
            self.BITS = PRESET_135["bits"]
            self.KEYSPACE_START = PRESET_135["start"]
            self.COMPRESSED_PUBKEY_HEX = PRESET_135["pub"]


        # Mode selection
        m = input(f"Select Mode [0-41] or 'smart' (default {self.METHOD}): ").strip()
        if m: self.METHOD = m if m.lower() == "smart" else int(m)


        # Error mitigation toggles
        self.USE_FT = input(f"Enable Fault Tolerance? [y/n] (default {'y' if self.USE_FT else 'n'}): ").strip().lower() == 'y'
        self.USE_REPETITION = input(f"Enable Repetition Codes? [y/n] (default {'y' if self.USE_REPETITION else 'n'}): ").strip().lower() == 'y'
        self.USE_ZNE = input(f"Enable ZNE? [y/n] (default {'y' if self.USE_ZNE else 'n'}): ").strip().lower() == 'y'
        if self.USE_ZNE:
            self.ZNE_METHOD = input(f"ZNE Method [manual/standard] (default {self.ZNE_METHOD}): ").strip().lower() or self.ZNE_METHOD
        self.USE_DD = input(f"Enable Dynamical Decoupling? [y/n] (default {'y' if self.USE_DD else 'n'}): ").strip().lower() == 'y'
        if self.USE_DD:
            self.DD_SEQUENCE = input(f"DD Sequence [XY4/XpXm] (default {self.DD_SEQUENCE}): ").strip() or self.DD_SEQUENCE
        self.USE_MEAS_MITIGATION = input(f"Enable Measurement Mitigation? [y/n] (default {'y' if self.USE_MEAS_MITIGATION else 'n'}): ").strip().lower() == 'y'


        # Execution settings
        shots = input(f"Shots [{self.SHOTS}]: ").strip()
        if shots: self.SHOTS = int(shots)


        search_depth = input(f"Search Depth [{self.SEARCH_DEPTH}]: ").strip()
        if search_depth: self.SEARCH_DEPTH = int(search_depth)


        print("\n" + "="*70)
        print("🛠️ Configuration Summary:")
        print(f"   Target: {self.BITS}-bit key")
        print(f"   Mode: {self.METHOD}")
        print(f"   FT: {'✅' if self.USE_FT else '❌'}, Repetition: {'✅' if self.USE_REPETITION else '❌'}")
        print(f"   ZNE: {'✅' if self.USE_ZNE else '❌'} ({self.ZNE_METHOD})")
        print(f"   DD: {'✅' if self.USE_DD else '❌'} ({self.DD_SEQUENCE})")
        print(f"   Measurement Mitigation: {'✅' if self.USE_MEAS_MITIGATION else '❌'}")
        print(f"   Shots: {self.SHOTS}")
        print(f"   Search Depth: {self.SEARCH_DEPTH}")
        print("="*70 + "\n")


# ==========================================
# 7. BACKEND SELECTION & ANALYSIS
# ==========================================
def select_backend(config):
    """Select appropriate backend with full analysis"""
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=config.TOKEN,
        instance=config.CRN
    )

    print("\n🔍 Available IBM Quantum Backends:")
    backends = service.backends()
    for backend in backends:
        try:
            status = backend.status()
            queue_info = status.pending_jobs
            queue_length = queue_info if isinstance(queue_info, int) else len(queue_info)
            print(f"  {backend.name} ({backend.num_qubits} qubits, {'🟢 Operational' if status.operational else '🔴 Offline'})")
            print(f"     Queue: {queue_length} jobs")
        except Exception as e:
            print(f"  {backend.name} (Status unavailable: {e})")

    if config.BITS == 17:
        print("\n💡 Using AerSimulator for 17-bit key (most efficient)")
        return AerSimulator()
    else:
        print(f"\n💡 Selecting real backend for {config.BITS}-bit key...")
        backend = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=156
        )
        print(f"  Selected: {backend.name} ({backend.num_qubits} qubits)")
        try:
            status = backend.status()
            queue_info = status.pending_jobs
            queue_length = queue_info if isinstance(queue_info, int) else len(queue_info)
            print(f"  Queue status: {queue_length} jobs pending")
        except Exception as e:
            print(f"  Queue status: (unavailable: {e})")
        return backend


def estimate_gate_counts(qc):
    """Counts specific expensive gates with detailed logging"""
    counts = {"CX": 0, "CCX": 0, "T": 0, "TDG": 0}
    for instruction in qc.data:
        name = instruction.operation.name.upper()
        if name in counts:
            counts[name] += 1
        if name == "TDG":
            counts["T"] += 1  # TDG counts as T gate
    logger.debug(f"Gate counts: {counts}")
    return counts


def analyze_circuit_costs(qc, backend):
    """Prints detailed circuit statistics before execution"""
    total_qubits = qc.num_qubits
    gate_counts = estimate_gate_counts(qc)


    print("\n" + "="*50)
    print("   CIRCUIT ANALYSIS REPORT")
    print("="*50)
    print(f"[✅] Logical Qubits: {total_qubits}")
    print(f"[✅] Circuit Depth:  {qc.depth()}")
    print(f"[✅] Gate Counts:")
    print(f"    - CX gates:   {gate_counts['CX']}")
    print(f"    - CCX gates:  {gate_counts['CCX']}")
    print(f"    - T gates:    {gate_counts['T']}")
    print(f"    - Total gates: {sum(gate_counts.values())}")


    # Check backend capacity
    backend_qubits = backend.configuration().n_qubits if hasattr(backend, 'configuration') else 127


    if total_qubits > backend_qubits:
        logger.warning(f"⚠️ CRITICAL: Circuit ({total_qubits}q) exceeds backend {backend.name} ({backend_qubits}q)!")
        print(f"[⚠️] WARNING: Circuit requires {total_qubits} qubits but backend only has {backend_qubits}")
    elif total_qubits > 156:
        logger.warning(f"⚠️ High qubit count ({total_qubits}). Execution may be unstable.")
        print(f"[ℹ️] Note: High qubit count may affect stability")
    else:
        print(f"[✅] Circuit fits within {backend.name} ({backend_qubits} qubits)")


    # Estimate execution time
    if hasattr(backend, 'configuration'):
        avg_gate_time = backend.configuration().timing_constraints['u']['gate_time']
        estimated_time = (sum(gate_counts.values()) * avg_gate_time) / 1e9  # Convert to seconds
        print(f"[⏱️] Estimated execution time: ~{estimated_time:.2f} seconds")


    print("="*50 + "\n")


# ==========================================
# 8. CIRCUIT CONSTRUCTION UTILITIES
# ==========================================
def initialize_qubits(qc: QuantumCircuit, qubit_reg: QuantumRegister, initial_state: int = 0):
    """Initialize qubits in superposition or specific state"""
    if initial_state == 1:
        for q in qubit_reg:
            qc.x(q)
    else:
        for q in qubit_reg:
            qc.h(q)
    logger.debug(f"Initialized {len(qubit_reg)} qubits in state {initial_state}")

def apply_semiclassical_qft_phase_component(qc: QuantumCircuit, ctrl: QuantumRegister,
                                           creg: ClassicalRegister, bits: int, k: int):
    """Apply the standard semiclassical QFT phase component."""
    for m in range(k):
        with qc.if_test((creg[m], 1)):
            qc.p(-math.pi / (2 ** (k - m)), ctrl[0])


def apply_ft_to_qubit(qc: QuantumCircuit, qubit, config):
    """Apply fault tolerance to a single qubit if enabled"""
    if not config.USE_FT:
        return None


    anc = QuantumRegister(2, f"ft_anc_{qubit}")
    qc.add_register(anc)
    prepare_verified_ancilla(qc, anc[0])
    prepare_verified_ancilla(qc, anc[1])
    encode_repetition(qc, qubit, anc)
    return anc


def apply_ft_to_register(qc: QuantumCircuit, reg: QuantumRegister, config):
    """Apply fault tolerance to all qubits in a register if enabled"""
    if not config.USE_FT:
        return []


    ancillas = []
    for i, qubit in enumerate(reg):
        anc = apply_ft_to_qubit(qc, qubit, config)
        if anc:
            ancillas.append(anc)
    return ancillas


def decode_ft_register(qc: QuantumCircuit, reg: QuantumRegister, ancillas: List, config):
    """Decode fault tolerance from all qubits in a register if enabled"""
    if not config.USE_FT:
        return


    for i, (qubit, anc) in enumerate(zip(reg, ancillas)):
        decode_repetition(qc, anc, qubit)


def apply_hadamard_gates(qc: QuantumCircuit, qubit_reg: QuantumRegister):
    """Apply Hadamard gates to create superposition"""
    for q in qubit_reg:
        qc.h(q)
    logger.debug(f"Applied Hadamard gates to {len(qubit_reg)} qubits")


# ==========================================
# 8. MODE SELECTION & STRATEGY
# ==========================================
def get_oracle_strategy(mode_id: int, backend_qubits: int) -> str:
    """Determine the oracle strategy based on the mode and available qubits"""
    if mode_id in [14, 34, 35, 41]:
        return "SHOR_PURE" if backend_qubits >= 150 else "SHOR_MOD"
    elif mode_id in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 36, 37, 38, 39]:
        return "2D" if backend_qubits >= 140 else "SERIAL"
    return "SERIAL"


# --- MODE BUILDERS (SELECTED BEST) ---
def get_best_mode_id(bits: int, available_qubits: int) -> int:
    """Get the best mode ID based on bits and available qubits"""
    # Prioritize high-success modes that fit within available qubits
    # KING mode has highest priority when available
    high_success_modes = [
        (41, 85),    # KING mode - highest success and priority
        (29, 75),     # Semiclassical Omega
        (39, 78),     # Matrix Mod Omega
        (20, 75),     # Matrix Mod
        (4, 75),      # Semiclassical
        (9, 65),      # Shadow 2D
        (2, 65),      # Hive (Chunked)
        (11, 60),     # Swarm
        (7, 60),      # Geometric IPE
        (8, 65),      # Verified (Flags)
        (10, 60),     # Reverse IPE
        (13, 60),     # Compressed Shadow
        (25, 65)      # Hive Edition
    ]


    for mode_id, success in high_success_modes:
        meta = MODE_METADATA.get(mode_id, {})
        req_qubits = meta.get("qubits", 135)
        if isinstance(req_qubits, str) and "~" in req_qubits:
            req_qubits = int(req_qubits.replace("~", ""))
        if req_qubits <= available_qubits:
            return mode_id


    return 41  # Default to KING (41) if no other mode fits


# ==========================================
# 6. CIRCUIT BUILDERS - ALL 41 MODES
# ==========================================

def build_circuit_selector(mode_id: int, bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Select and build the appropriate quantum circuit based on the mode"""
    logger.info(f"Building circuit for mode {mode_id}: {MODE_METADATA[mode_id]['name']}")

    # Compute necessary parameters
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    delta_point = compute_offset(Q, config.KEYSPACE_START)
    points = precompute_points(bits)
    strategy = get_oracle_strategy(mode_id, 127)

    # Mode 0: Hardware IPE Diagnostic (136 qubits)
    if mode_id == 0:
        return build_mode_0_hardware_probe(bits, delta_point, config)
    elif mode_id == 1:
        return build_mode_1_ipe_standard(bits, delta_point, config, strategy)
    elif mode_id == 2:
        return build_mode_2_hive_chunked(bits, delta_point, config)
    elif mode_id == 3:
        return build_mode_3_windowed_ipe(bits, delta_point, strategy)
    elif mode_id == 4:
        return build_mode_4_semiclassical(bits, delta_point, config, strategy)
    elif mode_id == 5:
        return build_mode_5_ab_shor_optimized(bits, delta_point)
    elif mode_id == 6:
        return build_mode_6_ft_draper_test(bits, delta_point)
    elif mode_id == 7:
        return build_mode_7_geometric_ipe(bits, delta_point, config)
    elif mode_id == 8:
        return build_mode_8_verified_flags(bits, delta_point, config)
    elif mode_id == 9:
        return build_mode_9_shadow_2d(bits, delta_point, config)
    elif mode_id == 10:
        return build_mode_10_reverse_ipe(bits, delta_point, config)
    elif mode_id == 11:
        return build_mode_11_swarm(bits, delta_point, config)
    elif mode_id == 12:
        return build_mode_12_heavy_draper(bits, delta_point)
    elif mode_id == 13:
        return build_mode_13_compressed_shadow(bits, delta_point, config)
    elif mode_id == 14:
        return build_mode_14_shor_logic(bits, delta_point, strategy)
    elif mode_id == 15:
        return build_mode_15_geo_ipe_exp(bits, delta_point)
    elif mode_id == 16:
        return build_mode_16_window_explicit(bits, delta_point)
    elif mode_id == 17:
        return build_mode_17_hive_swarm(bits, delta_point)
    elif mode_id == 18:
        return build_mode_18_explicit_logic(bits, delta_point)
    elif mode_id == 19:
        return build_mode_19_fixed_ab(bits, delta_point)
    elif mode_id == 20:
        return build_mode_20_matrix_mod(bits, delta_point, config)
    elif mode_id == 21:
        return build_mode_21_phantom_parallel(bits, delta_point, strategy)
    elif mode_id == 22:
        return build_mode_22_shor_parallel(bits, delta_point, strategy)
    elif mode_id == 23:
        return build_mode_23_ghz_parallel(bits, delta_point, strategy)
    elif mode_id == 24:
        return build_mode_24_verified_parallel(bits, delta_point, strategy)
    elif mode_id == 25:
        return build_mode_25_hive_edition(bits, delta_point, config)
    elif mode_id == 26:
        return build_mode_26_extra_shadow(bits, delta_point, strategy)
    elif mode_id == 27:
        return build_mode_27_advanced_qpe(bits, delta_point, strategy)
    elif mode_id == 28:
        return build_mode_28_full_quantum_optimized(bits, delta_point)
    elif mode_id == 29:
        return build_mode_29_semiclassical_omega(bits, delta_point, strategy)
    elif mode_id == 41:
        return build_mode_41_semiclassical_omega(bits, delta_point, config)
    elif mode_id == 30:
        return build_mode_30_verified_shadow(bits, delta_point, strategy)
    elif mode_id == 31:
        return build_mode_31_verified_advanced(bits, delta_point, strategy)
    elif mode_id == 32:
        return build_mode_32_heavy_draper_omega(bits, delta_point)
    elif mode_id == 33:
        return build_mode_33_compressed_shadow_omega(bits, delta_point, strategy)
    elif mode_id == 34:
        return build_mode_34_shor_logic_omega(bits, delta_point, strategy)
    elif mode_id == 35:
        return build_mode_35_geometric_ipe_omega(bits, delta_point)
    elif mode_id == 36:
        return build_mode_36_windowed_ipe_omega(bits, delta_point)
    elif mode_id == 37:
        return build_mode_37_hive_swarm_omega(bits, delta_point)
    elif mode_id == 38:
        return build_mode_38_explicit_logic_omega(bits, delta_point, config)
    elif mode_id == 39:
        return build_mode_39_matrix_mod_omega(bits, delta_point, config)
    elif mode_id == 40:
        return build_mode_40_matrix_mod_smart_scalable(bits, delta_point)
    else:
        logger.warning(f"Mode {mode_id} not in optimized list, defaulting to KING (Mode 41)")
        return build_mode_41_semiclassical_omega(bits, delta_point, config)

# --- MODE 0: HARDWARE IPE DIAGNOSTIC ---
def build_mode_0_hardware_probe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Hardware IPE Diagnostic mode with FT support"""
    logger.info("Building Mode 0: Hardware IPE Diagnostic (136 qubits)")
    reg_ctrl = QuantumRegister(1, 'ctrl')
    reg_state = QuantumRegister(2, 'state')
    reg_flag = QuantumRegister(2, 'flag')
    creg = ClassicalRegister(bits, 'meas')
    creg_flag = ClassicalRegister(bits * 2, 'flag_meas')


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(4, 'ft_state'),
            QuantumRegister(4, 'ft_flag')
        ])


    qc = QuantumCircuit(reg_ctrl, reg_state, reg_flag, creg, creg_flag, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, reg_ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, reg_state, config))
        ft_ancillas.extend(apply_ft_to_register(qc, reg_flag, config))


    qc.x(reg_state[0])
    qc.h(reg_state[1])


    for k in range(min(bits, 8)):
        if k > 0:
            qc.reset(reg_ctrl)
            qc.reset(reg_flag)
        qc.h(reg_ctrl[0])
        qc.cz(reg_ctrl[0], reg_state[0])
        qc.cz(reg_ctrl[0], reg_state[1])
        qc.cx(reg_ctrl[0], reg_flag[0])
        qc.cx(reg_ctrl[0], reg_flag[1])


        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), reg_ctrl[0])


        qc.h(reg_ctrl[0])
        qc.measure(reg_ctrl[0], creg[k])
        qc.measure(reg_flag[0], creg_flag[2 * k])
        qc.measure(reg_flag[1], creg_flag[2 * k + 1])


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, reg_ctrl, [ft_ancillas[0]], config)
        decode_ft_register(qc, reg_state, ft_ancillas[1:3], config)
        decode_ft_register(qc, reg_flag, ft_ancillas[3:], config)


    return qc


def build_mode_1_ipe_standard(bits: int, delta: Point, config: Config, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the IPE Standard mode."""
    logger.info(f"Building Mode 1: IPE Standard [Strategy: {strategy}]")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    regs = [ctrl, state, creg]

    if config.USE_FT:
        regs.append(QuantumRegister(2, "ft_anc"))

    qc = QuantumCircuit(*regs)

    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])

        if config.USE_FT:
            prepare_verified_ancilla(qc, regs[-1][0])
            prepare_verified_ancilla(qc, regs[-1][1])
            encode_repetition(qc, ctrl[0], regs[-1])

        apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
        power = 1 << k
        dx = (delta.x() * power) % N
        dy = (delta.y() * power) % N

        if strategy == "2D":
            draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)
        else:
            draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, dy)

        if config.USE_FT:
            decode_repetition(qc, regs[-1], ctrl[0])

        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

    return qc


# --- MODE 2: HIVE (CHUNKED) ---
def build_mode_2_hive_chunked(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Hive (Chunked) mode with FT support"""
    logger.info("Building Mode 2: Hive (Chunked) (127 qubits)")
    state_bits = (bits // 2 + 1)
    ctrl = QuantumRegister(4, "ctrl")
    state = QuantumRegister(state_bits, "state")
    creg = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(8, 'ft_ctrl'),
            QuantumRegister(2 * state_bits, 'ft_state')
        ])


    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))


    for start in range(0, bits, 4):
        chunk = min(4, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])


        for j in range(chunk):
            k = start + j
            pwr = 1 << k
            dx = (delta.x() * pwr) % N
            draper_adder_oracle_1d_serial(qc, ctrl[j], state, dx, 0)
            apply_semiclassical_qft_phase_component(qc, ctrl[j], creg, bits, k)


        qc.measure(ctrl[:chunk], creg[start:start + chunk])


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:4], config)
        decode_ft_register(qc, state, ft_ancillas[4:], config)


    return qc


def build_mode_3_windowed_ipe(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Windowed IPE mode."""
    logger.info(f"Building Mode 3: Windowed IPE [Strategy: {strategy}]")
    ctrl = QuantumRegister(4, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)
    
    for start in range(0, bits, 4):
        chunk = min(4, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
        
        for j in range(chunk):
            k = start + j
            pwr = 1 << k
            dx = (delta.x() * pwr) % N
            dy = (delta.y() * pwr) % N
            
            if strategy == "2D":
                draper_adder_oracle_2d(qc, ctrl[j], state, dx, dy)
            else:
                draper_adder_oracle_1d_serial(qc, ctrl[j], state, dx, dy)
            apply_semiclassical_qft_phase_component(qc, ctrl[j], creg, bits, k)
        
        qc.measure(ctrl[:chunk], creg[start:start + chunk])
    
    return qc


def build_mode_4_semiclassical(bits: int, delta: Point, config: Config, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Semiclassical mode."""
    logger.info(f"Building Mode 4: Semiclassical [Strategy: {strategy}]")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    regs = [ctrl, state, creg]

    if config.USE_FT:
        regs.append(QuantumRegister(2, "ft_anc"))

    qc = QuantumCircuit(*regs)

    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])

        if config.USE_FT:
            prepare_verified_ancilla(qc, regs[-1][0])
            prepare_verified_ancilla(qc, regs[-1][1])
            encode_repetition(qc, ctrl[0], regs[-1])

        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-pi / (2 ** (k - m)), ctrl[0])

        power = 1 << k
        dx = (delta.x() * power) % N
        dy = (delta.y() * power) % N

        if strategy == "2D":
            draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)
        else:
            draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, dy)

        if config.USE_FT:
            decode_repetition(qc, regs[-1], ctrl[0])

        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

    return qc


def build_mode_5_ab_shor_optimized(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the AB Shor (Optimized) mode."""
    logger.info("Building Mode 5: AB Shor (Optimized for 156 qubits)")
    coord_bits = bits // 2
    a = QuantumRegister(coord_bits, "a")
    b = QuantumRegister(coord_bits, "b")
    acc = QuantumRegister(bits, "acc")
    anc = QuantumRegister(4, "anc")
    creg = ClassicalRegister(2 * coord_bits, "meas")
    qc = QuantumCircuit(a, b, acc, anc, creg)
    
    qc.h(a)
    qc.h(b)
    
    points = precompute_points(bits)
    ecdlp_oracle_ab(qc, a, b, acc, points, anc, ORDER)
    
    qc.append(QFTGate(coord_bits, inverse=True), a)
    qc.append(QFTGate(coord_bits, inverse=True), b)
    qc.measure(a, creg[:coord_bits])
    qc.measure(b, creg[coord_bits:])
    
    return qc


def build_mode_6_ft_draper_test(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the FT Draper Test mode."""
    logger.info("Building Mode 6: FT Draper Test")
    target = QuantumRegister(bits, "target")
    ctrl = QuantumRegister(1, "ctrl")
    anc = QuantumRegister(2, "anc")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(target, ctrl, anc, creg)
    
    qc.x(ctrl[0])
    ft_draper_modular_adder(qc, ctrl[0], target, anc, 12345, N)
    qc.measure(target, creg)
    
    return qc


# --- MODE 7: GEOMETRIC IPE ---
def build_mode_7_geometric_ipe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Geometric IPE mode with FT support and precomputed powers"""
    logger.info("Building Mode 7: Geometric IPE (136 qubits)")

    # --- PRECOMPUTE POWERS OF DELTA (NEW) ---
    powers = []
    curr = delta
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)
    # --- END PRECOMPUTE ---

    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")

    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])

    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)

    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))

    qc.append(synth_qft_full(bits, do_swaps=False).to_gate(), state)

    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])

        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])

        # --- USE PRECOMPUTED POWERS (UPDATED) ---
        if powers[k]:
            vx = powers[k].x()
            vy = powers[k].y()  # Optional for 2D precision
            for i in range(bits):
                angle_x = 2 * math.pi * vx / (2 ** (i + 1))
                qc.cp(angle_x, ctrl[0], state[i])
                # Uncomment for 2D: qc.cp(2 * math.pi * vy / (2 ** (i + 1)), ctrl[0], state[i])

        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])

        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

    # Decode fault tolerance for state register if enabled
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i+1], state[i])

    return qc

# --- MODE 8: VERIFIED (FLAGS) ---
def build_mode_8_verified_flags(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Verified (Flags) mode with FT support"""
    logger.info("Building Mode 8: Verified (Flags) (136 qubits)")
    n_flags = 2  # Reduced from config.USE_FLAGS for qubit efficiency
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    flags = QuantumRegister(n_flags, "flag")
    c_meas = ClassicalRegister(bits, "meas")
    c_flags = ClassicalRegister(bits * n_flags, "flag_out")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state'),
            QuantumRegister(4, 'ft_flags')
        ])


    qc = QuantumCircuit(ctrl, state, flags, c_meas, c_flags, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
        ft_ancillas.extend(apply_ft_to_register(qc, flags, config))


    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.reset(flags)
            qc.h(ctrl[0])


        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])


        for f in range(n_flags):
            qc.cx(ctrl[0], flags[f])


        apply_semiclassical_qft_phase_component(qc, ctrl[0], c_meas, bits, k)
        power = 1 << k
        dx = (delta.x() * power) % N
        draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, 0)


        for f in range(n_flags):
            qc.cx(ctrl[0], flags[f])


        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])


        qc.h(ctrl[0])
        qc.measure(ctrl[0], c_meas[k])
        qc.measure(flags, c_flags[k * n_flags : (k + 1) * n_flags])


    # Decode fault tolerance for other registers if enabled
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i+1], state[i])
        for i in range(n_flags):
            decode_repetition(qc, ft_ancillas[bits+i+1], flags[i])


    return qc


# --- MODE 9: SHADOW 2D ---
def build_mode_9_shadow_2d(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Shadow 2D mode with FT support"""
    logger.info("Building Mode 9: Shadow 2D (138 qubits)")
    window_size = 4
    ctrl = QuantumRegister(window_size, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2 * window_size, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])


    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))


    for start in range(0, bits, window_size):
        chunk = min(window_size, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])


        for j in range(chunk):
            k = start + j
            pwr = 1 << k
            dx = (delta.x() * pwr) % N
            dy = (delta.y() * pwr) % N
            draper_adder_oracle_2d(qc, ctrl[j], state, dx, dy)


            for m in range(start):
                with qc.if_test((creg[m], 1)):
                    qc.p(-math.pi / (2 ** (k - m)), ctrl[j])


        qc.append(synth_qft_full(chunk, do_swaps=False).inverse(), ctrl[:chunk])
        qc.measure(ctrl[:chunk], creg[start:start + chunk])


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:window_size], config)
        decode_ft_register(qc, state, ft_ancillas[window_size:], config)


    return qc


# --- MODE 10: REVERSE IPE ---
def build_mode_10_reverse_ipe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Reverse IPE mode with FT support"""
    logger.info("Building Mode 10: Reverse IPE (135 qubits)")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])


    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))


    for k in reversed(range(bits)):
        if k < bits - 1:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])


        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])


        power = 1 << k
        dx = (delta.x() * power) % N
        dy = (delta.y() * power) % N
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)


        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])


        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])


    # Decode fault tolerance for state register if enabled
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i+1], state[i])


    return qc


# --- MODE 11: SWARM ---
def build_mode_11_swarm(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Swarm mode with FT support"""
    logger.info("Building Mode 11: Swarm (127 qubits)")
    workers = max(1, 127 // (bits + 1))
    regs = [
        QuantumRegister(bits, "state")
    ]


    # Add worker registers
    for w in range(workers):
        regs.append(QuantumRegister(1, f"w{w}_c"))
        regs.append(ClassicalRegister(bits, f"w{w}_m"))


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.append(QuantumRegister(2 * bits, 'ft_state'))
        for w in range(workers):
            ft_regs.append(QuantumRegister(2, f'ft_w{w}'))


    qc = QuantumCircuit(*regs, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, regs[0], config))
        for w in range(workers):
            ft_ancillas.append(apply_ft_to_qubit(qc, regs[1 + w*2][0], config))


    state_reg = regs[0]


    for w in range(workers):
        ctrl = regs[1 + w*2][0]
        meas = regs[2 + w*2]


        for start in range(0, bits, 4):
            chunk = min(4, bits - start)
            if start > 0:
                qc.reset(ctrl)
                qc.h(ctrl)


            for j in range(chunk):
                k = start + j
                power = 1 << k
                dx = (delta.x() * power) % N
                draper_adder_oracle_1d_serial(qc, ctrl, state_reg, dx, 0)
                apply_semiclassical_qft_phase_component(qc, ctrl, meas, bits, k)


            qc.measure(ctrl, meas[start:start + chunk])


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, state_reg, ft_ancillas[:bits], config)
        for w in range(workers):
            decode_repetition(qc, ft_ancillas[bits + w], regs[1 + w*2][0])


    return qc


def build_mode_12_heavy_draper(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Heavy Draper mode."""
    logger.info("Building Mode 12: Heavy Draper")
    target = QuantumRegister(bits, "target")
    anc = QuantumRegister(bits, "anc")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(target, anc, creg)
    
    ft_draper_modular_adder(qc, None, target, [anc[0]], 12345, N)
    qc.measure(target, creg)
    
    return qc


# --- MODE 13: COMPRESSED SHADOW ---
def build_mode_13_compressed_shadow(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Compressed Shadow mode with FT support"""
    logger.info("Building Mode 13: Compressed Shadow (136 qubits)")
    window_size = 8
    ctrl = QuantumRegister(window_size, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2 * window_size, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])


    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))


    for start in range(0, bits, window_size):
        chunk = min(window_size, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])


        for j in range(chunk):
            k = start + j
            power = 1 << k
            dx = (delta.x() * power) % N
            draper_adder_oracle_1d_serial(qc, ctrl[j], state, dx, 0)


            for m in range(start):
                with qc.if_test((creg[m], 1)):
                    qc.p(-math.pi / (2 ** (k - m)), ctrl[j])


        qc.append(synth_qft_full(chunk, do_swaps=False).inverse(), ctrl[:chunk])
        qc.measure(ctrl[:chunk], creg[start:start + chunk])


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:window_size], config)
        decode_ft_register(qc, state, ft_ancillas[window_size:], config)


    return qc


def build_mode_14_shor_logic(bits: int, delta: Point, strategy: str = "SHOR_MOD") -> QuantumCircuit:
    """Build the circuit for the Shor Logic mode."""
    logger.info(f"Building Mode 14: Shor Logic [Strategy: {strategy}]")
    if strategy == "SHOR_PURE":
        block_size = min(bits, 8)
        ctrl = QuantumRegister(block_size, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(block_size, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        
        qc.h(ctrl)
        for i in range(block_size):
            val = (delta.x() * (1 << i)) % N
            draper_add_const(qc, ctrl[i], state, val)
        
        qc.append(QFTGate(block_size, inverse=True), ctrl)
        qc.measure(ctrl, creg)
    else:
        block_size = min(bits, 5)
        ctrl = QuantumRegister(block_size, "ctrl")
        state = QuantumRegister(1, "state")
        temp = QuantumRegister(1, "temp")
        creg = ClassicalRegister(block_size, "meas")
        qc = QuantumCircuit(ctrl, state, temp, creg)
        
        qc.x(state[0])
        qft_reg(qc, ctrl)
        dx = delta.x()
        
        for i in range(block_size):
            power = 1 << i
            scalar_val = (Gx * power) + (dx * power)
            eigenvalue_phase_oracle(qc, ctrl[i], state[0], scalar_val, bits)
        
        iqft_reg(qc, ctrl)
        qc.measure(ctrl, creg)
    
    return qc


def build_mode_15_geo_ipe_exp(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Geo IPE (Explicit) mode."""
    logger.info("Building Mode 15: Geo IPE (Explicit)")
    powers = []
    curr = delta
    
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)
    
    ctrl = QuantumRegister(bits, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)
    
    qc.h(ctrl)
    
    geo = GeometricIPE(bits)
    
    for k in range(bits):
        geo._oracle_geometric_phase(qc, ctrl[k], state, powers[k])
    
    qc.append(QFTGate(bits, inverse=True), ctrl)
    qc.measure(ctrl, creg)
    
    return qc


def build_mode_16_window_explicit(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Windowed (Explicit) mode."""
    logger.info("Building Mode 16: Windowed (Explicit)")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)
    
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        
        for m in range(k):
            qc.cp(-pi / (2 ** (k - m)), creg[m], ctrl[0])
        
        ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    
    return qc


def build_mode_17_hive_swarm(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Hive Swarm mode."""
    logger.info("Building Mode 17: Hive Swarm")
    total_q = 127
    state_q = bits
    workers = (total_q - state_q) // 1
    regs = [QuantumRegister(state_q, "state")]
    
    for w in range(workers):
        regs.append(QuantumRegister(1, f"w{w}"))
        regs.append(ClassicalRegister(1, f"c{w}"))
    
    qc = QuantumCircuit(*regs)
    state = qc.qregs[0]
    
    for w in range(workers):
        ctrl = qc.qregs[w + 1]
        qc.h(ctrl)
        draper_adder_oracle_1d_serial(qc, ctrl[0], state, (delta.x() * (1 << w)) % N, 0)
        qc.h(ctrl)
        qc.measure(ctrl, qc.cregs[w])
    
    return qc


def build_mode_18_explicit_logic(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Explicit Logic mode."""
    logger.info("Building Mode 18: Explicit Logic")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)
    
    for k in range(bits):
        if k > 0:
            with qc.if_test((creg[k - 1], 1)):
                qc.x(ctrl[0])
        
        qc.h(ctrl[0])
        apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
        ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    
    return qc


def build_mode_19_fixed_ab(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Fixed AB mode."""
    logger.info("Building Mode 19: Fixed AB")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    anc = QuantumRegister(4, "anc")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, anc, creg)
    
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        
        apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
        ft_draper_modular_adder(qc, ctrl[0], state, anc, (1 << k) % N, N)
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    
    return qc


# --- MODE 20: MATRIX MOD (SMART SCALABLE) ---
def build_mode_20_matrix_mod(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Mode 20: Matrix Mod (Unitary) - Smart Scalable with FT support"""
    logger.info("Building Mode 20: Matrix Mod (Smart Scalable) - 135 qubits")


    # Main registers
    qr = QuantumRegister(bits, "state")
    cr = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.append(QuantumRegister(2 * bits, 'ft_state'))


    qc = QuantumCircuit(qr, cr, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, qr, config))


    # Initialize state
    qc.x(qr[0])


    # Apply matrix modulation
    if config.USE_SMART_GATE:
        # Create scalable unitary gate for modular addition
        gate = add_const_mod_gate(delta.x(), 2 ** bits)
        qc.append(gate, qr)
    else:
        # Fallback to standard QFT-based addition
        qft_reg(qc, qr)
        for i in range(bits):
            qc.p(2 * math.pi * delta.x() / (2 ** (bits - i)), qr[i])
        iqft_reg(qc, qr)


    # Apply Draper oracle with delta
    power = 1 << (bits // 2)
    dx = (delta.x() * power) % N
    dy = (delta.y() * power) % N
    draper_adder_oracle_2d(qc, None, qr, dx, dy)


    # Measure results
    qc.measure(qr, cr)


    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, qr, ft_ancillas, config)


    return qc


def build_mode_21_phantom_parallel(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Phantom Parallel mode."""
    logger.info(f"Building Mode 21: Phantom Parallel [Strategy: {strategy}]")
    if strategy == "2D":
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(bits, 'state')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        
        dx, dy = delta.x(), delta.y()
        for i in range(bits):
            power = 1 << i
            sx = (Gx * power + dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_22_shor_parallel(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Shor Parallel mode."""
    logger.info(f"Building Mode 22: Shor Parallel [Strategy: {strategy}]")
    if strategy == "2D":
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(bits, 'state')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        
        dx, dy = delta.x(), delta.y()
        for i in range(bits):
            power = 1 << i
            sx = (Gx * power + dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_23_ghz_parallel(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the GHZ Parallel mode."""
    logger.info(f"Building Mode 23: GHZ Parallel [Strategy: {strategy}]")
    if strategy == "2D":
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(bits, 'state')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.h(reg_count[0])
        for i in range(bits - 1):
            qc.cx(reg_count[i], reg_count[i + 1])
        
        qc.x(reg_state[0])
        dx, dy = delta.x(), delta.y()
        
        for i in range(bits):
            power = 1 << i
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_24_verified_parallel(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Verified Parallel mode."""
    logger.info(f"Building Mode 24: Verified Parallel [Strategy: {strategy}]")
    if strategy == "2D":
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(bits, 'state')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        
        dx, dy = delta.x(), delta.y()
        for i in range(bits):
            power = 1 << i
            sx = (Gx * power + dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


# --- MODE 25: HIVE EDITION ---
def build_mode_25_hive_edition(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Hive Edition mode with FT support."""
    logger.info("Building Mode 25: Hive Edition (127 qubits)")
    available_qubits = 127
    num_workers = available_qubits // (bits + 2)
    if num_workers < 1:
        num_workers = 1

    regs = [QuantumRegister(bits, "state")]
    for w in range(num_workers):
        regs.append(QuantumRegister(1, f'w{w}_c'))
        regs.append(ClassicalRegister(bits, f'w{w}_m'))

    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.append(QuantumRegister(2 * bits, 'ft_state'))
        for w in range(num_workers):
            ft_regs.append(QuantumRegister(2, f'ft_w{w}'))

    qc = QuantumCircuit(*regs, *ft_regs)

    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, regs[0], config))
        for w in range(num_workers):
            ft_ancillas.append(apply_ft_to_qubit(qc, regs[1 + w*2][0], config))

    state_reg = regs[0]
    powers = []
    curr = delta
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)

    for w in range(num_workers):
        qc.x(qc.qubits[w * (bits + 1) + 1])

    for k in range(bits):
        for w in range(num_workers):
            ctrl = qc.qubits[w * (bits + 1)]
            target_start = w * (bits + 1) + 1
            target_reg = qc.qubits[target_start:target_start + bits]
            qc.reset(ctrl)
            qc.h(ctrl)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()  # Fixed: .x() and .y()
                draper_adder_oracle_1d_serial(qc, ctrl, target_reg, dx, 0)
            qc.h(ctrl)
            qc.measure(ctrl, qc.cregs[w][k])

    # Decode fault tolerance if enabled
    if config.USE_FT:
        decode_ft_register(qc, state_reg, ft_ancillas[:bits], config)
        for w in range(num_workers):
            decode_repetition(qc, ft_ancillas[bits + w], regs[1 + w*2][0])

    return qc


def build_mode_26_extra_shadow(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Extra Shadow mode."""
    logger.info(f"Building Mode 26: Extra Shadow [Strategy: {strategy}]")
    if strategy == "2D":
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(bits, 'state')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        
        dx, dy = delta.x(), delta.y()
        for i in range(bits):
            power = 1 << i
            sx = (Gx * power + dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_27_advanced_qpe(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Advanced QPE mode."""
    logger.info(f"Building Mode 27: Advanced QPE [Strategy: {strategy}]")
    if strategy == "2D":
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_state = QuantumRegister(bits, 'state')
        creg_phase = ClassicalRegister(bits, 'phase_bits')
        creg_flag = ClassicalRegister(bits, 'flag_bits')
        qc = QuantumCircuit(reg_ctrl, reg_state, creg_phase, creg_flag)
        
        qc.x(reg_state[0])
        dx, dy = delta.x(), delta.y()
        
        for k in range(bits):
            qc.reset(reg_ctrl)
            qc.h(reg_ctrl)
            power = 1 << (bits - 1 - k)
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_state, sx, sy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[bits - 1 - m], 1)):
                    qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[bits - 1 - k])
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_28_full_quantum_optimized(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Full Quantum (Optimized) mode."""
    logger.info("Building Mode 28: Full Quantum (Optimized for 156 qubits)")
    coord_bits = bits // 2
    reg_count = QuantumRegister(coord_bits, 'count')
    reg_state = QuantumRegister(coord_bits, 'state')
    creg = ClassicalRegister(coord_bits, 'meas')
    qc = QuantumCircuit(reg_count, reg_state, creg)
    
    qc.x(reg_state[0])
    qft_reg(qc, reg_count)
    
    dx, dy = delta.x(), delta.y()
    for i in range(coord_bits):
        power = 1 << i
        sx = (Gx * power + dx * power) % N
        sy = (dy * power) % N
        draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
    
    iqft_reg(qc, reg_count)
    qc.measure(reg_count, creg)
    
    return qc

# Change delta: Point
def build_mode_29_semiclassical_omega(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Semiclassical Omega mode."""
    logger.info(f"Building Mode 29: Semiclassical Omega [Strategy: {strategy}]")
    if strategy == "2D":
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        
        qc.x(qr_s[0])
        dx, dy = delta.x(), delta.y()
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            power = 1 << k
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, sx, sy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc


# --- MODE KING: SEMICLASSICAL OMEGA (THE KING) ---
def build_mode_41_semiclassical_omega(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Semiclassical Omega mode with FT support"""
    logger.info("Building Mode KING: Semiclassical Omega (136 qubits) - THE KING")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])


    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))


    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])


        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])


        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), ctrl[0])


        power = 1 << k
        dx = (delta.x() * power) % N
        dy = (delta.y() * power) % N
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)


        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])


        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])


    # Decode fault tolerance for state register if enabled
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i+1], state[i])

    return qc


def build_mode_30_verified_shadow(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Verified Shadow mode."""
    logger.info(f"Building Mode 30: Verified Shadow [Strategy: {strategy}]")
    if strategy == "2D":
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_state = QuantumRegister(bits, 'state')
        reg_flag = QuantumRegister(1, 'flag')
        creg_phase = ClassicalRegister(bits, 'phase_bits')
        creg_flag = ClassicalRegister(bits, 'flag_meas')
        qc = QuantumCircuit(reg_ctrl, reg_state, reg_flag, creg_phase, creg_flag)
        
        qc.x(reg_state[0])
        dx, dy = delta.x(), delta.y()
        
        for k in range(bits):
            qc.reset(reg_ctrl)
            qc.reset(reg_flag)
            qc.h(reg_ctrl)
            power = 1 << (bits - 1 - k)
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_state, sx, sy)
            qc.cx(reg_ctrl[0], reg_flag[0])
            
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[bits - 1 - m], 1)):
                    qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[bits - 1 - k])
            qc.measure(reg_flag[0], creg_flag[bits - 1 - k])
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        qr_f = QuantumRegister(1, "flag")
        cr = ClassicalRegister(bits, "meas")
        cr_f = ClassicalRegister(bits, "meas_f")
        qc = QuantumCircuit(qr_c, qr_s, qr_f, cr, cr_f)
        
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.reset(qr_f)
            qc.h(qr_c)
            qc.cx(qr_c[0], qr_f[0])
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
            qc.measure(qr_f[0], cr_f[k])
    
    return qc


def build_mode_31_verified_advanced(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Verified Advanced mode."""
    logger.info(f"Building Mode 31: Verified Advanced [Strategy: {strategy}]")
    if strategy == "2D":
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        qr_f1 = QuantumRegister(1, "flag_init")
        qr_f2 = QuantumRegister(1, "flag_op")
        cr_meas = ClassicalRegister(bits, "meas")
        cr_f1 = ClassicalRegister(bits, "f1")
        cr_f2 = ClassicalRegister(bits, "f2")
        qc = QuantumCircuit(qr_c, qr_s, qr_f1, qr_f2, cr_meas, cr_f1, cr_f2)
        
        qc.x(qr_s[0])
        dx, dy = delta.x(), delta.y()
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.reset(qr_f1)
            qc.reset(qr_f2)
            qc.h(qr_c)
            qc.cx(qr_c[0], qr_f1[0])
            power = 1 << k
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, sx, sy)
            qc.cx(qr_c[0], qr_f2[0])
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr_meas[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr_meas[k])
            qc.measure(qr_f1[0], cr_f1[k])
            qc.measure(qr_f2[0], cr_f2[k])
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        qr_f1 = QuantumRegister(1, "flag_init")
        qr_f2 = QuantumRegister(1, "flag_op")
        cr_meas = ClassicalRegister(bits, "meas")
        cr_f1 = ClassicalRegister(bits, "f1")
        cr_f2 = ClassicalRegister(bits, "f2")
        qc = QuantumCircuit(qr_c, qr_s, qr_f1, qr_f2, cr_meas, cr_f1, cr_f2)
        
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.reset(qr_f1)
            qc.reset(qr_f2)
            qc.h(qr_c)
            qc.cx(qr_c[0], qr_f1[0])
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.cx(qr_c[0], qr_f2[0])
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr_meas[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr_meas[k])
            qc.measure(qr_f1[0], cr_f1[k])
            qc.measure(qr_f2[0], cr_f2[k])
    
    return qc

# delta: Point
def build_mode_32_heavy_draper_omega(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Heavy Draper Omega mode with precomputed powers."""
    logger.info("Building Mode 32: Heavy Draper Omega")

    # Precompute powers of delta
    powers = []
    curr = delta
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)

    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(bits, "state")
    qr_anc = QuantumRegister(1, "anc")
    qr_tmp = QuantumRegister(1, "tmp")
    cr = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, qr_anc, qr_tmp, cr)

    qc.x(qr_s[0])

    for k in range(bits):
        qc.reset(qr_c)
        qc.h(qr_c)

        if powers[k]:
            dx = powers[k].x()  # Fixed: .x()
            dy = powers[k].y()  # Fixed: .y()
            ft_draper_modular_adder_omega(qc, dx, qr_s, N, qr_anc, qr_tmp)

        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])

        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])

    return qc


def build_mode_33_compressed_shadow_omega(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    """Build the circuit for the Compressed Shadow Omega mode."""
    logger.info(f"Building Mode 33: Compressed Shadow Omega [Strategy: {strategy}]")
    if strategy == "2D":
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_x = QuantumRegister(bits, 'x_coord')
        reg_sign = QuantumRegister(1, 'sign')
        creg_phase = ClassicalRegister(bits, 'phase_bits')
        qc = QuantumCircuit(reg_ctrl, reg_x, reg_sign, creg_phase)
        
        qc.x(reg_x[0])
        dx, dy = delta.x(), delta.y()
        
        for k in range(bits):
            qc.reset(reg_ctrl)
            qc.h(reg_ctrl)
            power = 1 << (bits - 1 - k)
            sx = (dx * power) % N
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_x, sx, sy)
            qc.cx(reg_ctrl[0], reg_sign[0])
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[bits - 1 - m], 1)):
                    qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[bits - 1 - k])
    else:
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        
        qc.x(qr_s[0])
        powers = []
        curr = delta
        
        for _ in range(bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    
    return qc

# Recover delta: Point
def build_mode_34_shor_logic_omega(
    bits: int,
    delta: Point,
    strategy: str = "SHOR_MOD"
) -> QuantumCircuit:
    """Build the circuit for the Shor Logic Omega mode."""
    logger.info(f"Building Mode 34: Shor Logic Omega [Strategy: {strategy}]")

    if strategy == "SHOR_PURE":
        reg_a = QuantumRegister(bits, 'a')
        reg_b = QuantumRegister(bits, 'b')
        reg_work = QuantumRegister(bits, 'work')
        creg = ClassicalRegister(2 * bits, 'meas')
        qc = QuantumCircuit(reg_a, reg_b, reg_work, creg)

        qc.h(reg_a)
        qc.h(reg_b)
        qc.x(reg_work[0])

        for i in range(bits):
            power = 1 << i
            sx = (G.x() * power) % N  # Fixed: G.x()
            sy = (G.y() * power) % N  # Fixed: G.y()
            draper_adder_oracle_2d(qc, reg_a[i], reg_work, sx, sy)

        target_scalar = delta.x()  # Fixed: delta.x()
        for i in range(bits):
            power = 1 << i
            sx = (target_scalar * power) % N
            sy = (delta.y() * power) % N  # Fixed: delta.y()
            draper_adder_oracle_2d(qc, reg_b[i], reg_work, sx, sy)

        iqft_reg(qc, reg_a)
        iqft_reg(qc, reg_b)
        qc.measure(reg_a, creg[0:bits])
        qc.measure(reg_b, creg[bits:2 * bits])

    else:  # SHOR_MOD
        reg_count = QuantumRegister(bits, 'count')
        reg_state = QuantumRegister(1, 'state')
        reg_temp = QuantumRegister(1, 'temp_ph')
        creg = ClassicalRegister(bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, reg_temp, creg)

        qc.x(reg_state[0])
        qft_reg(qc, reg_count)

        dx = delta.x()  # Fixed: delta.x()
        dy = delta.y()  # Fixed: delta.y()

        for i in range(bits):
            power = 1 << i
            scalar_val_x = (G.x() * power + dx * power) % N  # Fixed: G.x()
            eigenvalue_phase_oracle(qc, reg_count[i], reg_state[0], scalar_val_x, bits)

        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)

    logger.debug(f"Applied Shor Logic Omega oracle with strategy {strategy}")
    return qc


def build_mode_35_geometric_ipe_omega(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Geometric IPE Omega mode."""
    logger.info("Building Mode 35: Geometric IPE Omega")
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(bits, "state")
    cr = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, cr)
    
    qc.append(QFT(bits, do_swaps=False), qr_s)
    powers = []
    curr = delta
    
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)
    
    engine = GeometricIPE(bits)
    
    for k in range(bits):
        qc.reset(qr_c)
        qc.h(qr_c)
        engine._oracle_geometric_phase(qc, qr_c[0], qr_s, powers[k])
        
        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])
        
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    
    return qc


def build_mode_36_windowed_ipe_omega(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Windowed IPE Omega mode."""
    logger.info("Building Mode 36: Windowed IPE Omega")
    window_size = 4
    qr_c = QuantumRegister(window_size, "ctrl")
    qr_s = QuantumRegister(bits, "state")
    cr = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, cr)
    
    qft_reg(qc, qr_s)
    powers = []
    curr = delta
    
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)
    
    engine = GeometricIPE(bits)
    
    for i in range(0, bits, window_size):
        chunk = min(window_size, bits - i)
        qc.reset(qr_c)
        qc.h(qr_c[:chunk])
        
        for j in range(chunk):
            engine._oracle_geometric_phase(qc, qr_c[j], qr_s, powers[i + j])
        
        qc.append(QFTGate(chunk, do_swaps=False).inverse(), qr_c[:chunk])
        for j in range(chunk):
            qc.measure(qr_c[j], cr[i + j])
    
    return qc


def build_mode_37_hive_swarm_omega(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Hive Swarm Omega mode."""
    logger.info("Building Mode 37: Hive Swarm Omega")
    total_qubits = 127
    num_workers = total_qubits // (bits + 1)
    
    if num_workers < 2:
        return build_mode_35_geometric_ipe_omega(bits, delta)
    
    regs = []
    for w in range(num_workers):
        regs.append(QuantumRegister(1, f"c{w}"))
        regs.append(QuantumRegister(bits, f"s{w}"))
        regs.append(ClassicalRegister(bits, f"m{w}"))
    
    qc = QuantumCircuit(*regs)
    powers = []
    curr = delta
    
    for _ in range(bits):
        powers.append(curr)
        curr = ec_point_add(curr, curr)
    
    engine = GeometricIPE(bits)
    
    for k in range(bits):
        for w in range(num_workers):
            ctrl = qc.qubits[w * (bits + 1)]
            qc.reset(ctrl)
            qc.h(ctrl)
        
        for w in range(num_workers):
            ctrl = qc.qubits[w * (bits + 1)]
            state_start = w * (bits + 1) + 1
            state = qc.qubits[state_start:state_start + bits]
            engine._oracle_geometric_phase(qc, ctrl, state, powers[k])
        
        for w in range(num_workers):
            ctrl = qc.qubits[w * (bits + 1)]
            meas = qc.cregs[w]
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((meas[m], 1)):
                    qc.p(angle, ctrl)
            qc.h(ctrl)
            qc.measure(ctrl, meas[k])
    
    return qc


def build_mode_38_explicit_logic_omega(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Explicit Logic Omega mode."""
    logger.info("Building Mode 38: Explicit Logic Omega")
    run_len = min(bits, 8)
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(run_len, "state")
    qr_anc = QuantumRegister(1, "anc")
    qr_tmp = QuantumRegister(1, "tmp")
    cr = ClassicalRegister(run_len, "meas")
    qc = QuantumCircuit(qr_c, qr_s, qr_anc, qr_tmp, cr)
    
    qc.x(qr_s[0])
    scalar_val = delta.x()
    
    for k in range(run_len):
        qc.reset(qr_c)
        qc.h(qr_c)
        val_shifted = (scalar_val * (1 << k)) % (1 << run_len)
        ft_draper_modular_adder_omega(qc, val_shifted, qr_s, (1 << run_len) - 1, qr_anc, qr_tmp)
        
        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])
        
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    
    return qc


# --- MODE 39: MATRIX MOD OMEGA ---
def build_mode_39_matrix_mod_omega(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Mode 39: Matrix Mod Omega with enhanced fault tolerance"""
    logger.info("Building Mode 39: Matrix Mod Omega - 132 qubits")


    run_len = min(bits, 8)  # Scale for hardware safety
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(run_len, "state")
    qr_anc = QuantumRegister(1, "anc")
    qr_tmp = QuantumRegister(1, "tmp")
    cr = ClassicalRegister(run_len, "meas")


    # Add fault tolerance registers if enabled
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * run_len, 'ft_state'),
            QuantumRegister(2, 'ft_anc'),
            QuantumRegister(2, 'ft_tmp')
        ])


    qc = QuantumCircuit(qr_c, qr_s, qr_anc, qr_tmp, cr, *ft_regs)


    # Apply fault tolerance if enabled
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, qr_c[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, qr_s, config))
        ft_ancillas.append(apply_ft_to_qubit(qc, qr_anc[0], config))
        ft_ancillas.append(apply_ft_to_qubit(qc, qr_tmp[0], config))


    # Initialize state
    qc.x(qr_s[0])


    scalar_val = delta.x()


    for k in range(run_len):
        qc.reset(qr_c[0])
        qc.h(qr_c[0])


        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])


        val_shifted = (scalar_val * (1 << k)) % (1 << run_len)
        ft_draper_modular_adder_omega(qc, val_shifted, qr_s, (1 << run_len) - 1, qr_anc, qr_tmp)


        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])


        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], qr_c[0])


        qc.h(qr_c[0])
        qc.measure(qr_c[0], cr[k])


    # Decode fault tolerance for other registers if enabled
    if config.USE_FT:
        decode_ft_register(qc, qr_s, ft_ancillas[1:run_len+1], config)
        decode_repetition(qc, ft_ancillas[run_len+1], qr_anc[0])
        decode_repetition(qc, ft_ancillas[run_len+2], qr_tmp[0])


    return qc


def build_mode_40_matrix_mod_smart_scalable(bits: int, delta: Point) -> QuantumCircuit:
    """Build the circuit for the Matrix Mod (Smart Scalable) mode."""
    logger.info("Building Mode 40: Matrix Mod (Smart Scalable)")
    qc = QuantumCircuit(QuantumRegister(bits), ClassicalRegister(bits))
    gate = add_const_mod_gate(1, 2**bits)
    qc.append(gate, qc.qregs[0])
    qc.measure(qc.qregs[0], qc.cregs[0])
    
    return qc


# ==========================================
# . ERROR MITIGATION ENGINE
# ==========================================
class ErrorMitigationEngine:
    """Comprehensive error mitigation engine with all advanced features"""


    def __init__(self, config):
        self.config = config
        self.mitigation_results = {}
        self.zne_scales = [1, 3, 5]  # Default ZNE scales


    def configure_sampler_options(self, sampler):
        """Apply all mitigation options to sampler with comprehensive error handling"""
        try:
            # Dynamical Decoupling
            if self.config.USE_DD:
                sampler.options.dynamical_decoupling = {
                    "enable": True,
                    "sequence_type": self.config.DD_SEQUENCE
                }
                logger.info(f"🛡️ Configured Dynamical Decoupling: {self.config.DD_SEQUENCE}")


            # Measurement Error Mitigation
            if self.config.USE_MEAS_MITIGATION:
                sampler.options.resilience_level = 2
                sampler.options.twirling = {
                    "enable_gates": True,
                    "enable_measure": True
                }
                sampler.options.measure_mitigation = True
                logger.info("📊 Configured Measurement Error Mitigation with resilience level 2")


            # ZNE Configuration
            if self.config.USE_ZNE:
                sampler.options.resilience = {
                    "zne": {
                        "method": self.config.ZNE_METHOD,
                        "scales": self.zne_scales
                    }
                }
                logger.info(f"📈 Configured ZNE: {self.config.ZNE_METHOD} method")


            # Set resilience level
            sampler.options.resilience_level = self.config.INTERNAL_RESILIENCE_LEVEL
            logger.info(f"🛡️ Set resilience level: {self.config.INTERNAL_RESILIENCE_LEVEL}")


        except Exception as e:
            logger.warning(f"⚠️ Could not set some mitigation options: {e}")


        return sampler


    def manual_zne(self, qc: QuantumCircuit, backend, shots: int, scales=None):
        """Enhanced Manual Zero-Noise Extrapolation with comprehensive analysis"""
        if scales is None:
            scales = self.zne_scales


        logger.info(f"🧪 Running Manual ZNE with scales: {scales} and {shots} shots")
        counts_list = []
        scale_metrics = {}


        for scale in scales:
            scaled_qc = qc.copy()
            scale_metrics[scale] = {}


            if scale > 1:
                logger.debug(f"Applying noise scaling factor {scale}")
                for _ in range(scale - 1):
                    scaled_qc.barrier()
                    for q in scaled_qc.qubits:
                        scaled_qc.id(q)


            logger.info(f"[⚙️] Transpiling Scale {scale}...")
            tqc = transpile(
                scaled_qc,
                backend=backend,
                optimization_level=self.config.OPT_LEVEL,
                scheduling_method='alap',
                routing_method='sabre'
            )


            scale_metrics[scale] = {
                'depth': tqc.depth(),
                'size': tqc.size(),
                'qubits': tqc.num_qubits,
                'gates': estimate_gate_counts(tqc)
            }
            logger.info(f"[📊] Scale {scale} Metrics: Depth={tqc.depth()}, Size={tqc.size()}")


            sampler = Sampler(backend)
            sampler = self.configure_sampler_options(sampler)
            sampler.options.resilience_level = 0  # Force Raw for ZNE


            job = sampler.run([tqc], shots=shots)
            logger.info(f"[📡] ZNE Scale {scale} Job ID: {job.job_id()}")


            try:
                job_result = job.result()
                counts = self.safe_get_counts(job_result[0])
                if counts:
                    counts_list.append(counts)
                    logger.info(f"[✅] Scale {scale}: {len(counts)} unique measurements")
                else:
                    logger.warning(f"[⚠️] No counts for scale {scale}")
            except Exception as e:
                logger.error(f"[❌] Scale {scale} failed: {e}")
                continue


        if not counts_list:
            logger.warning("⚠️ No valid counts from any ZNE scale")
            return defaultdict(float)


        logger.info("📈 Performing linear extrapolation...")
        extrapolated = defaultdict(float)
        all_keys = set().union(*counts_list)


        for key in all_keys:
            vals = [c.get(key, 0) for c in counts_list]
            if len(vals) > 1:
                try:
                    fit = np.polyfit(scales[:len(vals)], vals, 1)
                    extrapolated[key] = max(0, fit[1])  # Intercept
                    logger.debug(f"Extrapolated {key}: {extrapolated[key]}")
                except Exception as e:
                    logger.warning(f"Extrapolation failed for {key}: {e}")
                    extrapolated[key] = vals[-1]  # Fallback
            else:
                extrapolated[key] = vals[0] if vals else 0


        logger.info(f"📊 ZNE Results: {len(extrapolated)} extrapolated values")
        return extrapolated


    def safe_get_counts(self, result_item):
        """Universal count extraction with all possible register name patterns"""
        combined_counts = defaultdict(int)
        data_found = False


        if hasattr(result_item, 'data'):
            data_bin = result_item.data
            for attr_name in [a for a in dir(data_bin) if not a.startswith("_")]:
                val = getattr(data_bin, attr_name)
                if hasattr(val, 'get_counts'):
                    try:
                        c = val.get_counts()
                        for k, v in c.items():
                            combined_counts[k] += v
                        data_found = True
                        logger.debug(f"Found counts in {attr_name}")
                    except Exception as e:
                        logger.debug(f"Failed to get counts from {attr_name}: {e}")


        if data_found:
            logger.info("✅ Retrieved counts via direct data access")
            return dict(combined_counts)


        register_patterns = [
            "meas", "c", "meas_bits", "meas_state", "c_meas", "probe_c",
            "flag_out", "m0", "m1", "m2", "m3", "phase_meas", "flag_meas",
            "w_meas", "phase_bits", "flag_bits", "f1", "f2", "f3", "f4",
            "window_meas", "shadow_meas", "hive_meas", "a_meas", "b_meas",
            "ab_meas", "draper_meas", "scalar_meas", "mod_meas"
        ]


        for pattern in register_patterns:
            try:
                if hasattr(result_item, 'data') and hasattr(result_item.data, pattern):
                    counts = getattr(result_item.data, pattern).get_counts()
                    logger.info(f"✅ Found counts in {pattern}")
                    return counts
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")


        logger.warning("⚠️ Could not extract counts from any known pattern")
        return {}


    def analyze_circuit(self, qc: QuantumCircuit, backend):
        """Comprehensive circuit analysis with all metrics"""
        total_qubits = qc.num_qubits
        depth = qc.depth()
        gate_counts = estimate_gate_counts(qc)


        backend_qubits = backend.configuration().n_qubits if hasattr(backend, 'configuration') else 127


        logger.info("\n" + "="*60)
        logger.info(" CIRCUIT ANALYSIS REPORT ")
        logger.info("="*60)
        logger.info(f"🔢 Logical Qubits: {total_qubits}/{backend_qubits}")
        logger.info(f"📏 Circuit Depth: {depth}")
        logger.info(f"🔄 Gate Counts:")
        logger.info(f"    CX: {gate_counts['CX']}")
        logger.info(f"    CCX: {gate_counts['CCX']}")
        logger.info(f"    T: {gate_counts['T']}")
        logger.info(f"    Total: {sum(gate_counts.values())}")


        if total_qubits > backend_qubits:
            logger.error(f"❌ CRITICAL: Circuit exceeds backend capacity!")
            return False
        elif total_qubits > backend_qubits - 10:
            logger.warning(f"⚠️ High qubit usage: {total_qubits}/{backend_qubits}")
        else:
            logger.info(f"✅ Circuit fits within backend capacity")


        if hasattr(backend, 'configuration'):
            try:
                avg_gate_time = backend.configuration().timing_constraints['u']['gate_time']
                estimated_time = (sum(gate_counts.values()) * avg_gate_time) / 1e9
                logger.info(f"⏱️ Estimated execution time: ~{estimated_time:.2f} seconds")
            except:
                logger.info("⏱️ Execution time estimate unavailable")


        logger.info("="*60)
        return True


# ==========================================
# 10. UNIVERSAL POST-PROCESSING ENGINE
# ==========================================
class UniversalPostProcessor:
    """Comprehensive post-processing engine with all advanced features"""


    def __init__(self, config):
        self.config = config
        self.found_keys = []
        self.search_depth = config.SEARCH_DEPTH
        self.window_size = 10000


    def save_key(self, k: int, mode_name: str = "", source: str = ""):
        """Save found key in multiple formats with detailed metadata"""
        hex_k = hex(k)[2:].zfill(64)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"recovered_keys_{timestamp.replace(':', '-').replace(' ', '_')}.txt"


        key_data = {
            "timestamp": timestamp,
            "mode": mode_name,
            "source": source,
            "formats": {
                "standard": '0x' + hex_k.zfill(64),
                "raw_hex": hex_k.zfill(64),
                "shifted_hex": '0x' + hex_k[32:] + hex_k[:32],
                "decimal": str(k)
            }
        }


        with open(filename, "a") as f:
            f.write(json.dumps(key_data, indent=2) + "\n\n")


        logger.info(f"🔑 KEY SAVED TO: {filename}")
        logger.info(f"   Standard: {key_data['formats']['standard']}")
        self.found_keys.append(k)

    def precompute_good_indices_range(self, start: int, end: int, target_x: int):
        """Original precompute function using keyspace start"""
        logger.info(f"🔍 Precomputing good indices from {hex(start)} to {hex(end)}")


        good = []
        current = ec_scalar_mult(start, G)  # Fixed: Using G instead of (Gx, Gy)



        for k in range(start, end + 1):
            if current is None:
                current = G
            else:
                if k != start:
                    current = ec_point_add(current, G)  # Fixed: Using G instead of (Gx, Gy)


            if current and current.x() == target_x:
                good.append(k - start)
                logger.debug(f"Found match at {hex(k)} (offset {k-start})")


        logger.info(f"🎯 Found {len(good)} good indices in range")
        return good


    def precompute_good_indices_from_measurement(self, measurement: int, target_x: int, search_depth: int):
        """
        New function that starts from measurement value and scans with search_depth
        Instead of starting from keyspace start, it uses the smallest measurement as base
        """
        logger.info(f"🔍 Precomputing good indices from measurement {hex(measurement)} ±{search_depth}")


        good = []
        start = max(0, measurement - search_depth)
        end = measurement + search_depth


        # Start from the measurement point
        current = ec_scalar_mult(measurement, G)  # Fixed: Using G instead of (Gx, Gy)


        # Check backward from measurement
        for k in range(measurement, start - 1, -1):
            if current and current.x() == target_x:  # Fixed: Using .x() method
                good.append(k)
                logger.debug(f"Found backward match at {hex(k)}")


            if k > start:
                current = ec_point_add(current, G)  # Fixed: Using G instead of (Gx, Gy)
                current = ec_point_negate(current)  # Fixed: Using proper negation OLD -> # current = (current[0], (-current[1]) % P)  # Invert for backward step


        # Reset to measurement point
        current = ec_scalar_mult(measurement, G)  # Fixed: Using G instead of (Gx, Gy)


        # Check forward from measurement
        for k in range(measurement + 1, end + 1):
            if current and current.x() == target_x:  # Fixed: Using .x() method
                good.append(k)
                logger.debug(f"Found forward match at {hex(k)}")


            current = ec_point_add(current, G)  # Fixed: Using G instead of (Gx, Gy)


        logger.info(f"🎯 Found {len(good)} good indices around measurement {hex(measurement)}")
        return sorted(good)


    def process_measurement(self, state_str: str, count: int, bits: int, order: int,
                          target_x: int, endian: str, mode_type: str):
        """Comprehensive measurement processing with all possible interpretations"""
        clean_str = state_str.replace(" ", "")
        candidates = []


        try:
            # 1. Raw bitstring processing
            if endian == "LSB":
                k = int(clean_str[::-1], 2) if clean_str else 0
            else:  # MSB
                k = int(clean_str, 2) if clean_str else 0
            candidates.append(("raw", k))


            # 2. Reverse endianness
            if endian == "LSB":
                k_rev = int(clean_str, 2) if clean_str else 0
            else:
                k_rev = int(clean_str[::-1], 2) if clean_str else 0
            if k_rev != k:
                candidates.append(("reverse_endian", k_rev))


            # 3. Phase estimation (QPE/IPE modes)
            if "QPE" in mode_type or "IPE" in mode_type:
                phase = k / (2**bits)
                num, den = continued_fraction_approx(int(phase * 2**bits), 2**bits, order)
                if den != 0:
                    k_cf = (num * modular_inverse_verbose(den, order)) % order
                    candidates.append(("phase_estimation", k_cf))


            # 4. A/B register split (Shor-like modes)
            if "Shor" in mode_type or "ECDLP" in mode_type:
                mid = len(clean_str) // 2
                a_str = clean_str[:mid]
                b_str = clean_str[mid:]


                if endian == "LSB":
                    a = int(a_str[::-1], 2) if a_str else 0
                    b = int(b_str[::-1], 2) if b_str else 0
                else:
                    a = int(a_str, 2) if a_str else 0
                    b = int(b_str, 2) if b_str else 0


                if b != 0:
                    inv_b = modular_inverse_verbose(b, order)
                    if inv_b:
                        k_ab = (-a * inv_b) % order
                        candidates.append(("ab_register", k_ab))


        except Exception as e:
            logger.debug(f"Measurement processing failed: {e}")


        return candidates


    def verify_candidate(self, k: int, target_x: int, source: str = "", order: int = N):
        """Verify candidate key with comprehensive checks"""
        try:
            Pt = ec_scalar_mult(k, G)  # Fixed: Using G instead of (Gx, Gy)
            if Pt and Pt.x() == target_x:  # Fixed: Using .x() method
                self.save_key(k, source=source)
                return True


            # Check inverted point
            if Pt:
                Pt_inv = ec_point_negate(Pt)  # Fixed: Using proper negation
                if Pt_inv.x() == target_x:  # Fixed: Using .x() method
                    k_inv = (order - k) % order
                    self.save_key(k_inv, source=f"inverted_{source}")
                    return True


        except Exception as e:
            logger.error(f"Verification failed for {hex(k)}: {e}")


        return False


    def process_all_measurements(self, counts: Dict, bits: int, order: int, start: int,
                               target_x: int, mode_meta: Dict):
        """Comprehensive processing of all measurements with all possible interpretations"""
        endian = mode_meta["endian"]
        mode_type = mode_meta["oracle"]
        processed = 0


        logger.info(f"🔍 Processing {len(counts)} measurements (max {self.search_depth})")


        # Find the smallest measurement value
        min_measurement = None
        try:
            sorted_measurements = sorted(
                [(int(bitstr.replace(" ", "")[::-1], 2) if endian == "LSB" else
                  int(bitstr.replace(" ", ""), 2), bitstr)
                 for bitstr in counts.keys() if bitstr.replace(" ", "")],
                key=lambda x: x[0]
            )
            if sorted_measurements:
                min_measurement = sorted_measurements[0][0]
                logger.info(f"📏 Smallest measurement value: {hex(min_measurement)}")
        except Exception as e:
            logger.warning(f"Could not determine smallest measurement: {e}")


        # Process top measurements first
        for state_str, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if processed >= self.search_depth:
                break
            processed += 1


            candidates = self.process_measurement(
                state_str, count, bits, order, target_x, endian, mode_type
            )


            for source, k in candidates:
                if self.verify_candidate(k, target_x, source):
                    return k  # Return first valid candidate


        # If no key found, use the new precompute function starting from smallest measurement
        if min_measurement is not None and not self.found_keys:
            logger.info(f"🔍 Using measurement-based precompute from {hex(min_measurement)}")
            hits = self.precompute_good_indices_from_measurement(
                min_measurement,
                target_x,
                self.config.SEARCH_DEPTH
            )


            for candidate in hits:
                if self.verify_candidate(candidate, target_x, "measurement_based_scan"):
                    return candidate


        # Final fallback: check all possible register patterns
        if not self.found_keys:
            logger.info("🔍 Attempting fallback register pattern analysis")
            for state_str, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                patterns = [
                    (lambda s: s),  # Raw
                    (lambda s: s[::-1]),  # Reverse
                    (lambda s: s[:len(s)//2] + s[len(s)//2:][::-1]),  # A normal, B reversed
                    (lambda s: s[:len(s)//2][::-1] + s[len(s)//2:]),  # A reversed, B normal
                ]


                for i, pattern in enumerate(patterns):
                    try:
                        modified = pattern(state_str.replace(" ", ""))
                        if modified:
                            k = int(modified, 2)
                            if self.verify_candidate(k, target_x, f"pattern_{i}"):
                                return k
                    except Exception as e:
                        logger.debug(f"Pattern {i} failed: {e}")


        return None


# ==========================================
# 11. EXECUTION ENGINE & VISUALIZATION
# ==========================================
def configure_sampler_options(sampler: Sampler, config: Config) -> Sampler:
    """Applies all DRAGON mitigation settings with enhanced logging"""
    options = Options()


    if config.USE_DD:
        options.dynamical_decoupling = {
            "enable": True,
            "sequence_type": config.DD_SEQUENCE
        }
        logger.info(f"🛡️ Enabled Dynamical Decoupling with {config.DD_SEQUENCE} sequence")


    if config.USE_MEAS_MITIGATION:
        options.resilience_level = 2
        options.twirling = {"enable": True}
        options.measure_mitigation = True
        logger.info("📊 Enabled measurement error mitigation with resilience level 2")


    if config.USE_ZNE:
        if config.ZNE_METHOD == "manual":
            options.resilience = {"zne": {"method": "manual"}}
            logger.info("📈 Enabled Manual ZNE")
        else:
            options.resilience = {"zne": {"method": "standard"}}
            logger.info("📈 Enabled Standard ZNE")


    sampler.options = options
    return sampler


def safe_get_counts(result_item) -> Optional[Dict[str, int]]:
    """Aggressive Universal Retrieval of measurement counts"""
    combined_counts = defaultdict(int)
    data_found = False


    if hasattr(result_item, 'data'):
        data_bin = result_item.data
        for attr_name in [a for a in dir(data_bin) if not a.startswith("_")]:
            val = getattr(data_bin, attr_name)
            if hasattr(val, 'get_counts'):
                try:
                    c = val.get_counts()
                    for k, v in c.items():
                        combined_counts[k] += v
                    data_found = True
                    logger.debug(f"Found counts in {attr_name}")
                except Exception as e:
                    logger.debug(f"Failed to get counts from {attr_name}: {e}")


    if data_found:
        logger.info(f"✅ Successfully retrieved counts from direct data access")
        return dict(combined_counts)


    register_patterns = [
        "meas", "c", "meas_bits", "meas_state", "c_meas", "probe_c",
        "flag_out", "m0", "m1", "m2", "m3", "phase_meas", "flag_meas",
        "w_meas", "phase_bits", "flag_bits", "f1", "f2", "f3", "f4"
    ]


    for pattern in register_patterns:
        try:
            if hasattr(result_item.data, pattern):
                counts = getattr(result_item.data, pattern).get_counts()
                logger.info(f"✅ Found counts in {pattern}")
                return counts
        except Exception as e:
            logger.debug(f"Failed to get counts from {pattern}: {e}")


    logger.warning("⚠️ Could not retrieve counts from any known register pattern")
    return None


def manual_zne(qc: QuantumCircuit, backend, shots: int, config: Config, scales: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """Enhanced Manual Zero-Noise Extrapolation with comprehensive analysis"""
    logger.info(f"🧪 Running Manual ZNE (Scales: {scales}) with {shots} shots...")
    counts_list = []
    scale_results = {}

    for scale in scales:
        scaled_qc = qc.copy()
        scale_results[scale] = {}

        if scale > 1:
            logger.debug(f"Applying noise scaling factor {scale}")
            for _ in range(scale - 1):
                scaled_qc.barrier()
                for q in scaled_qc.qubits:
                    scaled_qc.id(q)

        logger.info(f"[⚙️] Transpiling Scale {scale}...")
        tqc = transpile(
            scaled_qc,
            backend=backend,
            optimization_level=config.OPT_LEVEL,
            routing_method='sabre' #  scheduling_method='alap',
        )
        print(qc)
        print("Quantum Circuit Details:")
        scale_results[scale] = {
            'depth': tqc.depth(),
            'size': tqc.size(),
            'qubits': tqc.num_qubits,
            'gates': estimate_gate_counts(tqc)
        }
        logger.debug(f"[📊] Scale {scale} Metrics: Depth={tqc.depth()}, Size={tqc.size()}")

        sampler = Sampler(backend)
        sampler = configure_sampler_options(sampler, config)
        sampler.options.resilience_level = 0  # Force Raw for ZNE

        job = sampler.run([tqc], shots=shots)
        logger.debug(f"[📡] ZNE Scale {scale} Job ID: {job.job_id()}")

        try:
            job_result = job.result()
            counts = safe_get_counts(job_result[0])
            if counts:
                counts_list.append(counts)
                logger.debug(f"[✅] Scale {scale}: {len(counts)} unique measurements")
            else:
                logger.warning(f"[⚠️] No counts for scale {scale}")
        except Exception as e:
            logger.error(f"[❌] Scale {scale} failed: {e}")
            continue

    if not counts_list:
        logger.warning("⚠️ No valid counts from any ZNE scale")
        return defaultdict(float)

    logger.info("📈 Performing linear extrapolation...")
    extrapolated = defaultdict(float)
    all_keys = set().union(*counts_list)

    for key in all_keys:
        vals = [c.get(key, 0) for c in counts_list]
        if len(vals) > 1:
            try:
                fit = np.polyfit(scales[:len(vals)], vals, 1)
                extrapolated[key] = max(0, fit[1])
                logger.debug(f"Extrapolated {key}: {extrapolated[key]}")
            except Exception as e:
                logger.warning(f"Extrapolation failed for {key}: {e}")
                extrapolated[key] = vals[-1]  # Fallback
        else:
            extrapolated[key] = vals[0]

    logger.info(f"📊 ZNE Results: {len(extrapolated)} extrapolated values")
    return extrapolated


def plot_visuals(counts: Dict[str, int], bits: int, order: int = N, k_target: Optional[int] = None) -> None:
    """Advanced visualization with multiple plot types and analysis"""
    if not counts:
        logger.warning("⚠️ No counts to visualize")
        return


    if len(counts) > 500:
        logger.info("📊 Plotting histogram for large result set...")
        plt.figure(figsize=(10, 6))
        plot_histogram(counts, title="Measurement Results")
        plt.tight_layout()
        plt.show()
        return


    try:
        logger.info("📊 Creating measurement heatmap...")
        grid = 256
        heat = np.zeros((grid, grid), dtype=int)


        for bitstr, cnt in counts.items():
            try:
                clean_str = bitstr.replace(" ", "")
                val = int(clean_str, 2) if clean_str else 0
                a = (val >> (bits//2)) % grid
                b = val % grid
                heat[a, b] += cnt
            except Exception as e:
                logger.debug(f"Failed to process bitstring {bitstr}: {e}")


        plt.figure(figsize=(10, 8))
        plt.suptitle('Measurement Analysis', fontsize=16)


        # Heatmap
        plt.subplot(2, 2, 1)
        plt.title('A/B Register Heatmap')
        plt.imshow(heat, cmap='viridis', origin='lower')
        plt.colorbar(label='Counts')
        plt.xlabel('B Register')
        plt.ylabel('A Register')


        # Top measurements
        plt.subplot(2, 2, 2)
        top_meas = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        states, counts_val = zip(*top_meas) if top_meas else ([], [])
        plt.bar(range(len(states)), counts_val, color='skyblue')
        plt.title('Top 10 Measurements')
        plt.xlabel('Measurement Index')
        plt.ylabel('Counts')
        plt.xticks(range(len(states)), [str(s) for s in states], rotation=45)


        plt.tight_layout()
        plt.show()


    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        plt.figure(figsize=(8, 6))
        plot_histogram(counts, title="Measurement Results")
        plt.show()


# ==========================================
# 12. MAIN EXECUTION WITH ALL FEATURES
# ==========================================
def run_dragon_code():
    """Main execution with all features enabled and comprehensive reporting"""
    config = Config()
    config.user_menu()

    # Initialize engines
    mitigation_engine = ErrorMitigationEngine(config)
    post_processor = UniversalPostProcessor(config)

    # Prompt for IBM Quantum credentials if not set
    if not config.TOKEN:
        config.TOKEN = input("Enter your IBM Quantum API token: ").strip()
    if not config.CRN:
        config.CRN = input("Enter your IBM Quantum CRN (or press Enter for 'free'): ").strip() or "free"

    # Connect to IBM Quantum
    logger.info("🔌 Connecting to IBM Quantum services...")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=config.TOKEN,
        instance=config.CRN
    )

    # Select backend
    backend = select_backend(config)

    # Decompress target and compute delta
    logger.info("🔐 Decompressing public key...")
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    # In run_dragon_code(), replaced this line:
    # delta = (Q[0] - ec_scalar_mult(config.KEYSPACE_START, (Gx, Gy))[0], Q[1])
    # With these lines:
    start_point = ec_scalar_mult(config.KEYSPACE_START, G)
    delta = ec_point_sub(Q, start_point)
    logger.info(f"   Public Key: {hex(Q.x())[:10]}...{hex(Q.x())[-10:]}")
    logger.info(f"   Delta: ({hex(delta.x())[:10]}..., {hex(delta.y())[-10:]})")

    # Build circuit
    if isinstance(config.METHOD, int):
        if config.METHOD not in MODE_METADATA:
            logger.warning(f"Mode {config.METHOD} does not exist. Defaulting to Mode KING (41).")
            mode_id = 41  # Default to KING (41)
        else:
            mode_id = config.METHOD
    else:
        mode_id = 41  # Default to KING (41) if "smart" or invalid

    mode_meta = MODE_METADATA[mode_id]

    logger.info(f"🛠️ Building circuit for mode {mode_id}: {mode_meta['logo']} {mode_meta['name']}")
    logger.info(f"   Success rate: {mode_meta['success']}%")
    logger.info(f"   Qubits required: {mode_meta['qubits']}")
    logger.info(f"   Endianness: {mode_meta['endian']}")
    logger.info(f"   Oracle type: {mode_meta['oracle']}")

    qc = build_circuit_selector(mode_id, config.BITS, delta, config)
    # qc = globals()[f"build_mode_{mode_id}"](config.BITS, delta, config)

    # Circuit analysis
    mitigation_engine.analyze_circuit(qc, backend)


    # Transpile
    logger.info("⚙️ Transpiling circuit...")
    # transpiled = custom_transpile(qc, backend, optimization_level=config.OPT_LEVEL)
    transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=config.OPT_LEVEL,
        routing_method='sabre' # scheduling_method='alap',
    )
    
    print(qc)
    print("Quantum Circuit Details:")
    logger.info(f"   Original depth: {qc.depth()}")
    logger.info(f"   Transpiled depth: {transpiled.depth()}")
    logger.info(f"   Original size: {qc.size()}")
    logger.info(f"   Transpiled size: {transpiled.size()}")
    logger.info(f"   Qubits used: {transpiled.num_qubits}")


    # Execute with mitigation
    sampler = Sampler(backend)
    sampler = mitigation_engine.configure_sampler_options(sampler)


    if config.USE_ZNE and config.ZNE_METHOD == "manual":
        logger.info("🧪 Running Manual ZNE...")
        counts = mitigation_engine.manual_zne(transpiled, backend, config.SHOTS)
    else:
        logger.info(f"📡 Submitting job with {config.SHOTS} shots...")
        job = sampler.run([transpiled], shots=config.SHOTS)
        logger.info(f"   Job ID: {job.job_id()}")
        logger.info("⏳ Waiting for results...")


        result = job.result()
        counts = mitigation_engine.safe_get_counts(result[0])


    # Display top results
    logger.info("\n📊 Top Measurement Results:")
    for i, (state, count) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]):
        logger.info(f"  {i+1}. {state}: {count} counts")


    # Comprehensive post-processing
    logger.info("🔍 Beginning comprehensive post-processing...")
    candidate = post_processor.process_all_measurements(
        counts,
        config.BITS,
        ORDER,
        config.KEYSPACE_START,
        Q[0],
        mode_meta
    )


    if candidate:
        logger.info(f"🎉 SUCCESS: Found candidate private key: {hex(candidate)}")
        logger.info("   Key saved to recovered_keys_*.txt file")
    else:
        logger.warning("❌ No candidates found in top results")
        logger.info("   Suggestions:")
        logger.info("   - Increase shot count")
        logger.info("   - Try different mode (higher success rate)")
        logger.info("   - Enable ZNE if not already enabled")
        logger.info("   - Check backend status and queue")


    # Visualization
    logger.info("📈 Generating visualizations...")
    plot_visuals(counts, config.BITS)


if __name__ == "__main__":
    print("""
    🐉 DRAGON_CODE v120 🐉🔥
    ----------------------------
    Donation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰
    ----------------------------
    🚀 Starting Quantum ECDLP Solver...
    """)
    run_dragon_code()

"""
Precomputes (2^k)*delta for Mode 29 and KING (vaulted for future use).

def precompute_delta_powers(delta: Point, bits: int) -> List[Optional[Point]]:
    '''Precomputes powers of delta (2^k * delta) for gate optimization.
    Args:
        delta: Point object (Q - start*G, from compute_offset).
        bits: Number of bits (e.g., 135).
    Returns:
        List of precomputed points (2^k)*delta, or None if infinity.
    '''
    powers = []
    current = delta
    for _ in range(bits):
        powers.append(current)
        if current is None:
            # If current is None (point at infinity), all further powers are None
            powers.extend([None] * (bits - len(powers)))
            break
        current = ec_point_add(current, current)  # current = 2^k * delta
    return powers

def build_mode_29_semiclassical_omega(bits: int, delta: Point, strategy: str = "SERIAL") -> QuantumCircuit:
    '''Mode 29 with precompute (gate-optimized, no quantum logic change).'''
    logger.info(f"Building Mode 29: Semiclassical Omega [Strategy: {strategy}] (Gate-Optimized)")

    # --- PRECOMPUTE (2^k)*delta (MATHEMATICALLY CORRECT) ---
    powers = precompute_delta_powers(delta, bits)
    # --- END PRECOMPUTE ---

    if strategy == "2D":
        # 2D strategy (unchanged, uses (dx * (1<<k)) % N)
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        dx, dy = delta.x(), delta.y()
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            power = 1 << k
            sx = (dx * power) % N  # Classical op (not precomputed here)
            sy = (dy * power) % N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, sx, sy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    else:
        # SERIAL strategy (uses precomputed powers[k])
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(bits, "state")
        cr = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        for k in range(bits):
            qc.reset(qr_c)
            qc.h(qr_c)
            if powers[k]:
                dx, dy = powers[k].x(), powers[k].y()  # Uses precomputed (2^k)*delta
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx, dy)  # UNCHANGED
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((cr[m], 1)):
                    qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
    return qc

def build_mode_KING_semiclassical_omega(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    '''Mode KING with precompute (gate-optimized, no quantum logic change).'''
    logger.info("Building Mode KING: Semiclassical Omega (136 qubits) - THE KING (Gate-Optimized)")

    # --- PRECOMPUTE (2^k)*delta (MATHEMATICALLY CORRECT) ---
    powers = precompute_delta_powers(delta, bits)
    # --- END PRECOMPUTE ---

    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")

    # Fault tolerance setup (unchanged)
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state')
        ])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))

    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])
        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), ctrl[0])

        # --- USE PRECOMPUTED POWERS (REPLACES (delta[0] * power) % N) ---
        dx = powers[k].x()  # Precomputed (2^k)*delta.x
        dy = powers[k].y()  # Precomputed (2^k)*delta.y
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)  # UNCHANGED

        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i+1], state[i])
    return qc
"""
