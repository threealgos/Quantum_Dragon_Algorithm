# Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#============================================================================================
""" TODO: The Qiskit Code will Be Converted To Guppy ) Quantum programming language --->  NEXT We Can Use it in Q-Nexus Platformes .

=========🐉 DRAGON_CODE v135 🐉🔥=============
🏆 Ultimate Quantum ECDLP Solver - 15 Optimized Modes
🔢 Features: Full Draper/QPE Oracles, Advanced Mitigation, Smart Mode Selection
💰 Donation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai

📌  Components :
- Multiple Quantum Attacks with best oracles total the First One is mod_0_porb Just for Futur_use for Google Quantum QPU 1 PHisical Qubit ~ 1 million Logical Qubits  .
- Complete Draper 1D/2D/Scalar + QPE oracle implementations
- ZNE Advanced error mitigation both manual/standard
- Powerful post-processing with window scanning
- Full circuit analysis and visualization
- Smart mode selection based on backend capabilities
# For Extra-informations To Save an IBM account Use This save method 
# With Credentials Already included inside the Code.

API_TOKEN = "YOUR_API_TOKEN"
QiskitRuntimeService.save_account(channel="ibm_cloud", token=api_token, overwrite=True)

service = QiskitRuntimeService(
    instance="<CRN>"
)
"""
# 1. IMPORTS & CONSTANTS
# ========================================== 
# Standard libs
from IPython.display import display, HTML
from qiskit.synthesis import synth_qft_full
from qiskit.visualization import plot_histogram, plot_distribution, plot_state_qsphere, plot_bloch_multivector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeLagosV2, FakeManilaV2
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Union, Any
from fractions import Fraction
from ecdsa.ellipticcurve import Point, CurveFp
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, HTML
from qiskit import ClassicalRegister, AncillaRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import HGate, ZGate, MCXGate, RYGate, CXGate, CCXGate
from qiskit.circuit.library import QFTGate, QFT
from qiskit.synthesis import synth_qft_full
from qiskit.visualization import plot_histogram, plot_distribution, plot_state_qsphere, plot_bloch_multivector
from collections import defaultdict
from typing import List, Optional, Dict
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Union, Any
from fractions import Fraction
from ecdsa.ellipticcurve import Point, CurveFp
from mpl_toolkits.mplot3d import Axes3D
try:
    import qiskit_ibm_runtime.options as options_mod
except Exception:
    options_mod = None

# qiskit-ibm-runtime V2-only imports (direct)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_ibm_runtime.options import ResilienceOptionsV2, DynamicalDecouplingOptions, ZneOptions, SamplerExecutionOptionsV2
# Crypto / math helpers and visualization (used elsewhere in your code)
from fractions import Fraction
from ecdsa.ellipticcurve import Point, CurveFp
from typing import Optional, List, Dict, Tuple, Union, Any
from collections import defaultdict
from qiskit_aer import AerSimulator
# V2 runtime imports (direct)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from fractions import Fraction
from ecdsa.ellipticcurve import Point, CurveFp
from collections import Counter, defaultdict
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from typing import Optional, List, Dict
from qiskit.circuit import Parameter
import numpy as np
import logging
import math
import os
import time
import json
from math import gcd
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Union, Any
from types import SimpleNamespace

# Numeric / plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Qiskit core
from qiskit.circuit.library import ZGate, MCXGate, RYGate, QFTGate
from qiskit.circuit.library import UnitaryGate, QFTGate, HGate, CXGate, CCXGate, QFT
from qiskit.synthesis import synth_qft_full
from qiskit.visualization import (
    plot_histogram,
    plot_distribution,
    plot_state_qsphere,
    plot_bloch_multivector,
)
# qiskit-ibm-runtime (V2-first)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeLagosV2, FakeManilaV2
# Optional runtime options helpers (commit-specific; best-effort import)
try:
    import qiskit_ibm_runtime.options as options_mod
except Exception:
    options_mod = None

# Crypto / math helpers used elsewhere
from fractions import Fraction
from ecdsa.ellipticcurve import Point, CurveFp
from qiskit.providers import BackendV2
import pickle, os, time
try:
    # optional DraperQFTAdder
    from qiskit.circuit.library import DraperQFTAdder
except Exception:
    DraperQFTAdder = None

# ==========================================
# 1. CONFIGURATION
# ---------------- Logging ----------------
CACHE_DIR = "cache/"
os.makedirs(CACHE_DIR, exist_ok=True)
# Logging setup 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------
# --- (SECP256k1) Constants  ---
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
ORDER = N
CURVE = CurveFp(P, A, B)
G = Point(CURVE, Gx, Gy)

#---------------------------------------------
# --- Presets Targets---
PRESET_21 = {
    "bits": 21, "start": 0x90000,
    "pub": "037d14b19a95fe400b88b0debe31ecc3c0ec94daea90d13057bde89c5f8e6fc25c",
    "logo": "🔢", "description": "Small key for testing"
}
PRESET_25 = {
    "bits": 25, "start": 0xE00000,
    "pub": "038ad4f423459430771c0f12a24df181ed0da5142ec676088031f28a21e86ea06d",
    "logo": "🔐", "description": "Standard 25-bit key"
}

# ==========================================
# 8. CONFIG, BACKEND SELECTION & HELPERS (V2-only)
# ==========================================

class Config:
    """Configuration class with user menu and transpile override flags."""

    def __init__(self):
        # Default fallback values
        self.BITS = 25
        self.KEYSPACE_START = PRESET_25["start"]
        self.COMPRESSED_PUBKEY_HEX = PRESET_25["pub"]
        self.BACKEND = "ibm_Fez"    # And For Future backends ~1386 Qubits Nighthawk/Kookaburra
        self.TOKEN = None
        self.CRN = None
        self.METHOD = "smart"      # -- Mode Selection Set to integer 0-10 or "smart"
        self.USE_FT = False        # Fault Tolerance
        self.USE_REPETITION = False    # Toggle
        self.USE_ZNE = True
        self.ZNE_METHOD = "manual"  # Default: manual ZNE
        self.USE_DD = True
        self.DD_SEQUENCE = "XY4"   # "XpXm & "XX"
        self.USE_MEAS_MITIGATION = True
        self.SHOTS = 8192          # MAX_SHOTS = 16384 & 100,000 & 1 million
        self.OPT_LEVEL = 3
        self.SEARCH_DEPTH = 10000
        self.USE_ALAP = False
        self.USE_SABRE = True
        self.OVERRIDE_COUPLING_MAP = False
        self.CUSTOM_COUPLING_MAP = None   # avoid it
        self.OVERRIDE_BASIS_GATES = False  # Current Version avoid this too
        self.CUSTOM_BASIS_GATES: Optional[List[str]] = None

    def calculate_keyspace_start(self, bits: int) -> int:
        """
        Calculates a logical starting point for the keyspace.
        Logic: start at 2^(bits-1) + a small offset, or specific hex patterns.
        """
        # Example: for 17 bits, min is 0x10000, max is 0x1FFFF
        min_range = 1 << (bits - 1)
        # We can add an offset or just start at the floor of the bit-range
        return int(min_range)

    def user_menu(self):
        """Interactive configuration menu (adds override prompts)."""
        print("\n" + "="*70)
        print(" 🐉 DRAGON_CODE v135 - CUSTOM TARGET ENABLED 🔥")
        print("="*70)

        # 1. Preset or Custom Selection
        choice = input("Select Target [21 / 25 / Custom]: ").strip().lower()

        if choice == "21":
            self.BITS = PRESET_21["bits"]
            self.KEYSPACE_START = PRESET_21["start"]
            self.COMPRESSED_PUBKEY_HEX = PRESET_21["pub"]
        elif choice == "25":
            self.BITS = PRESET_25["bits"]
            self.KEYSPACE_START = PRESET_25["start"]
            self.COMPRESSED_PUBKEY_HEX = PRESET_25["pub"]
        else:
            # --- CUSTOM LOGIC ---
            self.COMPRESSED_PUBKEY_HEX = input("Enter Custom Compressed PubKey (Hex): ").strip()
            bit_input = input("Enter Target Bit Length (e.g., 17, 20, 30): ").strip()
            self.BITS = int(bit_input) if bit_input else 25

            # Manual Start Input (Optional)
            start_input = input(f"Enter keyspace_start (Hex) [ Press Enter for Auto-Calculation (Optional) ]: ").strip()
            if start_input:
                self.KEYSPACE_START = int(start_input, 16)
            else:
                self.KEYSPACE_START = self.calculate_keyspace_start(self.BITS)
                logger.info(f"✨ Auto-calculated keyspace_start: {hex(self.KEYSPACE_START)}")

        # --- Mode Selection ---
        print("\n📌 Best Available Modes of Attacks :")
        for mode_id, meta in MODE_METADATA.items():
            print(f"  {mode_id}: {meta['logo']} {meta['name']} ({meta['qubits']} qubits)")
        print("-"*70)

        self.METHOD = input("Select Mode_id [1/2/27/29/32/34/35/39/41/42/43/44/99] (default 41): ").strip() or "99"
        mode_id = int(self.METHOD) if self.METHOD != "smart" else 99
        meta = MODE_METADATA.get(mode_id, MODE_METADATA[41])

        self.USE_FT = input(f"Enable Fault Tolerance? [y/n] (default y, supported: {meta['supports_ft']}): ").strip().lower() == 'y' and meta['supports_ft']
        self.USE_REPETITION = input(f"Enable Repetition Codes? [y/n] (default n, supported: {meta['supports_rep']}): ").strip().lower() == 'y' and meta['supports_rep']
        self.USE_ZNE = input("Enable ZNE? [y/n] (default y): ").strip().lower() == 'y'
        if self.USE_ZNE:
            self.ZNE_METHOD = input("ZNE Method [manual/standard] (default manual): ").strip() or "manual"
        self.USE_DD = input("Enable Dynamical Decoupling? [y/n] (default y): ").strip().lower() == 'y'
        if self.USE_DD:
            self.DD_SEQUENCE = input("DD Sequence [XY4/XpXm] (default XY4): ").strip() or "XY4"
        self.USE_ALAP = input("Enable ALAP scheduling? [y/n] (default n): ").strip().lower() == 'y'
        if self.USE_ALAP:
            logger.warning("⚠️ ALAP scheduling may increase circuit depth significantly!")
        self.USE_SABRE = input("Enable SABRE routing? [y/n] (default y): ").strip().lower() == 'y'
        shots = input(f"CURRENT_NUM_SHOTS= 8192 & MAX_SHOTS= 16384 Without mitigation [{self.SHOTS}]: ").strip()
        if shots: self.SHOTS = int(shots)
        search_depth = input(f"Search Depth ~ MAX_SHOTS [{self.SEARCH_DEPTH}]: ").strip()
        if search_depth: self.SEARCH_DEPTH = int(search_depth)
        if input("Enable manual coupling_map override? [y/n] (default n): ").strip().lower() == 'y':
            self.OVERRIDE_COUPLING_MAP = True
            logger.info("Coupling map override enabled — be aware this will override backend target data and may invalidate durations.")
        if input("Enable manual basis_gates override? [y/n] (default n): ").strip().lower() == 'y':
            self.OVERRIDE_BASIS_GATES = True
            bgs = input("Enter basis gates comma-separated (u3,cx,id) default (rz,sx,x): ").strip()
            if bgs:
                self.CUSTOM_BASIS_GATES = [s.strip() for s in bgs.split(",")]

# ==========================================
# 2. MATHEMATICAL UTILITIES
# ==========================================
def gcd_verbose(a: int, b: int) -> int:
    """Compute GCD using Euclidean algorithm."""
    logger.debug(f"Calculating GCD of {hex(a)} and {hex(b)}")
    while b: a, b = b, a % b
    logger.debug(f"GCD result: {hex(a)}")
    return a

def extended_euclidean(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if b == 0: return (a, 1, 0)
    g, x1, y1 = extended_euclidean(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)

def modular_inverse_verbose(a: int, m: int) -> Optional[int]:
    """Compute modular inverse with error handling."""
    try: return pow(a, -1, m)
    except ValueError:
        g, x, y = extended_euclidean(a, m)
        if g != 1:
            logger.warning(f"No inverse exists for {hex(a)} mod {hex(m)}")
            return None
        return x % m

def tonelli_shanks_sqrt(n: int, p: int) -> int:
    """Tonelli-Shanks square root modulo prime."""
    if pow(n, (p - 1) // 2, p) != 1: return 0
    if p % 4 == 3: return pow(n, (p + 1) // 4, p)
    s, e = p - 1, 0
    while s % 2 == 0: s //= 2; e += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1: z += 1
    x = pow(n, (s + 1) // 2, p)
    b, g, r = pow(n, s, p), pow(z, s, p), e
    while True:
        t, m = b, 0
        for m in range(r):
            if t == 1: break
            t = pow(t, 2, p)
        if m == 0: return x
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m

def continued_fraction_approx(num: int, den: int, max_den: int = 1000000) -> Tuple[int, int]:
    """Compute continued fraction approximation."""
    if den == 0:
        logger.warning("Denominator is zero, returning (0, 1)")
        return 0, 1
    common_divisor = gcd_verbose(num, den)
    if common_divisor > 1:
        num //= common_divisor
        den //= common_divisor
        logger.debug(f"Simplified {num}/{den} to {num}/{den}")
    approximation = Fraction(num, den).limit_denominator(max_den)
    logger.debug(f"Best approximation: {approximation.numerator}/{approximation.denominator}")
    return approximation.numerator, approximation.denominator

def decompress_pubkey(hex_key: str) -> Point:
    """Decompress SECP256K1 public key."""
    hex_key = hex_key.lower().replace("0x", "").strip()
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    y_sq = (pow(x, 3, P) + B) % P
    y = tonelli_shanks_sqrt(y_sq, P)
    if y == 0: raise ValueError("Invalid public key")
    if (prefix == 2 and y % 2 != 0) or (prefix == 3 and y % 2 == 0): y = P - y
    return Point(CURVE, x, y)

def ec_point_add(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Elliptic curve point addition."""
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    if x1 == x2 and (y1 + y2) % P == 0: return None
    if x1 == x2:  # Point doubling
        lam = (3 * x1 * x1 + A) * modular_inverse_verbose(2 * y1, P) % P
    else:  # Point addition
        lam = (y2 - y1) * modular_inverse_verbose(x2 - x1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return Point(CURVE, x3, y3)

def ec_point_negate(point: Optional[Point]) -> Optional[Point]:
    """Negate elliptic curve point."""
    if point is None: return None
    return Point(CURVE, point.x(), (-point.y()) % P)

def ec_point_sub(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Point subtraction using ec_point_negate."""
    return ec_point_add(p1, ec_point_negate(p2)) if p2 else p1

def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    """Scalar multiplication on elliptic curve."""
    if k == 0 or point is None: return None
    result = None
    addend = point
    while k:
        if k & 1: result = ec_point_add(result, addend)
        addend = ec_point_add(addend, addend)
        k >>= 1
    return result

def precompute_powers(delta: Point, bits: int) -> List[Point]:
    """Precompute (2^k)*delta for optimization."""
    powers = []
    current = delta
    for _ in range(bits):
        powers.append(current)
        current = ec_point_add(current, current)
    return powers

# ==========================================
# 3. FAULT TOLERANCE & REPETITION CODES
# ==========================================
def prepare_verified_ancilla(qc: QuantumCircuit, qubit, initial_state: int = 0):
    """Prepare a verified ancilla qubit."""
    qc.reset(qubit)
    if initial_state == 1: qc.x(qubit)
    logger.debug(f"Prepared ancilla qubit {qubit} in state {initial_state}")

def encode_repetition(qc: QuantumCircuit, logical_qubit, ancillas: List):
    """Encode 1 logical qubit into 3 physical qubits (repetition code)."""
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])
    logger.debug(f"Encoded logical qubit {logical_qubit} with ancillas {ancillas}")

def decode_repetition(qc: QuantumCircuit, ancillas: List, logical_qubit):
    """Decode 3 physical qubits back to 1 logical qubit (repetition code)."""
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)
    logger.debug(f"Decoded logical qubit {logical_qubit} from ancillas {ancillas}")

def apply_ft_to_qubit(qc: QuantumCircuit, qubit, config):
    """Apply fault tolerance to a single qubit if enabled."""
    if not config.USE_FT: return None
    anc = QuantumRegister(2, f"ft_anc_{qubit}")
    qc.add_register(anc)
    prepare_verified_ancilla(qc, anc[0])
    prepare_verified_ancilla(qc, anc[1])
    encode_repetition(qc, qubit, anc)
    return anc

def decode_ft_qubit(qc: QuantumCircuit, ancillas, logical_qubit):
    """Decode fault tolerance from a single qubit."""
    if ancillas: decode_repetition(qc, ancillas, logical_qubit)

# -------------------------
# New Mode 1 builder (qubit-based QPE helper)
# -------------------------
def apply_QPE_phase_correction_qubit(qc: QuantumCircuit, ctrl_qubit, creg: ClassicalRegister, k: int):
    # Applying QPE phase correction 
    if k <= 0:
        return
    logger.debug(f"[Qubit-helper] Applying QPE phase corrections for k={k} on ctrl {ctrl_qubit}")
    for m in range(k):
        with qc.if_test((creg[m], 1)):
            qc.p(-math.pi / (2 ** (k - m)), ctrl_qubit)

def apply_QPE_phase_correction(qc: QuantumCircuit, ctrl_qubit, creg: ClassicalRegister, k: int):
    """
    QPE correction: Advanced Normalization for Heron/Eagle Stability.
    
    Fixes:
    - Error 6050: Large integer overflow via modulo.
    - Floating Point Drift: Centering angles in the [-pi, pi] range.
    - Precision: Explicit float conversion for hardware controller buffers.
    """
    if k <= 0: return
    
    two_pi = 2.0 * math.pi
    for m in range(k):
        # 1. Standard QPE correction formula
        raw_angle = -math.pi / (2 ** (k - m))
        
        # 2. Symmetric Normalization: Map to (-pi, pi]
        # This keeps the microwave pulses short and mathematically stable
        angle = math.fmod(raw_angle, two_pi)
        if angle <= -math.pi:
            angle += two_pi
        elif angle > math.pi:
            angle -= two_pi
            
        # 3. Explicit casting to avoid Symbolic Parameter errors
        final_angle = float(angle)
        
        # 4. Dynamic Hardware Branching
        with qc.if_test((creg[m], 1)):
            qc.p(final_angle, ctrl_qubit)

# ==========================================
# 4. QUANTUM ORACLES
# ==========================================
def qft_reg(qc: QuantumCircuit, reg: QuantumRegister):
    """Apply QFT to register."""
    logger.debug(f"Applying QFT to {len(reg)} qubits")
    qc.append(synth_qft_full(len(reg), do_swaps=False).to_gate(), reg)

def iqft_reg(qc: QuantumCircuit, reg: QuantumRegister):
    """Apply inverse QFT to register."""
    logger.debug(f"Applying IQFT to {len(reg)} qubits")
    qc.append(synth_qft_full(len(reg), do_swaps=False).inverse().to_gate(), reg)

def _normalize_angle_signed(angle: float) -> float:
    """Normalize to (-pi, pi]. Keeps sign (handy for negative rotations)."""
    two_pi = 2.0 * math.pi
    a = math.fmod(angle, two_pi)
    if a <= -math.pi:
        a += two_pi
    elif a > math.pi:
        a -= two_pi
    return float(a)

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

def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl, target: QuantumRegister, dx: int, dy: int = 0):
    """
    1D Draper Oracle: Mathematically Perfect & Hardware Safe.
    Ensures 25-bit precision by reducing integers before float conversion.
    """
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        # Determine the period for this specific qubit
        divisor = 2 ** (i + 1)
        # Integer Modulo: Prevents floating point precision loss on large keys
        dx_reduced = int(dx % divisor)
        
        # Resulting angle is safe: [0, 2*pi]
        angle = float((2.0 * math.pi * dx_reduced) / divisor)
        
        if ctrl is not None: qc.cp(angle, ctrl, target[i])
        else: qc.p(angle, target[i])
    iqft_reg(qc, target)

def draper_adder_oracle_2d(qc: QuantumCircuit, ctrl, target: QuantumRegister, dx: int, dy: int):
    """
    2D Draper Oracle: Heron-Tuned for 25-bit ECDLP.
    Uses 'Sum-Modulo' to keep pulse parameters minimally small.
    """
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        divisor = 2 ** (i + 1)
        
        # OPTIMIZATION: Sum the coordinates in integer space BEFORE the modulo
        # This keeps the final angle perfectly centered and accurate
        combined_reduced = int((dx + dy) % divisor)
        
        angle = float((2.0 * math.pi * combined_reduced) / divisor)
        
        if ctrl is not None: qc.cp(angle, ctrl, target[i])
        else: qc.p(angle, target[i])
    iqft_reg(qc, target)

def ft_draper_modular_adder(qc: QuantumCircuit, ctrl, target_reg: QuantumRegister,
                          ancilla_reg: QuantumRegister, value: int, modulus: int = N):
    """Fault-tolerant Draper modular adder."""
    n = len(target_reg)
    temp_overflow = ancilla_reg[0]
    qft_reg(qc, target_reg)
    draper_adder_oracle_1d_serial(qc, ctrl, target_reg, value)
    draper_adder_oracle_1d_serial(qc, None, target_reg, -modulus)
    iqft_reg(qc, target_reg)
    qc.cx(target_reg[-1], temp_overflow)
    qft_reg(qc, target_reg)
    qc.cx(temp_overflow, target_reg[-1])
    draper_adder_oracle_1d_serial(qc, temp_overflow, target_reg, modulus)
    qc.cx(temp_overflow, target_reg[-1])
    iqft_reg(qc, target_reg)
    qc.reset(temp_overflow)
    logger.debug(f"Applied fault-tolerant Draper adder with value {value} mod {modulus}")

def ft_draper_modular_adder_omega(qc: QuantumCircuit, value: int, target_reg: QuantumRegister,
                                modulus: int, ancilla_reg: QuantumRegister, temp_reg: QuantumRegister):
    """Fault-tolerant Draper modular adder for Omega modes."""
    return ft_draper_modular_adder(qc, None, target_reg, ancilla_reg, value, modulus)

class QPEOracle:
    """Complete QPE Oracle Implementation."""
    def __init__(self, n_bits: int):
        self.n = n_bits

    def oracle_phase(self, qc: QuantumCircuit, ctrl, point_reg: QuantumRegister,
                    delta_point: Point, k_step: int, order: int = ORDER):
        """Apply QPE phase oracle."""
        if delta_point is None: return
        dx, dy = delta_point.x(), delta_point.y()
        power = 1 << k_step
        const_x = (dx * power) % order
        if const_x != 0: draper_adder_oracle_1d_serial(qc, ctrl, point_reg, const_x)
        logger.debug(f"Applied QPE phase oracle with power={power}, const_x={hex(const_x)[:10]}...")

class GeometricQPE:
    """Enhanced Geometric QPE implementation."""
    def __init__(self, n_bits: int):
        self.n = n_bits

    def _oracle_geometric_phase(self, qc: QuantumCircuit, ctrl, state_reg: QuantumRegister, point_val: Point):
        """Apply geometric phase oracle."""
        if point_val is None: return
        vx = point_val.x()
        for i in range(self.n):
            angle_x = 2 * math.pi * vx / (2 ** (i + 1))
            if ctrl is not None: qc.cp(angle_x, ctrl, state_reg[i])
            else: qc.p(angle_x, state_reg[i])
        logger.debug(f"Applied geometric QPE oracle with vx={hex(vx)[:10]}...")

def modified_shor_oracle(qc: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister,
                        state_reg: QuantumRegister, points: List[Point], ancilla_reg: QuantumRegister):
    """Modified Shor-style oracle for ECDLP."""
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        if pt: ft_draper_modular_adder(qc, a_reg[i], state_reg, ancilla_reg, pt.x() % N)
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)]
        if pt: ft_draper_modular_adder(qc, b_reg[i], state_reg, ancilla_reg, pt.x() % N)
    logger.debug("Applied modified Shor oracle")

def ecdlp_oracle_ab(qc: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister,
                   point_reg: QuantumRegister, points: List[Point], ancilla_reg: QuantumRegister):
    """ECDLP oracle implementation for AB registers."""
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        if pt: ft_draper_modular_adder(qc, a_reg[i], point_reg, ancilla_reg, pt.x() % N)
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)]
        if pt: ft_draper_modular_adder(qc, b_reg[i], point_reg, ancilla_reg, pt.x() % N)
    logger.debug("Applied ECDLP oracle")

# ==========================================
# 5. MODE METADATA (added Mode 1 and Mode 2)
# ==========================================
MODE_METADATA = {
    0: {"name": "Hardware QPE Diagnostic 👻 ", "qubits": 136, "infos": 50, "endian": "LSB", "oracle": "Diagnostic", "logo": "🔢", "supports_ft": False, "supports_rep": False, "method": "phase"},
    1: {"name": "QPE Standard (qubit helper) 🔁", "qubits": 136, "infos": 65, "logo": "🔁", "oracle": "QPE_STD", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    2: {"name": "QPE Adaptive (no-dynamic) ⚙️", "qubits": 136, "infos": 70, "logo": "⚙️", "oracle": "QPE_ADAPT", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    27: {"name": "Advanced QPE", "qubits": 136, "infos": 70, "endian": "LSB", "oracle": "AdvancedQPE", "logo": "📊", "supports_ft": False, "supports_rep": False, "method": "phase"},
    28: {"name": "Full Quantum 🕸️ Optimized QPE ", "qubits": 156, "infos": 55, "endian": "MSB", "oracle": "FullQuantum", "logo": "💎", "supports_ft": False, "supports_rep": False, "method": "phase"},
    29: {"name": "QPE Omega 🏆", "qubits": 136, "infos": 75, "logo": "🏆", "oracle": "QPE", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    32: {"name": "FT-Draper Omega 🛡️", "qubits": 128, "infos": 68, "logo": "🛡️", "oracle": "DraperFT", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    34: {"name": "Modified Shor-Omega 🧠", "qubits": 156, "infos": 70, "logo": "🧠", "oracle": "ModifiedShor", "endian": "MSB", "supports_ft": False, "supports_rep": False, "method": "a&b"},
    35: {"name": "Geo-QPE Omega 🌍", "qubits": 134, "infos": 76, "logo": "🌍", "oracle": "GeoQPE", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    39: {"name": "Matrix Mod Omega 🌀", "qubits": 132, "infos": 78, "logo": "🌀", "oracle": "Matrix", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    41: {"name": "Shor 👑", "qubits": 136, "infos": 85, "logo": "👑", "oracle": "QPEShor", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    42: {"name": "Hive-Shor 🐝👑", "qubits": 127, "infos": 82, "logo": "🐝👑", "oracle": "HiveShor", "endian": "LSB", "supports_ft": False, "supports_rep": False, "method": "hive"},
    43: {"name": "Hive-Omega 🐝🏆", "qubits": 127, "infos": 78, "logo": "🐝🏆", "oracle": "HiveOmega", "endian": "LSB", "supports_ft": False, "supports_rep": False, "method": "hive"},
    44: {"name": "Matrix Unitary 🆕", "qubits": 135, "infos": 80, "logo": "📦", "oracle": "MatrixUnitary", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "phase"},
    99: {"name": "ULTIMATE_SHOR ⚡", "qubits": 136, "infos": 85, "logo": "⚡", "oracle": "Auto", "endian": "LSB", "supports_ft": True, "supports_rep": False, "method": "auto"}
}

# ==========================================
# 6. CIRCUIT BUILDERS (Updated and extended)
# ==========================================
def build_mode_0_hardware_probe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Hardware QPE Diagnostic mode with FT support"""
    logger.info("Building Mode 0: Hardware QPE Diagnostic (136 qubits)")
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

def build_mode_27_advanced_qpe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    """Build the circuit for the Advanced QPE mode."""
    logger.info(f"Building Mode 27: Advanced QPE Corrections")
 
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
 
    return qc

def build_mode_28_full_quantum_optimized(bits: int, delta: Point, config: Config) -> QuantumCircuit:
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

def build_mode_29_QPE_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 29: QPE Omega (iterative full Quantum phase corrections)."""
    logger.info("🛠️ Building Mode 29 : QPE Omega 🏆 (with QPE corrections)")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    ft_anc = None
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+2 qubits)...")
        ft_anc = QuantumRegister(2, "ft_anc_shared")
        qc = QuantumCircuit(ctrl, state, creg, ft_anc)
    else:
        qc = QuantumCircuit(ctrl, state, creg)

    # Initialize control and state
    logger.info("   - Initializing control and state...")
    qc.h(ctrl[0])
    qc.x(state[0])

    powers = precompute_powers(delta, bits)
    for k in range(bits):
        logger.info(f"   - Iteration {k+1}/{bits}: QPE QPE step")
        # Prepare control qubit in |+>
        qc.h(ctrl[0])
        logger.info(f"  - Applied Hadamard gate to control qubit...")
        # Apply QPE phase corrections based on previously measured bits
        if k > 0:
            apply_QPE_phase_correction(qc, ctrl[0], creg, k)

        # Oracle (controlled phase/driven adder)
        dx, dy = powers[k].x(), powers[k].y()
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)

        # Finish with H and measure control
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

        # Prepare for next iteration
        if k < bits - 1:
            qc.reset(ctrl[0])
        if config.USE_FT:
            qc.reset(ft_anc)
            qc.cx(ctrl[0], ft_anc[0])
            qc.cx(ctrl[0], ft_anc[1])

    logger.info("   - Final circuit for Mode 29 built")
    print(qc)
    return qc


def build_mode_32_ft_draper_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 32: FT-Draper Omega (iterative, QPE corrections included)."""
    logger.info("🛠️ Building Mode 32 : FT-Draper Omega 🛡️ (with QPE corrections)")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    anc = QuantumRegister(2, "anc")
    creg = ClassicalRegister(bits, "meas")
    ft_anc = None
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+4 qubits)...")
        ft_anc = QuantumRegister(2, "ft_anc_shared")
        qc = QuantumCircuit(ctrl, state, anc, creg, ft_anc)
    else:
        qc = QuantumCircuit(ctrl, state, anc, creg)

    # Initialize
    logger.info("   - Initializing control and state...")
    qc.h(ctrl[0])
    qc.x(state[0])
    logger.info(f"  - Applied Hadamard gate to control qubit...")
    powers = precompute_powers(delta, bits)
    for k in range(bits):
        logger.info(f"   - Iteration {k+1}/{bits}: FT Draper QPE step")
        qc.h(ctrl[0])

        # QPE phase correction from previous measurement bits
        if k > 0:
            apply_QPE_phase_correction(qc, ctrl[0], creg, k)

        # Fault-tolerant Draper modular adder (controlled)
        dx = powers[k].x()
        ft_draper_modular_adder(qc, ctrl[0], state, anc, dx)

        # Finish and measure
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

        if k < bits - 1:
            qc.reset(ctrl[0])
        if config.USE_FT:
            qc.reset(ft_anc)
            qc.cx(ctrl[0], ft_anc[0])
            qc.cx(ctrl[0], ft_anc[1])

    logger.info("   - Final circuit for Mode 32 built")
    print(qc)
    return qc

def build_mode_34_modified_shor_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 34: Modified Shor-Omega (uses synth_qft_full for symmetry via qft_reg/iqft_reg)."""
    logger.info("🛠️ Building Mode 34 : Modified Shor-Omega 🧠 (QFT symmetry enforced)")
    a_reg = QuantumRegister(bits, "a")
    b_reg = QuantumRegister(bits, "b")
    state = QuantumRegister(bits, "state")
    anc = QuantumRegister(4, "anc")
    creg = ClassicalRegister(2 * bits, "meas")
    qc = QuantumCircuit(a_reg, b_reg, state, anc, creg)

    # Initialize in superposition
    logger.info("   - Initializing A and B registers in superposition...")
    qc.h(a_reg)
    qc.h(b_reg)
    qc.x(state[0])
    logger.info(f"  - Applied Hadamard gate to control qubit...")
    points = precompute_powers(delta, bits)
    logger.info("   - Applying modified Shor oracle...")
    modified_shor_oracle(qc, a_reg, b_reg, state, points, anc)

    logger.info("   - Applying inverse QFT (synth_qft_full based) to A and B registers for symmetry")
    iqft_reg(qc, a_reg)
    iqft_reg(qc, b_reg)

    qc.measure(a_reg, creg[:bits])
    qc.measure(b_reg, creg[bits:])

    logger.info("   - Final circuit for Mode 34 built")
    print(qc)
    return qc


def build_mode_35_geo_QPE_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 35: Geo-QPE Omega (use iqft_reg instead of QFTGate to maintain synthesis symmetry)."""
    logger.info("🛠️ Building Mode 35 : Geo-QPE Omega 🌍 (QFT symmetry enforced)")
    ctrl = QuantumRegister(bits, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    ft_regs = []
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+2 qubits)...")
        ft_regs.append(QuantumRegister(2, "ft_anc"))
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)

    # Initialize: apply forward QFT to control register (qft_reg uses synth_qft_full)
    logger.info("   - Applying forward QFT to control register...")
    qft_reg(qc, ctrl)
    qc.x(state[0])
    logger.info("   - Applied QFT to control register")

    powers = precompute_powers(delta, bits)
    geo_oracle = GeometricQPE(bits)
    for k in range(bits):
        logger.info(f"   - Step {k+1}/{bits}: Applying geometric QPE oracle on ctrl[{k}]")
        # The geometric oracle expects a control qubit and applies controlled phases
        geo_oracle._oracle_geometric_phase(qc, ctrl[k], state, powers[k])

    # Use the synth_qft_full inverse consistently
    iqft_reg(qc, ctrl)
    qc.measure(ctrl, creg)

    logger.info("   - Final circuit for Mode 35 built")
    print(qc)
    return qc

def build_mode_39_matrix_mod_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 39: Matrix Mod Omega (use iqft_reg for inverse QFT symmetry)."""
    logger.info("🛠️ Building Mode 39 : Matrix Mod Omega 🌀 (QFT symmetry enforced)")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    ft_regs = []
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+2 qubits)...")
        ft_regs.append(QuantumRegister(2, "ft_anc"))
    qc = QuantumCircuit(state, creg, *ft_regs)

    # Initialize in superposition
    logger.info("   - Initializing qubits in superposition...")
    qc.h(state)
    qc.x(state[0])
    dx = delta.x()
    for i in range(bits):
        logger.debug(f"   - Applying phase rotation for qubit {i}")
        # Keep the original phase mapping; only QFT symmetry is changed
        qc.p(2 * math.pi * dx / (2 ** (bits - i)), state[i])

    # Use iqft_reg for inverse transform (synth_qft_full.inverse())
    iqft_reg(qc, state)
    qc.measure(state, creg)

    logger.info("   - Final circuit for Mode 39 built")
    print(qc)
    return qc

def build_mode_41_Shor(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 41: Shor ( full QPE corrections and FT support)."""
    logger.info("🛠️ Building Mode 41 : Shor 👑 (with QPE corrections)")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    ft_anc = None
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+2 qubits)...")
        ft_anc = QuantumRegister(2, "ft_anc_shared")
        qc = QuantumCircuit(ctrl, state, creg, ft_anc)
    else:
        qc = QuantumCircuit(ctrl, state, creg)

    # Initialize control and state
    logger.info("   - Initializing qubits in Superpositions state...")
    qc.h(ctrl[0])
    qc.x(state[0])
    logger.debug(f" Applied Hadamard gate to control qubits")
    powers = precompute_powers(delta, bits)
    for k in range(bits):
        logger.info(f"   - Iteration {k+1}/{bits}: Building circuit ")
        qc.h(ctrl[0])

        # QPE phase correction (if previous bits exist)
        if k > 0:
            apply_QPE_phase_correction(qc, ctrl[0], creg, k)

        dx, dy = powers[k].x(), powers[k].y()
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)

        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

        if k < bits - 1:
            qc.reset(ctrl[0])
        if config.USE_FT:
            qc.reset(ft_anc)
            qc.cx(ctrl[0], ft_anc[0])
            qc.cx(ctrl[0], ft_anc[1])

    logger.info("   - Final circuit for Mode 41 built")
    print(qc)
    return qc

def build_mode_42_hive_Shor(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 42: Corrected Hive-Shor (Parallel Worker QPE)."""
    workers = 4
    # Ensure each worker handles a fair share of the total bits
    state_bits = bits // workers 
    ctrl = QuantumRegister(workers, "ctrl")
    state = QuantumRegister(state_bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)
    logger.info(f"   - Preparing {workers} workers with {state_bits} qubits each...")
    powers = precompute_powers(delta, bits)
    # Initialize all worker qubits to superposition
    qc.h(ctrl)
    # Initialize state to |1>
    qc.x(state[0])
    for w in range(workers):
        logger.info(f"   - Hive-Worker {w+1}/{workers}: Applying parallel oracle steps...")
        for k in range(state_bits):
            idx = w * state_bits + k
            # Safety check to not exceed precomputed powers
            if idx >= len(powers):
                break
            dx = powers[idx].x()
            # 1. Apply phase corrections for iterative QPE
            if k > 0:
                # We use the index relative to this worker's classical bits
                apply_QPE_phase_correction(qc, ctrl[w], creg, k)
            # 2. Apply bit-specific oracle (Controlled Draper)
            draper_adder_oracle_1d_serial(qc, ctrl[w], state, dx)
            # 3. Final Hadamard for the worker before measurement
            qc.h(ctrl[w])
            # 4. MEASURE AND RECYCLE: Map the worker result to the global classical bit
            qc.measure(ctrl[w], creg[idx])  
            # 5. Reset worker for next bit in its stack
            if k < state_bits - 1:
                qc.reset(ctrl[w])
                qc.h(ctrl[w])

    logger.info("✅ Mode 42 (Hive-Shor) circuit built successfully.")
    return qc

def build_mode_43_hive_omega(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 43: Hive-Omega (127 qubits)."""
    logger.info("🛠️ Building Mode 43: Hive-Omega 🐝🏆")
    workers = 4
    state_bits = bits // workers
    ctrl = QuantumRegister(workers, "ctrl")
    state = QuantumRegister(state_bits, "state")
    creg = ClassicalRegister(bits, "meas")
    qc = QuantumCircuit(ctrl, state, creg)

    # Initialize in superposition
    logger.info(f"   - Preparing {workers} workers with {state_bits} qubits each...")
    for w in range(workers):
        qc.h(ctrl[w])  # Each worker gets their own fresh qubit
        qc.x(state[0])
    logger.info(f"  - Applied Hadamard gate to control workers...")
    powers = precompute_powers(delta, bits)
    for w in range(workers):
        logger.info(f"   - Worker {w+1}/{workers}: Applying Hive-Omega oracle...")
        for k in range(state_bits):
            idx = w * state_bits + k
            dx, dy = powers[idx].x(), powers[idx].y()
            draper_adder_oracle_2d(qc, ctrl[w], state, dx, dy)
        qc.measure(ctrl[w], creg[w * state_bits:(w + 1) * state_bits])

    logger.info("   - Final circuit:")
    print(qc)
    return qc

def build_mode_44_matrix_unitary(bits: int, delta: Point, config) -> QuantumCircuit:
    """Mode 44: Matrix Unitary (use iqft_reg instead of explicit QFTGate for symmetry)."""
    logger.info("🛠️ Building Mode 44 : Matrix Unitary 🆕 (QFT symmetry enforced)")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    ft_regs = []
    if config.USE_FT:
        logger.info("   - Enabling fault tolerance (+2 qubits)...")
        ft_regs.append(QuantumRegister(2, "ft_anc"))
    qc = QuantumCircuit(state, creg, *ft_regs)

    # Initialize
    logger.info("   - Initializing qubits in superposition...")
    qc.h(state)
    qc.x(state[0])
    dx = delta.x()
    for i in range(bits):
        logger.debug(f"   - Applying unitary phase for bit {i}")
        qc.p(2 * math.pi * dx / (2 ** (bits - i)), state[i])

    # Use synth_qft_full inverse for symmetry
    iqft_reg(qc, state)
    qc.measure(state, creg)

    logger.info("   - Final circuit for Mode 44 built")
    print(qc)
    return qc

# -------------------------
# New Mode 1 builder (qubit-based QPE helper)
# -------------------------
def build_mode_1_QPE_standard_qubit(bits: int, delta: Point, config) -> QuantumCircuit:
    """
    Mode 1: QPE Standard using qubit-based QPE helper (call with ctrl[0]).
    Returns a full iterative QPE QPE circuit that uses mid-circuit conditional rotations
    (qc.if_test). Requires backend support for dynamic circuits to run on hardware.
    """
    logger.info("🛠️ Building Mode 1: QPE Standard (qubit helper) 🔁")
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(bits, "meas")
    regs = [ctrl, state, creg]

    if config.USE_FT:
        ft_anc = QuantumRegister(2, "ft_anc")
        regs.append(ft_anc)
    else:
        ft_anc = None

    qc = QuantumCircuit(*regs)

    # Initialize control and state
    logger.info("   - Initializing qubits in superposition...")
    qc.h(ctrl[0])
    qc.x(state[0])
    logger.info("  - Applied Hadamard gate to control qubit and prepared state register")

    powers = precompute_powers(delta, bits)
    for k in range(bits):
        logger.info(f"   - Mode1 Iteration {k+1}/{bits}")

        # For iterations > 0 reset and re-prepare control into |+>
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])

        # If FT enabled, prepare ancillas and encode control logical qubit
        if config.USE_FT:
            # regs[-1] should be ft_anc
            prepare_verified_ancilla(qc, regs[-1][0])
            prepare_verified_ancilla(qc, regs[-1][1])
            encode_repetition(qc, ctrl[0], regs[-1])

        # Apply QPE phase corrections based on previously measured qubits
        apply_QPE_phase_correction_qubit(qc, ctrl[0], creg, k)

        # Oracle: compute powers and call the serial 1D Draper adder by default
        dx = powers[k].x()
        dy = powers[k].y()
        draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, dy)

        # Decode repetition encoding if used
        if config.USE_FT:
            decode_repetition(qc, regs[-1], ctrl[0])

        # Final Hadamard on control and measurement of the bit
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])

    logger.info("   - Final circuit for Mode 1 built")
    print(qc)
    return qc
    
# -------------------------
def build_mode_2_iteration(bits: int, delta: Point, config, k: int, measured_bits: List[int]) -> QuantumCircuit:
    ctrl = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg = ClassicalRegister(1, f"meas_k_{k}")
    regs = [ctrl, state, creg]

    if config.USE_FT:
        ft_anc = QuantumRegister(2, "ft_anc")
        regs.append(ft_anc)
    else:
        ft_anc = None

    qc = QuantumCircuit(*regs)

    # Prepare control and state fresh each iteration
    logger.info("   - Initializing qubits for adaptive iteration...")
    qc.h(ctrl[0])
    qc.x(state[0])

    # FT encoding if requested
    if config.USE_FT:
        prepare_verified_ancilla(qc, regs[-1][0])
        prepare_verified_ancilla(qc, regs[-1][1])
        encode_repetition(qc, ctrl[0], regs[-1])

    # Compute accumulated phase rotation from previous bits and apply
    total_angle = 0.0
    for m, bit in enumerate(measured_bits):
        if bit:
            total_angle += -math.pi / (2 ** (k - m))
    if abs(total_angle) > 0:
        qc.p(total_angle, ctrl[0])

    # Oracle (serial 1D Draper adder)
    power = 1 << k
    dx = (delta.x() * power) % N
    dy = (delta.y() * power) % N
    draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, dy)

    # FT decode if used
    if config.USE_FT:
        decode_repetition(qc, regs[-1], ctrl[0])

    qc.h(ctrl[0])
    qc.measure(ctrl[0], creg[0])

    # print for interactive debugging/inspection
    print(qc)
    return qc

def run_mode_2_adaptive(bits: int, delta: Point, config, backend: Optional[object] = None, shots: int = 1024) -> Dict[str, int]:
    """
    🔄 ADAPTIVE QPE MODE 2 RUNNER (FINAL VERSION)
    - Builds circuits iteratively with proper transpilation
    - Supports manual ZNE with IBM-compatible settings
    - Includes coupling map overrides with validation
    - Robust error handling and fallback mechanisms
    """
    logger.info("\n" + "="*70)
    logger.info("🔄 STARTING ADAPTIVE QPE MODE 2")
    logger.info(f"📏 Target: {bits} bits | 🎯 Shots: {shots}")
    logger.info("="*70)

    # Initialize components
    aer_sim = None
    runtime_sampler = None
    use_runtime_sampler = False
    eme = ErrorMitigationEngine(config)

    # Backend selection with validation
    if backend is None:
        logger.warning("⚠️ No backend provided! Falling back to AerSimulator")
        aer_sim = AerSimulator()
    else:
        try:
            if not hasattr(backend, 'target'):
                logger.error("❌ Invalid backend object! Must have .target attribute")
                raise ValueError("Invalid backend")

            logger.info("🔧 Building SamplerV2 with configuration...")
            runtime_sampler, runtime_opts = eme.build_sampler(backend, manual_zne=(config.ZNE_METHOD == "manual"))
            use_runtime_sampler = True
            logger.info("✅ SamplerV2 constructed successfully")
        except Exception as e:
            logger.error(f"❌ SamplerV2 construction failed: {str(e)}")
            logger.info("🔄 Falling back to AerSimulator")
            aer_sim = AerSimulator()
            use_runtime_sampler = False

    measured_bits = []
    logger.info(f"📊 Beginning adaptive measurement (0/{bits} bits completed)")

    for k in range(bits):
        logger.info(f"\n🔹 Iteration {k+1}/{bits} started...")
        logger.info(f"📝 Building circuit for bit {k}")

        # Build circuit
        qc_iter = build_mode_2_iteration(bits, delta, config, k, measured_bits)

        # Transpile with proper settings and overrides
        try:
            logger.info("🔧 Transpiling circuit...")

            transpile_kwargs = {
                "optimization_level": getattr(config, "OPT_LEVEL", 1),
                "routing_method": 'sabre' if getattr(config, "USE_SABRE", False) else None,
                "scheduling_method": 'alap' if getattr(config, "USE_ALAP", False) else None
                # No basis_gates - let backend.target handle it
            }

            # Apply coupling map override if enabled
            if getattr(config, "OVERRIDE_COUPLING_MAP", False) and getattr(config, "CUSTOM_COUPLING_MAP", None):
                transpile_kwargs["coupling_map"] = config.CUSTOM_COUPLING_MAP
                logger.warning("⚠️ Using CUSTOM coupling map override!")

            target_backend = runtime_sampler.backend if use_runtime_sampler else backend
            transpiled = transpile(qc_iter, target_backend, **transpile_kwargs)

            # Validate transpilation for IBM compatibility
            if any(instr.operation.name == "h" for instr in transpiled.data):
                logger.error("❌ Hadamard gates detected after transpilation!")
                logger.info("🔧 Attempting emergency decomposition...")
                transpiled = transpile(transpiled, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
                if any(instr.operation.name == "h" for instr in transpiled.data):
                    raise RuntimeError("Critical: Could not decompose Hadamard gates!")

            logger.info("✅ Transpilation successful")

        except Exception as e:
            logger.error(f"❌ Transpilation failed: {str(e)}")
            logger.info("🔄 Using original circuit (no transpilation)")
            transpiled = qc_iter

        # Execute circuit
        bit_result = 0
        try:
            if getattr(config, "USE_ZNE", False) and getattr(config, "ZNE_METHOD", "manual") == "manual":
                logger.info("🧪 Running MANUAL ZNE...")
                counts_map = eme.manual_zne(transpiled, backend if backend is not None else aer_sim, shots)
                if counts_map:
                    best = max(counts_map.items(), key=lambda x: x[1])[0]
                    bit_result = int(best.replace(" ", "")[-1])
                else:
                    logger.warning("⚠️ ZNE produced no counts! Defaulting to 0")
            else:
                if use_runtime_sampler:
                    logger.info("🚀 Submitting to runtime sampler...")
                    job = runtime_sampler.run([(transpiled,)], shots=shots)
                    res = job.result()
                    counts = safe_get_counts(res) or {}
                else:
                    logger.info("💻 Running on AerSimulator...")
                    job = aer_sim.run(transpiled, shots=shots)
                    res = job.result()
                    counts = res.get_counts()

                if counts:
                    best = max(counts.items(), key=lambda x: x[1])[0]
                    bit_result = int(best.replace(" ", "")[-1])
                else:
                    logger.warning("⚠️ No counts retrieved! Defaulting to 0")

        except Exception as e:
            logger.error(f"❌ Execution failed: {str(e)}")
            bit_result = 0

        measured_bits.append(bit_result)
        logger.info(f"💾 Measured bit {k} = {bit_result}")
        logger.info(f"📊 Progress: {len(measured_bits)}/{bits} bits completed")

    logger.info("\n" + "="*70)
    logger.info("🎉 ADAPTIVE QPE COMPLETE!")
    logger.info(f"📜 Final Result: {''.join(map(str, measured_bits))}")
    logger.info("="*70)
    return {''.join(map(str, measured_bits)): shots}
 
# -------------------------
# Keep build_mode_99_ultimate_shor as before (unchanged)
# -------------------------
def build_mode_99_ultimate_shor(bits: int, delta: Point, config, backend_qubits: int) -> QuantumCircuit:
    """Mode 99: ULTIMATE_SHOR (auto-selects best mode)."""
    logger.info("🛠️ Building Mode 99: ULTIMATE_SHOR ⚡")
    if backend_qubits >= 156:
        logger.info("   - Selected Shor mode (best for large backends).")
        return build_mode_41_Shor(bits, delta, config)
    elif backend_qubits >= 136:
        logger.info("   - Selected QPE Omega mode.")
        return build_mode_29_QPE_omega(bits, delta, config)
    else:
        logger.info("   - Selected Hive-Shor mode (best for small backends).")
        return build_mode_42_hive_Shor(bits, delta, config)

# -------------------------
# Build circuit selector (now includes Mode 1 and Mode 2)
# -------------------------
def build_circuit_selector(mode_id: int, bits: int, delta: Point, config, backend_qubits: int) -> QuantumCircuit:
    """Select and build the appropriate quantum circuit. Mode 2 is adaptive and typically run via run_mode_2_adaptive."""
    logger.info(f"Selecting mode {mode_id}...")
    meta = MODE_METADATA.get(mode_id, MODE_METADATA.get(99))
    if config.USE_FT and not meta.get("supports_ft", False):
        logger.warning(f"⚠️ Fault tolerance not supported for mode {mode_id}. Disabling FT.")
        config.USE_FT = False

    # New mode mappings:
    if mode_id == 0:
        return build_mode_0_hardware_probe(bits, delta, config)
    if mode_id == 1:
        # Mode 1: full iterative QPE using dynamic conditionals
        return build_mode_1_QPE_standard_qubit(bits, delta, config)
    if mode_id == 2:
        # Mode 2: adaptive no-dynamic approach — build and return a sample single-iteration circuit
        # (Note: run_mode_2_adaptive should be used to actually execute the adaptive sequence)
        logger.info("Mode 2 selected: returning a representative single-iteration circuit (use run_mode_2_adaptive to run the full adaptive workflow).")
        return build_mode_2_iteration(bits, delta, config, 0, [])  # iteration 0 circuit for inspection

    # Existing modes
    if mode_id == 29: return build_mode_29_QPE_omega(bits, delta, config)
    elif mode_id == 27: return build_mode_27_advanced_qpe(bits, delta, config)
    elif mode_id == 28: return build_mode_28_full_quantum_optimized(bits, delta, config)
    elif mode_id == 32: return build_mode_32_ft_draper_omega(bits, delta, config)
    elif mode_id == 34: return build_mode_34_modified_shor_omega(bits, delta, config)
    elif mode_id == 35: return build_mode_35_geo_QPE_omega(bits, delta, config)
    elif mode_id == 39: return build_mode_39_matrix_mod_omega(bits, delta, config)
    elif mode_id == 41: return build_mode_41_Shor(bits, delta, config)
    elif mode_id == 42: return build_mode_42_hive_Shor(bits, delta, config)
    elif mode_id == 43: return build_mode_43_hive_omega(bits, delta, config)
    elif mode_id == 44: return build_mode_44_matrix_unitary(bits, delta, config)
    elif mode_id == 99: return build_mode_99_ultimate_shor(bits, delta, config, backend_qubits)
    else:
        logger.warning(f"Mode {mode_id} not in top list, defaulting to Shor (Mode 41).")
        return build_mode_41_Shor(bits, delta, config)

# ==========================================
# 7. POST-PROCESSING 
# ==========================================

def process_measurement_result(measurement: int, bits: int, order: int, method: str) -> List[int]:
    """Process a single measurement result with phase estimation, A&B checks, and Hive stitching."""
    candidates = [measurement]
    logger.info(f"Processing measurement: {hex(measurement)} (method: {method})")

    # NEW: Hive Worker Stitching Logic (direct binary reconstruction)
    try:
        if method == "hive":
            # In Hive mode the measurement is a direct binary representation (LSB or MSB per mode metadata)
            candidate = measurement % order
            logger.info(f"  - Hive Reconstruction candidate: {hex(candidate)}")
            candidates.append(candidate)
            # Also try the bit-flip (in case noise flips sign / inverts)
            candidates.append((measurement ^ ((1 << bits) - 1)) % order)
            # return early for hive (optionally still fall through to other methods if desired)
            return candidates
    except Exception as e:
        logger.debug(f"Hive stitching failed: {e}")

    # Existing Phase estimation logic (unchanged)
    try:
        if method in ["phase", "auto"]:
            num, den = continued_fraction_approx(measurement, 2 ** bits, order)
            if den != 0:
                candidate = (num * modular_inverse_verbose(den, order)) % order
                logger.info(f"  - Phase estimation candidate: {hex(candidate)}")
                candidates.append(candidate)
    except Exception as e:
        logger.debug(f"Phase estimation failed: {e}")

    # Shor-style A&B logic (unchanged)
    try:
        if method == "shor":
            a = measurement >> (bits // 2)
            b = measurement & ((1 << (bits // 2)) - 1)
            logger.info(f"  - A register: {hex(a)}, B register: {hex(b)}")
            if b == (1 << (bits // 2)) - 1:
                logger.info("  - Detected b = -1 case, applying special handling")
                b = -1
            if b != 0:
                inv_b = modular_inverse_verbose(b, order)
                if inv_b:
                    candidate = (-a * inv_b) % order
                    logger.info(f"  - A&B GCD candidate: {hex(candidate)}")
                    candidates.append(candidate)
    except Exception as e:
        logger.debug(f"A&B processing failed: {e}")

    return candidates


class UniversalPostProcessor:
    """Post-processing with phase estimation, A&B checks, precomputation and verification."""

    def __init__(self, config):
        self.config = config
        self.search_depth = getattr(config, "SEARCH_DEPTH", 10000)
        self.found_keys: List[int] = []

    def save_key(self, k: int, mode_name: str = "", source: str = ""):
        """Save found key in multiple formats with metadata."""
        hex_k = hex(k)[2:].zfill(64)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recovered_keys_{timestamp}.txt"
        key_data = {
            "timestamp": timestamp,
            "mode": mode_name,
            "source": source,
            "formats": {
                "standard": "0x" + hex_k,
                "raw_hex": hex_k,
                "decimal": str(k)
            }
        }
        try:
            with open(filename, "a") as f:
                f.write(json.dumps(key_data) + "\n")
            logger.info(f"🔑 KEY SAVED TO: {filename}")
        except Exception as e:
            logger.warning(f"Could not save key to file: {e}")
        self.found_keys.append(k)

    def verify_candidate(self, k: int, target_x: int, target_y: int, order: int = ORDER) -> bool:
        """Verify a candidate key with point checks (full point match or X-only)."""
        try:
            Pt = ec_scalar_mult(k, G)
            if Pt is None:
                return False
            if target_x is not None and target_y is not None:
                if Pt.x() == target_x and Pt.y() == target_y:
                    logger.info(f"✅ Verified candidate {hex(k)}: full point match")
                    self.save_key(k, source="full_verify")
                    return True
            if Pt.x() == target_x:
                logger.info(f"✅ Verified candidate {hex(k)}: X coordinate matches")
                self.save_key(k, source="x_only_verify")
                return True
            Pt_inv = ec_point_negate(Pt)
            if Pt_inv and Pt_inv.x() == target_x:
                logger.info(f"✅ Verified candidate {hex(k)} via inverted point match")
                k_inv = (order - k) % order
                self.save_key(k_inv, source="inverted_verify")
                return True
        except Exception as e:
            logger.error(f"Verification failed for {hex(k)}: {e}")
        return False

    def process_all_measurements(self, counts: Dict[str, int], bits: int, order: int, start: int,
                                 target_x: int, target_y: int, mode_meta: Dict) -> Optional[int]:
        """Process all measurements with all post-processing steps and attempt verification."""
        method = mode_meta.get("method", "phase")
        endian = mode_meta.get("endian", "LSB")
        logger.info(f"📊 Processing top {min(self.search_depth, len(counts))} measurements...")
        candidates = []

        for state_str, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:self.search_depth]:
            try:
                clean_str = state_str.replace(" ", "")
                if endian == "LSB":
                    measurement = int(clean_str[::-1], 2) if clean_str else 0
                else:
                    measurement = int(clean_str, 2) if clean_str else 0
                logger.debug(f"  - Measurement {hex(measurement)} (counts: {count})")
                new_cands = process_measurement_result(measurement, bits, order, method)
                candidates.extend(new_cands)
            except Exception as e:
                logger.warning(f"  - Failed to process {state_str}: {e}")

        for k in set(candidates):
            if self.verify_candidate(k, target_x, target_y, order=order):
                return k

        logger.warning("❌ No valid key found after all post-processing steps")
        return None


def select_backend(config: Config):
    """Select appropriate backend using QiskitRuntimeService (V2)."""
    if QiskitRuntimeService is None:
        logger.warning("QiskitRuntimeService not available; falling back to AerSimulator.")
        return AerSimulator()

    service = QiskitRuntimeService(channel="ibm_cloud", token=config.TOKEN, instance=config.CRN)
    logger.info("🔍 Available IBM Quantum Backends:")
    try:
        backends = service.backends()
        for backend in backends:
            try:
                status = backend.status()
                queue_info = status.pending_jobs
                queue_length = queue_info if isinstance(queue_info, int) else len(queue_info)
                logger.info(f"  {backend.name} ({backend.num_qubits} qubits, {'🟢 Operational' if status.operational else '🔴 Offline'})")
                logger.info(f"     Queue: {queue_length} jobs")
            except Exception as e:
                logger.info(f"  {backend.name} (Status unavailable: {e})")
    except Exception as e:
        logger.warning(f"Could not list service.backends(): {e}")

    if config.BITS == 12:
        logger.info("💡 Using AerSimulator for 12-bit key (most efficient)")
        return AerSimulator()
    else:
        logger.info(f"💡 Selecting real backend for {config.BITS}-bit key...")
        try:
            backend = service.least_busy(simulator=False, operational=True, min_num_qubits=156)
            logger.info(f"  Selected: {backend.name} ({backend.num_qubits} qubits)")
            return backend
        except Exception as e:
            logger.warning(f"least_busy failed: {e}; falling back to AerSimulator")
            return AerSimulator()

def get_v2_options(self, manual_zne: bool = False) -> SamplerOptions:
    """
    Constructs SamplerOptions using the correct resilience attribute for the installed qiskit-ibm-runtime version.
    """
    logger.info("🔧 Building SamplerOptions (V2 Modern)...")

    zne_method = "manual" if manual_zne else getattr(self.config, "ZNE_METHOD", "manual")
    logger.info(f"📋 ZNE method: {zne_method}")

    opts = SamplerOptions()
    opts.default_shots = int(getattr(self.config, "SHOTS", 8192))

    # Dynamical Decoupling
    if getattr(self.config, "USE_DD", False):
        dd_seq = getattr(self.config, "DD_SEQUENCE", "XY4")
        opts.dynamical_decoupling = DynamicalDecouplingOptions(enable=True, sequence_type=dd_seq)
        logger.info(f"🛡️ Dynamical Decoupling enabled: {dd_seq}")

    # Resilience (ZNE)
    try:
        # Try to use ResilienceOptionsV2 (qiskit-ibm-runtime ≥ 0.24)
        resilience = ResilienceOptionsV2()
        if zne_method == "standard":
            resilience.zne_mitigation = True
            resilience.measure_mitigation = True
            logger.info("🔄 Standard ZNE enabled via ResilienceOptionsV2")
        else:
            resilience.zne_mitigation = False
            resilience.measure_mitigation = False
            logger.info("🔄 Manual ZNE: auto-ZNE disabled via ResilienceOptionsV2")
        opts.resilience = resilience
    except AttributeError:
        # Fallback to resilience_level (qiskit-ibm-runtime < 0.24)
        if zne_method == "standard":
            opts.resilience_level = 2
            logger.info("🔄 Standard ZNE enabled via resilience_level=2")
        else:
            opts.resilience_level = 0
            logger.info("🔄 Manual ZNE: auto-ZNE disabled via resilience_level=0")

    return opts

# ==========================================
# 11. ERROR MITIGATION ENGINE (V2-only)
# ==========================================
def build_sampler(self, backend: Any, manual_zne: bool = False) -> Tuple[Optional[Any], Any]:
    """Creates sampler with proper backend validation."""
    logger.info("🏗️ Building Sampler...")

    if not self.is_valid_backend(backend):
        logger.error("❌ INVALID BACKEND - must be a valid backend!")
        return None, self.get_v2_options(manual_zne)

    opts = self.get_v2_options(manual_zne=manual_zne)
    if Sampler is None:
        logger.error("❌ SamplerV2 primitive not available!")
        return None, opts

    try:
        # Try different constructor signatures
        try:
            sampler = Sampler(backend=backend, options=opts)
        except TypeError:
            try:
                sampler = Sampler(options=opts, backend=backend)
            except TypeError:
                sampler = Sampler(backend)
                # Try to set options after construction if possible
                if hasattr(sampler, 'set_options'):
                    sampler.set_options(opts)

        logger.info("✅ Sampler constructed successfully")
        return sampler, opts
    except Exception as e:
        logger.error(f"❌ Sampler construction failed: {str(e)}")
        return None, opts


def build_sampler_v2_with_options(backend: Any, config: Config, manual_zne: bool = True):
    """
    Construct and return a SamplerV2 instance with SamplerOptions passed to the constructor.
    Returns the constructed sampler and the options object.
    """
    opts = build_options_from_config(config)

    try:
        sampler = Sampler(options=opts)
        sampler.set_backend(backend)
    except Exception as e:
        logger.error(f"Sampler construction failed: {str(e)}")
        return None, opts

    logger.info("Created SamplerV2 with SamplerOptions in constructor")
    return sampler, opts


# ==========================================
# 11. ERROR MITIGATION ENGINE (V2 ROBUST FIX)
# ==========================================
def build_options_from_config(config: Config) -> SamplerOptions:
    """Construct a SamplerOptions (V2) object from Config and return it."""
    opts = SamplerOptions()

    try:
        if hasattr(config, "SHOTS") and config.SHOTS is not None:
            opts.default_shots = int(getattr(config, "SHOTS"))
    except Exception:
        logger.debug("Could not set default_shots on SamplerOptions")

    try:
        if getattr(config, "USE_DD", False):
            dd_seq = getattr(config, "DD_SEQUENCE", "XY4")
            opts.dynamical_decoupling = DynamicalDecouplingOptions(enable=True, sequence_type=dd_seq)
    except Exception:
        pass

    try:
        if getattr(config, "USE_ZNE", False):
            zne_method = getattr(config, "ZNE_METHOD", "manual")
            execution_options = SamplerExecutionOptionsV2()
            resilience_options = ResilienceOptionsV2()

            if zne_method == "standard":
                resilience_options.zne = ZneOptions()
            else:
                resilience_options.zne = None

            execution_options.resilience = resilience_options
            opts.execution = execution_options
    except Exception:
        pass

    return opts

# ==========================================
# 10. SAFE COUNTS EXTRACTION (UNCHANGED)
# ==========================================
def safe_get_counts(result_item) -> Optional[Dict[str, int]]:
    """Robust extraction of counts from runtime primitive results (V2-friendly)."""
    if result_item is None:
        return None
    try:
        try:
            first = result_item[0]
            if hasattr(first, "data") and hasattr(first.data, "meas"):
                try:
                    return dict(first.data.meas.get_counts())
                except Exception:
                    pass
            if hasattr(first, "get_counts"):
                return dict(first.get_counts())
        except Exception:
            pass

        if hasattr(result_item, "get_counts"):
            try:
                return dict(result_item.get_counts())
            except Exception:
                pass
        if hasattr(result_item, "data"):
            data = result_item.data
            for candidate in ("meas", "counts", "meas_counts", "measurement"):
                if hasattr(data, candidate):
                    try:
                        attr = getattr(data, candidate)
                        if hasattr(attr, "get_counts"):
                            return dict(attr.get_counts())
                        else:
                            return dict(attr)
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"safe_get_counts encountered error: {e}")
    return None

# ==========================================
# 11. ERROR MITIGATION ENGINE (V2 — FINAL FIX)
# ==========================================

class ErrorMitigationEngine:
    """
    🛡️ ERROR MITIGATION ENGINE (FINAL VERSION)
    - Handles ZNE, dynamical decoupling, and transpilation
    - Supports custom coupling maps and basis gates
    - Validates circuits for IBM compatibility
    - Robust error handling and fallback mechanisms
    - Comprehensive version compatibility for qiskit-ibm-runtime
    """
    def __init__(self, config: Any):
        logger.info("🛠️ Initializing Error Mitigation Engine...")
        self.config = config
        self.zne_scales = list(getattr(config, "ZNE_SCALES", [1, 3, 5]))
        self.use_gate_folding = getattr(config, "ZNE_USE_FOLDING", False)
        self.fold_method = getattr(config, "ZNE_FOLD_METHOD", "global")
        self.use_pub_tuple = getattr(config, "USE_PUB_TUPLE", True)
        logger.info(f"⚙️ Configuration loaded (ZNE scales: {self.zne_scales})")

    def _insert_id_padding_with_barrier(self, circ: QuantumCircuit, repeats: int) -> QuantumCircuit:
        """Insert identity gates with barriers for ZNE scaling."""
        padded = circ.copy()
        for _ in range(repeats):
            padded.barrier()
            for q in padded.qubits:
                padded.id(q)
            padded.barrier()
        return padded

    def _get_transpile_kwargs(self, backend: Any) -> dict:
        """Builds transpile kwargs with override support and IBM compatibility."""
        kwargs = {
            "optimization_level": getattr(self.config, "OPT_LEVEL", 3),
            "routing_method": 'sabre' if getattr(self.config, "USE_SABRE", True) else None,
            "scheduling_method": 'alap' if getattr(self.config, "USE_ALAP", True) else None
        }

        # Apply coupling map override if enabled
        if getattr(self.config, "OVERRIDE_COUPLING_MAP", False) and getattr(self.config, "CUSTOM_COUPLING_MAP", None):
            kwargs["coupling_map"] = self.config.CUSTOM_COUPLING_MAP
            logger.warning("⚠️ CUSTOM COUPLING MAP ENABLED - this may affect performance!")

        # Apply basis gates override if enabled and IBM-compatible
        if getattr(self.config, "OVERRIDE_BASIS_GATES", False) and getattr(self.config, "CUSTOM_BASIS_GATES", None):
            custom_gates = self.config.CUSTOM_BASIS_GATES
            ibm_compatible = all(gate in ["rz", "sx", "x", "cx", "id", "ecr"] for gate in custom_gates)

            if ibm_compatible:
                kwargs["basis_gates"] = custom_gates
                logger.info(f"🔧 Applied custom basis gates: {custom_gates}")
            else:
                logger.warning("⚠️ Custom basis gates contain non-IBM-compatible gates!")
                logger.warning("⚠️ Using backend.target basis gates instead for compatibility")
                logger.warning(f"⚠️ Requested gates: {custom_gates}")

        return kwargs

    def get_v2_options(self, manual_zne: bool = False) -> Any:
        logger.info("🔧 Building SamplerOptions...")
        # Determine ZNE method
        zne_method = "manual" if manual_zne else getattr(self.config, "ZNE_METHOD", "manual")
        logger.info(f"📋 Configuring for ZNE method: {zne_method}")

        # Fallback mode if SamplerOptions is not available
        if SamplerOptions is None:
            logger.warning("⚠️ SamplerOptions class not available - using dict fallback")
            opts = {}

            # Dynamical Decoupling configuration
            if getattr(self.config, "USE_DD", False):
                opts["dynamical_decoupling"] = {
                    "enable": True,
                    "sequence_type": getattr(self.config, "DD_SEQUENCE", "XY4")
                }
                logger.info("🛡️ Dynamical Decoupling enabled (dict mode)")

            # Resilience configuration for dict mode
            try:
                if zne_method == "standard":
                    opts["resilience_level"] = 2
                    logger.info("🔄 Standard ZNE configured (resilience level: 2)")
                else:
                    opts["resilience_level"] = 0
                    logger.info("🔄 Manual ZNE configured (resilience level: 0)")
            except Exception as e:
                logger.warning(f"resilience setting in dict mode: {str(e)}")
                # Try alternative resilience configuration
                opts["resilience"] = {"method": zne_method}
                logger.info(f"🔄 Fallback ZNE configuration: method={zne_method}")

            # Set default shots
            opts["default_shots"] = int(getattr(self.config, "SHOTS", 8192))
            logger.info(f"🎯 Default shots set to: {opts['default_shots']}")

            return opts

        # Create SamplerOptions instance
        options = SamplerOptions()

        # 1. Configure Dynamical Decoupling
        if getattr(self.config, "USE_DD", False):
            try:
                # Try direct attribute access first
                if hasattr(options, "dynamical_decoupling"):
                    options.dynamical_decoupling.enable = True
                    options.dynamical_decoupling.sequence_type = getattr(self.config, "DD_SEQUENCE", "XY4")
                    logger.info(f"🛡️ Dynamical Decoupling enabled (direct): {getattr(self.config, 'DD_SEQUENCE', 'XY4')}")
                else:
                    # Create the attribute if it doesn't exist
                    dd_class = type("DynamicalDecoupling", (), {
                        "enable": True,
                        "sequence_type": getattr(self.config, "DD_SEQUENCE", "XY4")
                    })
                    setattr(options, "dynamical_decoupling", dd_class)
                    logger.info(f"🛡️ Dynamical Decoupling enabled (created): {getattr(self.config, 'DD_SEQUENCE', 'XY4')}")
            except Exception as e:
                logger.warning(f"Failed to configure Dynamical Decoupling: {str(e)}")
                # Fallback to dict style
                options.dynamical_decoupling = {
                    "enable": True,
                    "sequence_type": "XY4"
                }
                logger.info("🛡️ Dynamical Decoupling enabled (fallback dict)")

        # 2. Configure Resilience/ZNE settings with comprehensive fallbacks
        try:
            # First try the newest API (resilience.level)
            try:
                if not hasattr(options, "resilience"):
                    resilience_class = type("Resilience", (), {})
                    setattr(options, "resilience", resilience_class)

                if zne_method == "standard":
                    options.resilience.level = 2
                    logger.info("🔄 Standard ZNE configured (resilience.level=2)")
                else:
                    options.resilience.level = 0
                    logger.info("🔄 Manual ZNE configured (resilience.level=0)")
            except Exception as e1:
                logger.debug(f"New resilience API failed: {str(e1)}")

                # Fallback to resilience_level attribute
                try:
                    if zne_method == "standard":
                        options.resilience_level = 2
                        logger.info("🔄 Standard ZNE configured (resilience_level=2)")
                    else:
                        options.resilience_level = 0
                        logger.info("🔄 Manual ZNE configured (resilience_level=0)")
                except Exception as e2:
                    logger.debug(f"resilience_level failed: {str(e2)}")

                    # Final fallback to execution.resilience
                    try:
                        if not hasattr(options, "execution"):
                            setattr(options, "execution", {})

                        if isinstance(options.execution, dict):
                            options.execution["resilience"] = {"method": zne_method}
                        else:
                            setattr(options.execution, "resilience", {"method": zne_method})

                        logger.info(f"🔄 ZNE configured via execution.resilience (method={zne_method})")
                    except Exception as e3:
                        logger.info(f"qiskit-ibm-runtime --upgrade 0.44.0 No need to identify Resilience Use pip install qiskit==0.24.0 instead Avoiding {str(e3)}")
                        logger.info(f"🔄 ZNE method {zne_method} selected ")

        except Exception as e:
            logger.error(f"Critical error configuring ZNE: {str(e)}")

        # 3. Configure default shots
        try:
            options.default_shots = int(getattr(self.config, "SHOTS", 8192))
            logger.info(f"🎯 Default shots set to: {options.default_shots}")
        except Exception as e:
            logger.warning(f"Failed to set default shots: {str(e)}")
            try:
                # Try alternative way to set shots
                if hasattr(options, "shots"):
                    options.shots = int(getattr(self.config, "SHOTS", 8192))
                    logger.info(f"🎯 Default shots set to: {options.shots} (alternative)")
            except Exception:
                logger.error("Could not set shots in any way")

        # 4. Additional resilience configuration for newer APIs
        try:
            # Check if we're using a newer API that supports resilience options
            if hasattr(options, "set_resilience"):
                if zne_method == "standard":
                    options.set_resilience(level=2)
                else:
                    options.set_resilience(level=0)
                logger.info(f"🔄 ZNE configured using set_resilience (level={'2' if zne_method == 'standard' else '0'})")
        except Exception as e:
            logger.debug(f"set_resilience method not available: {str(e)}")

        # 5. Final verification of configuration
        try:
            logger.info("✅ SamplerOptions configuration summary:")
            logger.info(f"   - ZNE Method: {zne_method}")
            logger.info(f"   - Dynamical Decoupling: {'Enabled' if getattr(self.config, 'USE_DD', False) else 'Disabled'}")

            # Try to report the actual resilience configuration
            resilience_config = "Unknown"
            try:
                if hasattr(options, "resilience") and hasattr(options.resilience, "level"):
                    resilience_config = f"resilience.level={options.resilience.level}"
                elif hasattr(options, "resilience_level"):
                    resilience_config = f"resilience_level={options.resilience_level}"
                elif hasattr(options, "execution") and hasattr(options.execution, "resilience"):
                    resilience_config = f"execution.resilience={options.execution.resilience}"
            except Exception:
                pass

            logger.info(f"   - Resilience Config: {resilience_config}")
            logger.info(f"   - Default Shots: {getattr(options, 'default_shots', 'Not set')}")
        except Exception:
            logger.warning("Could not generate configuration summary")

        return options

    def build_sampler(self, backend: Any, manual_zne: bool = False) -> Tuple[Optional[Any], Any]:
        """Creates sampler with proper backend validation."""
        logger.info("🏗️ Building Sampler...")

        # Validate backend - check for V2 attributes
        if not hasattr(backend, 'target') and not hasattr(backend, 'configuration'):
            logger.error("❌ INVALID BACKEND - must be a valid backend with target or configuration!")
            return None, self.get_v2_options(manual_zne)

        opts = self.get_v2_options(manual_zne=manual_zne)
        if Sampler is None:
            logger.error("❌ SamplerV2 primitive not available!")
            return None, opts

        try:
            # Try different constructor signatures
            try:
                sampler = Sampler(backend=backend, options=opts)
            except TypeError:
                try:
                    sampler = Sampler(options=opts, backend=backend)
                except TypeError:
                    sampler = Sampler(backend)
                    # Try to set options after construction if possible
                    if hasattr(sampler, 'set_options'):
                        sampler.set_options(opts)

            logger.info("✅ Sampler constructed successfully")
            return sampler, opts
        except Exception as e:
            logger.error(f"❌ Sampler construction failed: {str(e)}")
            return None, opts

    def _validate_circuit(self, qc: QuantumCircuit) -> bool:
        """Validates circuit for IBM compatibility."""
        illegal_gates = ["h", "u3", "cp"]
        for instr in qc.data:
            if instr.operation.name in illegal_gates:
                logger.error(f"❌ ILLEGAL GATE DETECTED: {instr.operation.name}")
                return False
        logger.info("✅ Circuit validation passed (IBM-compatible)")
        return True

    def manual_zne(self, qc: QuantumCircuit, backend: Any, shots: int) -> Dict[str, float]:
        """🧪 MANUAL ZNE IMPLEMENTATION (V2)
        - Locks hardware mapping once
        - Scales noise after mapping
        - Submits all scales in one job
        - Robust error handling and count extraction
        """
        logger.info("\n" + "="*60)
        logger.info("🧪 STARTING MANUAL ZNE PROCESS")
        logger.info(f"📏 Scales: {self.zne_scales}")
        logger.info(f"🎯 Shots: {shots}")
        logger.info("="*60)

        # Step 1: Transpile base circuit
        logger.info("🔧 Transpiling base circuit...")
        transpile_kwargs = self._get_transpile_kwargs(backend)

        try:
            base_tp = transpile(qc, backend, **transpile_kwargs)

            # Validate transpilation for IBM compatibility
            illegal_gates = ["h", "u3", "cp"]
            illegal_found = any(instr.operation.name in illegal_gates for instr in base_tp.data)

            if illegal_found:
                logger.error("❌ Illegal gates detected after transpilation!")
                logger.info("🔧 Attempting emergency decomposition with IBM-compatible basis...")

                # Try to fix with IBM-compatible basis gates
                base_tp = transpile(
                    base_tp,
                    basis_gates=["rz", "sx", "x", "cx", "id"],
                    optimization_level=0
                )

                # Verify the fix worked
                if any(instr.operation.name in illegal_gates for instr in base_tp.data):
                    raise RuntimeError("Critical: Could not decompose illegal gates!")

            logger.info("✅ Base circuit transpiled and validated")
        except Exception as e:
            logger.error(f"❌ Transpilation failed: {str(e)}")
            return defaultdict(float)

        # Step 2: Create scaled circuits
        logger.info("📦 Preparing scaled circuits...")
        scaled_circuits = []
        limit_depth = getattr(self.config, "MAX_ZNE_DEPTH", 1000000)

        for scale in self.zne_scales:
            logger.info(f"  🔹 Preparing scale {scale}...")

            if scale == 1:
                scaled = base_tp.copy()
            else:
                scaled = self._insert_id_padding_with_barrier(base_tp, scale-1)

            if scaled.depth() > limit_depth:
                logger.warning(f"⚠️ Scale {scale} exceeds depth limit ({limit_depth}) - skipping!")
                continue

            scaled_circuits.append(scaled)
            logger.info(f"  ✅ Scale {scale} prepared (depth: {scaled.depth()})")

        if not scaled_circuits:
            logger.error("❌ No valid scaled circuits prepared!")
            return defaultdict(float)

        # Step 3: Submit batch job
        logger.info("📤 Submitting ZNE batch job...")
        sampler, _ = self.build_sampler(backend, manual_zne=True)
        if sampler is None:
            logger.error("❌ No sampler available for ZNE!")
            return defaultdict(float)

        try:
            pub_list = [(c,) for c in scaled_circuits]
            job = sampler.run(pub_list, shots=shots)
            job_id = getattr(job, 'job_id', 'unknown')()
            logger.info(f"📡 ZNE Job submitted successfully (ID: {job_id})")
            logger.info("⏳ Waiting for results...")
            result = job.result()
            logger.info("✅ Job completed successfully")
        except Exception as e:
            logger.error(f"❌ ZNE job failed: {str(e)}")
            return defaultdict(float)

        # Step 4: Extract counts with robust search logic
        logger.info("📊 Extracting counts from results...")
        counts_list = []
        for i, scale in enumerate(self.zne_scales):
            if i >= len(scaled_circuits): continue

            try:
                # Try multiple ways to extract counts
                pub_result = result[i]

                # Method 1: Direct get_counts
                if hasattr(pub_result, 'get_counts'):
                    counts = pub_result.get_counts()
                # Method 2: Check data.meas
                elif hasattr(pub_result, 'data') and hasattr(pub_result.data, 'meas'):
                    counts = pub_result.data.meas.get_counts()
                # Method 3: Check all data attributes
                else:
                    counts = {}
                    for attr in dir(pub_result):
                        if attr.startswith('meas') or attr.startswith('count'):
                            try:
                                obj = getattr(pub_result, attr)
                                if hasattr(obj, 'get_counts'):
                                    counts = obj.get_counts()
                                    break
                            except Exception:
                                continue

                if not counts:
                    counts = {}

                counts_list.append(counts)
                logger.info(f"  📋 Scale {scale}: {len(counts)} counts retrieved")
            except Exception as e:
                logger.error(f"  ❌ Failed to extract counts for scale {scale}: {str(e)}")
                counts_list.append({})

        if not any(counts_list):
            logger.warning("⚠️ No valid counts retrieved from any scale!")
            return defaultdict(float)

        # Step 5: Extrapolate results
        logger.info("📈 Extrapolating results...")
        extrapolated = defaultdict(float)
        all_keys = set().union(*counts_list)

        for key in all_keys:
            vals = [c.get(key, 0) for c in counts_list]
            if len(vals) > 1:
                try:
                    fit = np.polyfit(self.zne_scales[:len(vals)], vals, 1)
                    extrapolated[key] = max(0.0, float(fit[1]))
                except Exception:
                    extrapolated[key] = float(vals[-1])
            else:
                extrapolated[key] = float(vals[0])

        logger.info(f"✅ ZNE Dragon Complete: {len(extrapolated)} bitstrings extrapolated")
        logger.info("="*60)
        return extrapolated

# ==========================================
# 12. ADAPTIVE Shor RUNNER (Mode 41) — uses EME.build_sampler (V2)
# ==========================================
def run_Shor_adaptive(bits: int, delta, config, backend, shots: int = 1024) -> Dict[str, int]:
    """
    👑 Shor ADAPTIVE RUNNER (V2)
    - Implements adaptive bit measurement
    - Supports manual ZNE and runtime sampling
    - Handles IBM backend validation
    """
    logger.info("\n" + "="*60)
    logger.info("👑 STARTING ADAPTIVE Shor MODE")
    logger.info(f"📏 Target: {bits} bits | 🎯 Shots: {shots}")
    logger.info("="*60)

    # Validate and prepare backend
    if not isinstance(backend, BackendV2) and not hasattr(backend, "target"):
        logger.error("❌ INVALID BACKEND DETECTED!")
        logger.info("🔄 Falling back to AerSimulator")
        backend = AerSimulator()

    eme = ErrorMitigationEngine(config)
    sampler, _ = eme.build_sampler(backend, manual_zne=(config.ZNE_METHOD == "manual"))
    powers = precompute_powers(delta, bits)
    measured_bits = []

    logger.info(f"📊 Beginning adaptive measurement (0/{bits} bits completed)")

    for k in range(bits):
        logger.info(f"\n🔹 Iteration {k+1}/{bits} started...")
        logger.info(f"📝 Building circuit for bit {k} with previous bits: {measured_bits}")

        # Build circuit
        ctrl = QuantumRegister(1, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(1, "meas_k")
        regs = [ctrl, state, creg]

        if getattr(config, "USE_FT", False):
            ft_anc = QuantumRegister(2, "ft_anc")
            regs.append(ft_anc)

        qc = QuantumCircuit(*regs)

        # Circuit construction
        qc.h(ctrl[0])
        qc.x(state[0])

        if getattr(config, "USE_FT", False):
            prepare_verified_ancilla(qc, regs[-1][0])
            prepare_verified_ancilla(qc, regs[-1][1])
            encode_repetition(qc, ctrl[0], regs[-1])

        total_angle = 0.0
        for m, bit in enumerate(measured_bits):
            if bit:
                total_angle += -math.pi / (2 ** (k - m))
        if total_angle != 0:
            qc.p(total_angle, ctrl[0])

        dx, dy = powers[k].x(), powers[k].y()
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)

        if getattr(config, "USE_FT", False):
            decode_repetition(qc, regs[-1], ctrl[0])

        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[0])

        # Transpile with proper settings
        try:
            logger.info("🔧 Transpiling circuit with IBM-compatible settings...")

            transpile_kwargs = eme._get_transpile_kwargs(backend)
            transpiled = transpile(qc, backend, **transpile_kwargs)

            # Validate transpilation
            if any(instr.operation.name == "h" for instr in transpiled.data):
                logger.error("❌ Hadamard gates detected after transpilation!")
                logger.info("🔧 Attempting emergency decomposition...")
                transpiled = transpile(transpiled, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
                if any(instr.operation.name == "h" for instr in transpiled.data):
                    raise RuntimeError("Critical: Could not decompose Hadamard gates!")

            logger.info("✅ Transpilation successful")

        except Exception as e:
            logger.error(f"❌ Transpilation failed: {str(e)}")
            logger.info("🔄 Using original circuit (no transpilation)")
            transpiled = qc

        # Execute circuit
        bit_result = 0
        try:
            if getattr(config, "USE_ZNE", False) and getattr(config, "ZNE_METHOD", "manual") == "manual":
                logger.info("🧪 Running MANUAL ZNE for this iteration...")
                counts_map = eme.manual_zne(transpiled, backend, shots)
                if counts_map:
                    best_key = max(counts_map.items(), key=lambda x: x[1])[0]
                    bit_result = int(best_key.replace(" ", "")[-1])
                    logger.info(f"📊 ZNE Result: {bit_result}")
                else:
                    logger.warning("⚠️ ZNE produced no counts! Defaulting to 0")
                    bit_result = 0
            else:
                if sampler is not None:
                    logger.info("🚀 Submitting to runtime sampler...")
                    job = sampler.run([(transpiled,)], shots=shots)
                    res = job.result()
                    counts = safe_get_counts(res) or {}

                    if counts:
                        best = max(counts.items(), key=lambda x: x[1])[0]
                        bit_result = int(best.replace(" ", "")[-1])
                        logger.info(f"📊 Runtime Result: {bit_result}")
                    else:
                        logger.warning("⚠️ No counts from runtime sampler! Defaulting to 0")
                        bit_result = 0
                else:
                    logger.error("❌ No sampler available!")
                    bit_result = 0

        except Exception as e:
            logger.error(f"❌ Execution failed: {str(e)}")
            logger.info("🔄 Defaulting bit result to 0")
            bit_result = 0

        measured_bits.append(bit_result)
        logger.info(f"💾 Measured bit {k} = {bit_result}")
        logger.info(f"📊 Progress: {len(measured_bits)}/{bits} bits completed")

    bitstr_lsb = "".join(str(b) for b in measured_bits)
    logger.info("\n" + "="*60)
    logger.info("🎉 ADAPTIVE Shor COMPLETE!")
    logger.info(f"📜 Final Result: {bitstr_lsb}")
    logger.info(f"🎯 Total Shots: {shots}")
    logger.info("="*60)

    return {bitstr_lsb: shots}

# ===== plot_visuals (unchanged) =====
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

        # Q-sphere
        plt.subplot(2, 2, 3)
        plot_state_qsphere(counts, title="Q-Sphere Visualization")

        # Bloch multivector
        plt.subplot(2, 2, 4)
        plot_bloch_multivector(counts, title="Bloch Multivector")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        plt.figure(figsize=(8, 6))
        plot_histogram(counts, title="Measurement Results")
        plt.show()

# ==========================================
# 13. MAIN EXECUTION (uses adaptive Shor when appropriate)
# ==========================================
def run_dragon_code(config: Config = None):
    """
    🐉 DRAGON CODE MAIN EXECUTION (FINAL VERSION With Multiple Quantum Attacks)
    - Handles backend selection with validation
    - Runs adaptive or full circuits with proper transpilation
    - Includes BOTH coupling_map AND basis_gates overrides with warnings
    - Maintains IBM compatibility with emergency fallbacks
    - Robust error handling and detailed logging
    """
    logger.info("\n" + "="*70)
    logger.info("🐉 DRAGON CODE v135 - QUANTUM ECDLP SOLVER 🔥 ")
    logger.info("="*70)

    # Initialize configuration
    if config is None:
        config = Config()
    config.user_menu()

    # Get IBM Quantum credentials if not set
    if not getattr(config, "TOKEN", None):
        config.TOKEN = input("🔑 Enter your IBM Quantum API token: ").strip()
    if not getattr(config, "CRN", None):
        config.CRN = input("🏢 Enter your IBM Quantum CRN Instance_Example_Full'' crn:v1:bluemix:public:quantum-computing:us-east:a/.....:: '' ( if your Subscription was free than type 'free'): ").strip() or "free"

    # Connect to IBM Quantum services
    logger.info("🔌 Connecting to IBM Quantum services...")
    service = None
    try:
        service = QiskitRuntimeService.save_account(channel="ibm_cloud", token=config.TOKEN, overwrite=True, instance=config.CRN)
        logger.info("✅ Connected successfully to IBM Quantum")
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        logger.info("🔄 Continuing with local simulator")

    # Select and validate backend
    logger.info("🔍 Selecting backend...")
    backend = select_backend(config)

    # Validate backend object
    if not hasattr(backend, 'target'):
        logger.error("❌ INVALID BACKEND DETECTED!")
        logger.info("🔄 Falling back to AerSimulator")
        backend = AerSimulator()
    else:
        logger.info(f"✅ Backend selected: {backend.name} ({getattr(backend, 'num_qubits', 'N/A')} qubits)")

    # Decompress public key
    logger.info("🔐 Decompressing public key...")
    try:
        Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
        delta = ec_point_sub(Q, ec_scalar_mult(config.KEYSPACE_START, G))
        logger.info("✅ Public key decompressed successfully")
    except Exception as e:
        logger.error(f"❌ Public key decompression failed: {str(e)}")
        return

    # Determine execution mode
    mode_id = 41 if config.METHOD == "smart" else int(config.METHOD)
    logger.info(f"🎯 Selected mode: {mode_id} ({MODE_METADATA.get(mode_id, {}).get('name', 'unknown')})")

    eme = ErrorMitigationEngine(config)

    # Adaptive Shor for large bit sizes
    if mode_id == 41 and config.BITS > 24:
        logger.info(f"👑 Using ADAPTIVE Shor for {config.BITS} bits")
        try:
            counts = run_Shor_adaptive(config.BITS, delta, config, backend, shots=config.SHOTS)

            candidate = UniversalPostProcessor(config).process_all_measurements(
                counts, config.BITS, ORDER, config.KEYSPACE_START, Q.x(), Q.y(), MODE_METADATA[mode_id]
            )

            if candidate:
                logger.info(f"🎉 SUCCESS: Found candidate {hex(candidate)}")
                Pt = ec_scalar_mult(candidate, G)
                logger.info(f"🔍 Verification: {hex(Pt.x())} == {hex(Q.x())}")
            else:
                logger.warning("⚠️ No valid candidate found after adaptive Shor")

            plot_visuals(counts, config.BITS)
        except Exception as e:
            logger.error(f"❌ Adaptive Shor failed: {str(e)}")
            logger.error("🔄 Attempting fallback to full circuit execution...")
        else:
            return  # Success - exit after adaptive Shor

    # Full circuit execution for smaller bit sizes
    logger.info("🏗️ Building full circuit...")
    try:
        backend_qubits = getattr(backend, "num_qubits", None)
        qc = build_circuit_selector(mode_id, config.BITS, delta, config, backend_qubits)
        logger.info(f"✅ Circuit built with {qc.num_qubits} qubits")
    except Exception as e:
        logger.error(f"❌ Circuit building failed: {str(e)}")
        return

    # Transpile with BOTH override supports
    try:
        logger.info("🔧 Transpiling circuit with proper settings...")

        # Base transpile kwargs
        transpile_kwargs = {
            "optimization_level": config.OPT_LEVEL,
            "routing_method": 'sabre' if config.USE_SABRE else None,
            "scheduling_method": 'alap' if config.USE_ALAP else None
        }

        # Check and apply overrides if enabled
        if config.OVERRIDE_COUPLING_MAP or config.OVERRIDE_BASIS_GATES:
            logger.warning("⚠️ Transpile override enabled: using custom coupling_map/basis_gates")
            logger.warning("⚠️ This overrides backend.target and may invalidate durations/error rates")

            if config.OVERRIDE_COUPLING_MAP and config.CUSTOM_COUPLING_MAP is not None:
                transpile_kwargs["coupling_map"] = config.CUSTOM_COUPLING_MAP
                logger.info(f"🔗 Applied custom coupling map with {len(config.CUSTOM_COUPLING_MAP)} connections")

            if config.OVERRIDE_BASIS_GATES and config.CUSTOM_BASIS_GATES is not None:
                # Apply basis gates override but with IBM compatibility check
                custom_gates = config.CUSTOM_BASIS_GATES
                ibm_compatible = all(gate in ["rz", "sx", "x", "cx", "id", "ecr"] for gate in custom_gates)

                if ibm_compatible:
                    transpile_kwargs["basis_gates"] = custom_gates
                    logger.info(f"🔧 Applied custom basis gates: {custom_gates}")
                else:
                    logger.warning("⚠️ Custom basis gates contain non-IBM-compatible gates!")
                    logger.warning("⚠️ Using backend.target basis gates instead for compatibility")
                    logger.warning(f"⚠️ Requested gates: {custom_gates}")

        # Perform transpilation
        transpiled = transpile(qc, backend, **transpile_kwargs)

        # Validate transpilation - check for illegal gates
        illegal_gates = ["h", "u3", "cp"]
        illegal_found = any(instr.operation.name in illegal_gates for instr in transpiled.data)

        if illegal_found:
            logger.error("❌ Illegal gates detected after transpilation!")
            logger.info("🔧 Attempting emergency decomposition with IBM-compatible basis...")

            # Try to fix with IBM-compatible basis gates
            transpiled = transpile(
                transpiled,
                basis_gates=["rz", "sx", "x", "cx", "id"],
                optimization_level=0
            )

            # Verify the fix worked
            if any(instr.operation.name in illegal_gates for instr in transpiled.data):
                logger.error("❌ Could not decompose illegal gates!")
                logger.info("🔄 Falling back to original circuit")
                transpiled = qc
            else:
                logger.info("✅ Emergency decomposition successful")

        logger.info(f"✅ Transpilation successful (depth: {transpiled.depth()})")

    except Exception as e:
        logger.error(f"❌ Transpilation failed: {str(e)}")
        logger.info("🔄 Using original circuit (no transpilation)")
        transpiled = qc

    # Execute circuit with proper error handling
    try:
        if getattr(config, "USE_ZNE", False) and getattr(config, "ZNE_METHOD", "manual") == "manual":
            logger.info("🧪 Running MANUAL ZNE on full circuit...")
            counts = eme.manual_zne(transpiled, backend, config.SHOTS)
        else:
            logger.info("🚀 Submitting circuit for execution...")
            sampler, _ = eme.build_sampler(backend, manual_zne=False)

            if sampler is not None:
                job = sampler.run([(transpiled,)], shots=config.SHOTS)
            else:
                job = backend.run(transpiled, shots=config.SHOTS)

            job_id = getattr(job, 'job_id', 'unknown')()
            logger.info(f"📡 Job submitted (ID: {job_id})")
            logger.info("⏳ Waiting for results...")
            res = job.result()
            counts = safe_get_counts(res) or {}

            # Additional count extraction attempts if primary method fails
            if not counts:
                try:
                    counts = dict(res[0].data.meas.get_counts())
                except Exception:
                    try:
                        counts = dict(res[0].get_counts())
                    except Exception:
                        counts = {}

        # Process and display results
        logger.info("📊 Top measurement results:")
        for i, (s, c) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            logger.info(f"  {i+1}. {s}: {c}")

        # Post-processing
        candidate = UniversalPostProcessor(config).process_all_measurements(
            counts, config.BITS, ORDER, config.KEYSPACE_START, Q.x(), Q.y(), MODE_METADATA[mode_id]
        )

        if candidate:
            logger.info(f"🎉 SUCCESS: Found candidate {hex(candidate)}")
            Pt = ec_scalar_mult(candidate, G)
            logger.info(f"🔍 Verification: {hex(Pt.x())} == {hex(Q.x())}")
        else:
            logger.warning("⚠️ No valid candidate found")

        plot_visuals(counts, config.BITS)

    except Exception as e:
        logger.error(f"❌ Execution failed: {str(e)}")
        logger.info("🔄 Attempting to continue with partial results...")
        counts = {}  # Empty counts for visualization
        plot_visuals(counts, config.BITS)

    logger.info("\n" + "="*70)
    logger.info("🏁 DRAGON CODE EXECUTION COMPLETE 🏁 Donation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰")
    logger.info("="*70)


# If run as script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("""
    🐉 DRAGON_CODE v135 Multiple Quantum Attacks 🐉🔥
    ----------------------------
    Donation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰
    ----------------------------
    🚀 Starting Quantum ECDLP Solver...
    """)
    
    run_dragon_code()


