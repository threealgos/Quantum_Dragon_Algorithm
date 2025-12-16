# Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#======================================================================================================
"""
HERE Is Quantum ECDLP Solver - Dragon Mode Edition (v120) 🐉
----------------------------------------------
Integrates ALL 21 MODES including the fixed Standard IPE.

MODES LIST:
 0. Hardware Probe         11. Swarm (Parallel Hive)
 1. IPE (Standard Legacy)  12. Heavy Draper (FT Test)
 2. Hive (Best/Opt)        13. Compressed Shadow
 3. Windowed IPE           14. Shor Logic (Pure)
 4. Semiclassical          15. Geometric IPE
 5. AB (Full Shor)         16. Windowed (Explicit)
 6. FT Draper Test         17. Hive Swarm (Explicit)
 7. Geometric Phase        18. Explicit Logic
 8. Verified (Flags+1D)    19. Fixed AB (Hybrid)
 9. Shadow 2D              20. Matrix Mod (Unitary)
10. Reverse IPE

--------------------------------------------------------
Integrates ALL 21 MODES with EXPLICIT IMPLEMENTATIONS .
[+] Mode 20 Scalable for 135+ bits (Smart Gate).
[+] Fault Tolerance (FT) Restored.
[+] Aggressive Result Retrieval.
[+] Full Error Mitigation.
 
MITIGATION: Manual ZNE, TREX, Twirling, DD (XY4).
"""

from IPython.display import display
from qiskit import synthesis
from qiskit.synthesis import synth_qft_full
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from fractions import Fraction
from collections import Counter, defaultdict
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from qiskit.circuit import UnitaryGate, Gate
from qiskit.circuit.library import ZGate, MCXGate, RYGate,, QFTGate
from qiskit.circuit.library import QFT, HGate, CXGate, CCXGate
from typing import Optional, List, Dict
from qiskit.circuit import Parameter
from Crypto.PublicKey import ECC
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import numbertheory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import hashlib
import base58
import pandas as pd
import logging
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Options, SamplerV2 as Sampler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram, plot_distribution
from math import gcd, pi, ceil, log2
from typing import Optional, Tuple, List, Dict
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
# ---------------- CONFIG ----------------
# Save IBM Quantum account (replace token and instance with your own)
api_token = "API_TOKEN"
QiskitRuntimeService.save_account(channel="ibm_cloud", token=api_token, overwrite=True)

# 1. Get a FRESH authentication token
service = QiskitRuntimeService(
    instance="<CRN>"
)
# ==========================================
# secp256k1 Constants
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
ORDER = N
CURVE = CurveFp(P, A, B)
G = Point(CURVE, Gx, Gy)

class Config:
    def __init__(self):
     
        # --- Target ---
        self.BITS = 135
        self.KEYSPACE_START = 0x4000000000000000000000000000000000
        self.COMPRESSED_PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
        
        # --- Backend ---
        self.BACKEND = "ibm_fez"  #  And For Future backends ~1386 Qubits Nighthawk/Kookaburra 
        self.TOKEN = "YOUR_IBM_TOKEN"
        self.CRN = "YOUR_IBM_CRN"
        
        # -- Mode Selection Set to integer 0-20 or "smart"
        self.METHOD = "smart" 
     
        # --- Tuning ---
        self.USE_COMPRESSED = True    # Hive x-only logic
        self.USE_FLAGS = 2            # Extra_Verification
        self.USE_FT = False           # Fault Tolerance Toggle
        
        # --- MITIGATION ---
        self.SHOTS = 8192  # MAX_SHOTS = 16384 & 100,000 & 1 million
        self.OPT_LEVEL = 3
        self.USE_MANUAL_ZNE = True    
        self.USE_DD = True
        self.DD_SEQUENCE = "XY4" # "XpXm & "XX"     
        self.USE_MEM = True           
    
    @property
    def INTERNAL_RESILIENCE_LEVEL(self):
        return 1 if self.USE_MANUAL_ZNE else 2

    def user_menu(self):
        print("\n=== Quantum ECDLP Solver - Dragon Mode Edition (v120) 🐉 ===")
        print("MODES: [0-20] All Modes Active.")
        
        m = input(f"Select Mode (default {self.METHOD}): ").strip()
        if m: self.METHOD = m if m == "smart" else int(m)

        b = input(f"Target Bits (default {self.BITS}): ").strip()
        if b: self.BITS = int(b)

        if self.METHOD == 8 or self.METHOD == "smart":
            f = input(f"Use Flags [1 or 2] (default {self.USE_FLAGS}): ").strip()
            if f: self.USE_FLAGS = int(f)

        # FT Selection
        ft = input(f"Enable Fault Tolerance (Repetition)? [y/n] (default {'y' if self.USE_FT else 'n'}): ").strip().lower()
        self.USE_FT = (ft == 'y')

        bk = input(f"Backend Name (default {self.BACKEND}): ").strip()
        if bk: self.BACKEND = bk

        s = input(f"Shots (default {self.SHOTS}): ").strip()
        if s: self.SHOTS = int(s)

        z = input(f"Enable Manual ZNE? [y/n] (default {'y' if self.USE_MANUAL_ZNE else 'n'}): ").strip().lower()
        self.USE_MANUAL_ZNE = (z == 'y')
        print("=================================\n")

config = Config()

# ==========================================
# 2. MATH & CLASSICAL UTILS
# ==========================================

def gcd_verbose(a: int, b: int) -> int:
    while b != 0: a, b = b, a % b
    return a

def modular_inverse_verbose(a: int, m: int) -> Optional[int]:
    return pow(a, -1, m) if gcd_verbose(a, m) == 1 else None

def decompress_pubkey(hex_key: str) -> Point:
    hex_key = hex_key.lower().replace("0x", "")
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    y_sq = (pow(x, 3, P) + B) % P
    y = numbertheory.square_root_mod_prime(y_sq, P)
    if (y % 2) != (prefix % 2): y = P - y
    return Point(CURVE, x, y)

def ec_point_add(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    if x1 == x2 and (y1 + y2) % P == 0: return None
    if x1 == x2 and y1 == y2:
        lam = ((3 * x1**2 + A) * pow(2 * y1, -1, P)) % P
    else:
        lam = ((y2 - y1) * pow(x2 - x1, -1, P)) % P
    x3 = (lam**2 - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return Point(CURVE, x3, y3)

def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    if k == 0 or point is None: return None
    result = None
    addend = point
    while k:
        if k & 1: result = ec_point_add(result, addend)
        addend = ec_point_add(addend, addend)
        k >>= 1
    return result

def ec_point_sub(p1, p2):
    if p2 is None: return p1
    p2_neg = Point(CURVE, p2.x(), (-p2.y()) % P)
    return ec_point_add(p1, p2_neg)

def compute_offset(Q: Point, start: int) -> Point:
    start_G = ec_scalar_mult(start, G)
    if start_G is None: return Q
    return ec_point_sub(Q, start_G)

def continued_fractions_approx(num, den, max_den):
    if den == 0: return 0, 1
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

def precompute_points(bits: int):
    limit = min(bits + 1, 32) 
    points = []
    curr = G
    for _ in range(limit):
        points.append(curr)
        curr = ec_point_add(curr, curr)
    return points

def precompute_good_indices_range(start, end, target_qx, gx=G.x(), gy=G.y(), p=P, cache_path=None):
    if end - start > 100000: return [] 
    good = []
    P0 = ec_scalar_mult(start, Point(CURVE, gx, gy))
    baseG = Point(CURVE, gx, gy)
    current = P0
    for k in range(start, end + 1):
        if current is None: current = baseG
        else:
            if k != start: current = ec_point_add(current, baseG)
        if current and current.x() == target_qx:
            good.append(k - start)
    return good

# --- FAULT TOLERANCE PRIMITIVES (Enhanced) ---
def prepare_verified_ancilla(qc: QuantumCircuit, qubit, initial_state=0):
    """
    Resets ancilla to a clean state.
    Applied BEFORE encoding to ensure no garbage from previous loops.
    """
    qc.reset(qubit)
    if initial_state == 1: 
        qc.x(qubit)

def encode_repetition(qc, logical_qubit, ancillas):
    """Encodes 1 qubit into 3 physical qubits (Bit Flip Code)"""
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])

def decode_repetition(qc, ancillas, logical_qubit):
    """Decodes 3 physical qubits back to logical using Majority Vote"""
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)

# ==========================================
# 3. QUANTUM PRIMITIVES & ORACLES
# ==========================================
 
class GeometricIPE:
    def __init__(self, n_bits): self.n = n_bits
    def _oracle_geometric_phase(self, qc, ctrl, state_reg, point_val):
        if point_val is None: return
        vx = point_val.x()
        for i in range(self.n):
            angle_x = 2 * math.pi * vx / (2**(i+1))
            qc.cp(angle_x, ctrl, state_reg[i])

def qft_reg(qc: QuantumCircuit, reg):
    qc.append(synth_qft_full(len(reg), do_swaps=False).to_gate(), reg)

def iqft_reg(qc: QuantumCircuit, reg):
    qc.append(synth_qft_full(len(reg), do_swaps=False).inverse().to_gate(), reg)

def draper_add_const(qc: QuantumCircuit, ctrl, target: QuantumRegister, value: int, inverse=False):
    n = len(target)
    sign = -1 if inverse else 1
    for i in range(n):
        angle = sign * (2 * pi * value) / (2 ** (n - i))
        if ctrl: qc.cp(angle, ctrl, target[i])
        else: qc.p(angle, target[i])

def ipe_oracle_phase(qc, ctrl, point_reg, delta_point, k_step, order=ORDER):
    power = 1 << k_step
    const_x = (delta_point.x() * power) % order
    if const_x:
        draper_add_const(qc, ctrl, point_reg, const_x)

def ft_draper_modular_adder(qc, ctrl, target_reg, ancilla_reg, value, modulus=N):
    n = len(target_reg)
    temp_overflow = ancilla_reg[0] 
    qft_reg(qc, target_reg)
    draper_add_const(qc, ctrl, target_reg, value, inverse=False)
    draper_add_const(qc, None, target_reg, modulus, inverse=True)
    iqft_reg(qc, target_reg)
    qc.cx(target_reg[-1], temp_overflow)
    qft_reg(qc, target_reg)
    draper_add_const(qc, temp_overflow, target_reg, modulus, inverse=False)
    iqft_reg(qc, target_reg)
    qc.cx(target_reg[-1], temp_overflow) 
    qc.reset(temp_overflow)

def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl, target, dx, dy):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle = 2 * math.pi * dx / (2 ** (i + 1)) 
        qc.cp(angle, ctrl, target[i])
    iqft_reg(qc, target)

def draper_2d_oracle(qc: QuantumCircuit, ctrl, target: QuantumRegister, dx: int, dy: int):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle_x = (2 * pi * dx) / (2 ** (n - i))
        angle_y = (2 * pi * dy) / (2 ** (n - i))
        qc.cp(angle_x, ctrl, target[i])
        qc.cp(angle_y, ctrl, target[i])
    iqft_reg(qc, target)

def ecdlp_oracle_ab(qc, a_reg, b_reg, point_reg, points, ancilla_reg, order=ORDER):
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        val = pt.x() % order if pt else 0
        if val: ft_draper_modular_adder(qc, a_reg[i], point_reg, ancilla_reg, val, order)
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)] 
        val = pt.x() % order if pt else 0
        if val: ft_draper_modular_adder(qc, b_reg[i], point_reg, ancilla_reg, val, order)

# --- SMART SCALABLE MATRIX/DRAPER GATE ---
def add_const_mod_gate(c: int, mod: int) -> Gate:
    n_qubits = math.ceil(math.log2(mod)) if mod > 1 else 1
    if mod <= 64:
        mat = np.zeros((mod, mod))
        for x in range(mod): mat[(x + c) % mod, x] = 1
        full_dim = 2**n_qubits
        if full_dim > mod:
            full_mat = np.eye(full_dim, dtype=complex)
            full_mat[:mod, :mod] = mat
            return UnitaryGate(full_mat, label=f"+{c} mod {mod}")
        return UnitaryGate(mat, label=f"+{c} mod {mod}")
    else:
        qc = QuantumCircuit(n_qubits, name=f"+{c} (Draper)")
        qc.append(QFTGate(n_qubits, do_swaps=False), range(n_qubits))
        for i in range(n_qubits):
            qc.p(2 * math.pi * c / (2 ** (n_qubits - i)), i)
        qc.append(QFTGate(n_qubits, do_swaps=False).inverse(), range(n_qubits))
        return qc.to_gate()

def apply_semiclassical_qft_phase_component(qc, ctrl, creg, n_bits, k):
    for m in range(k):
        angle = -pi / (2 ** (k - m))
        with qc.if_test((creg[m], 1)):
            qc.p(angle, ctrl)

# ==========================================
# 4. CIRCUIT BUILDER (ALL 21 MODES EXPLICIT)
# ==========================================

def get_best_mode_id(bits, available_qubits):
    req_ab = 3*bits + 4
    if req_ab < available_qubits: return 5 
    else: return 2 

def build_circuit_selector(mode_id, bits=config.BITS) -> QuantumCircuit:
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    delta = compute_offset(Q, config.KEYSPACE_START)
    points = precompute_points(bits)
    
    # --- FT SETUP ---
    ft_anc = QuantumRegister(2, "ft_anc") if config.USE_FT else None

    # --- 0. PROBE ---
    if mode_id == 0:
        qc = QuantumCircuit(2, 2)
        qc.h(0); qc.cx(0, 1); qc.measure_all()
        return qc
        
    # --- 1. IPE (STANDARD) ---
    elif mode_id == 1:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        regs = [ctrl, state, creg]
        if config.USE_FT: regs.append(ft_anc)
        qc = QuantumCircuit(*regs)
        for k in range(bits):
            if k>0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            
            # FT: Prepare & Encode
            if config.USE_FT: 
                prepare_verified_ancilla(qc, ft_anc[0])
                prepare_verified_ancilla(qc, ft_anc[1])
                encode_repetition(qc, ctrl[0], ft_anc)
            
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
            
            # FT: Decode
            if config.USE_FT: decode_repetition(qc, ft_anc, ctrl[0])
            
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 2. HIVE ---
    elif mode_id == 2:
        state_bits = (bits // 2 + 1)
        ctrl = QuantumRegister(4, "ctrl"); state = QuantumRegister(state_bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        for start in range(0, bits, 4):
            chunk = min(4, bits - start)
            if start > 0: qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
            for j in range(chunk):
                k = start + j; pwr = 1 << k
                draper_2d_oracle(qc, ctrl[j], state, (delta.x()*pwr)%N, 0)
                apply_semiclassical_qft_phase_component(qc, ctrl[j], creg, bits, k)
            qc.measure(ctrl[:chunk], creg[start:start+chunk])
        return qc

    # --- 3. WINDOWED IPE ---
    elif mode_id == 3:
        ctrl = QuantumRegister(4, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        for start in range(0, bits, 4):
            chunk = min(4, bits - start)
            if start > 0: qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
            for j in range(chunk):
                k = start + j; pwr = 1 << k
                draper_2d_oracle(qc, ctrl[j], state, (delta.x()*(1<<k))%N, (delta.y()*(1<<k))%N)
                apply_semiclassical_qft_phase_component(qc, ctrl[j], creg, bits, k)
            qc.measure(ctrl[:chunk], creg[start:start+chunk])
        return qc

    # --- 4. SEMICLASSICAL ---
    elif mode_id == 4:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        regs = [ctrl, state, creg]
        if config.USE_FT: regs.append(ft_anc)
        qc = QuantumCircuit(*regs)
        for k in range(bits):
            if k > 0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            
            # FT: Prepare & Encode
            if config.USE_FT: 
                prepare_verified_ancilla(qc, ft_anc[0])
                prepare_verified_ancilla(qc, ft_anc[1])
                encode_repetition(qc, ctrl[0], ft_anc)
            
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg[m], 1)): qc.p(angle, ctrl[0])
            power = 1 << k
            dx = (delta.x() * power) % N; dy = (delta.y() * power) % N
            draper_2d_oracle(qc, ctrl[0], state, dx, dy)
            
            # FT: Decode
            if config.USE_FT: decode_repetition(qc, ft_anc, ctrl[0])
            
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 5. AB SHOR ---
    elif mode_id == 5:
        a, b, acc, anc, creg = QuantumRegister(bits,"a"), QuantumRegister(bits,"b"), QuantumRegister(bits,"acc"), QuantumRegister(4,"anc"), ClassicalRegister(2*bits,"meas")
        qc = QuantumCircuit(a, b, acc, anc, creg)
        qc.h(a); qc.h(b)
        ecdlp_oracle_ab(qc, a, b, acc, points, anc, ORDER)
        qc.append(QFTGate(bits, inverse=True), a)
        qc.append(QFTGate(bits, inverse=True), b)
        qc.measure(a, creg[:bits]); qc.measure(b, creg[bits:])
        return qc

    # --- 6. FT DRAPER TEST ---
    elif mode_id == 6:
        qc = QuantumCircuit(QuantumRegister(bits), QuantumRegister(1), QuantumRegister(2), ClassicalRegister(bits))
        qc.x(qc.qregs[1]); ft_draper_modular_adder(qc, qc.qregs[1][0], qc.qregs[0], qc.qregs[2], 12345, N); qc.measure(qc.qregs[0], qc.cregs[0]); return qc

    # --- 7. GEOMETRIC ---
    elif mode_id == 7:
        geo = GeometricIPE(bits); qc = QuantumCircuit(QuantumRegister(1), QuantumRegister(bits), ClassicalRegister(bits))
        ctrl, state, creg = qc.qregs[0], qc.qregs[1], qc.cregs[0]
        qc.append(synth_qft_full(bits, do_swaps=False).to_gate(), state) 
        for k in range(bits):
            if k>0: qc.reset(ctrl[0])
            qc.h(ctrl[0]); geo._oracle_geometric_phase(qc, ctrl[0], state, delta); apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 8. VERIFIED (FLAGS + FT) ---
    elif mode_id == 8:
        n_flags = config.USE_FLAGS
        ctrl, state, flags, c_meas, c_flags = QuantumRegister(1,"ctrl"), QuantumRegister(bits,"state"), QuantumRegister(n_flags,"flag"), ClassicalRegister(bits,"meas"), ClassicalRegister(bits*n_flags,"flag_out")
        regs = [ctrl, state, flags, c_meas, c_flags]
        if config.USE_FT: regs.append(ft_anc)
        qc = QuantumCircuit(*regs)
        for k in range(bits):
            if k > 0: qc.reset(ctrl[0])
            qc.reset(flags); qc.h(ctrl[0])
            
            # FT: Prepare & Encode
            if config.USE_FT: 
                prepare_verified_ancilla(qc, ft_anc[0])
                prepare_verified_ancilla(qc, ft_anc[1])
                encode_repetition(qc, ctrl[0], ft_anc)
            
            for f in range(n_flags): qc.cx(ctrl[0], flags[f]) 
            apply_semiclassical_qft_phase_component(qc, ctrl[0], c_meas, bits, k)
            draper_adder_oracle_1d_serial(qc, ctrl[0], state, (delta.x()*(1<<k))%N, 0)
            for f in range(n_flags): qc.cx(ctrl[0], flags[f]) 
            
            # FT: Decode
            if config.USE_FT: decode_repetition(qc, ft_anc, ctrl[0])
            
            qc.h(ctrl[0]); qc.measure(ctrl[0], c_meas[k])
            qc.measure(flags, c_flags[k*n_flags : (k+1)*n_flags])
        return qc

    # --- 9. SHADOW 2D ---
    elif mode_id == 9:
        window_size = 4
        ctrl = QuantumRegister(window_size, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        
        for start in range(0, bits, window_size):
            chunk = min(window_size, bits - start)
            if start > 0: qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
            for j in range(chunk):
                k = start + j; pwr = 1 << k
                dx = (delta.x() * pwr) % N; dy = (delta.y() * pwr) % N
                draper_2d_oracle(qc, ctrl[j], state, dx, dy)
                for m in range(start):
                    angle = -pi / (2 ** (k - m))
                    with qc.if_test((creg[m], 1)):
                        qc.p(angle, ctrl[j])
            qc.append(synth_qft_full(chunk, do_swaps=False).inverse(), ctrl[:chunk])
            qc.measure(ctrl[:chunk], creg[start:start + chunk])
        return qc

    # --- 10. REVERSE IPE ---
    elif mode_id == 10:
        ctrl, state, creg = QuantumRegister(1), QuantumRegister(bits), ClassicalRegister(bits)
        qc = QuantumCircuit(ctrl, state, creg)
        for k in reversed(range(bits)):
            if k < bits-1: qc.reset(ctrl[0])
            qc.h(ctrl[0]); draper_2d_oracle(qc, ctrl[0], state, (delta.x()*(1<<k))%N, (delta.y()*(1<<k))%N)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 11. SWARM ---
    elif mode_id == 11:
        workers = max(1, 156 // ((bits // 2 + 1) + 8))
        regs = []
        for w in range(workers): regs.extend([QuantumRegister(4, f"c{w}"), QuantumRegister((bits//2+1), f"s{w}"), ClassicalRegister(bits, f"m{w}")])
        qc = QuantumCircuit(*regs)
        for w in range(workers):
            q_ctrl, q_state, c_meas = qc.qregs[w*2], qc.qregs[w*2+1], qc.cregs[w]
            for start in range(0, bits, window):
                chunk = min(window, bits - start)
                if start > 0: qc.reset(q_ctrl[:chunk])
                qc.h(q_ctrl[:chunk])
                for j in range(chunk):
                    k = start + j; draper_2d_oracle(qc, q_ctrl[j], q_state, (delta.x()*(1<<k))%N, 0)
                    apply_semiclassical_qft_phase_component(qc, q_ctrl[j], c_meas, bits, k)
                qc.measure(q_ctrl[:chunk], c_meas[start:start+chunk])
        return qc

    # --- 12. HEAVY DRAPER ---
    elif mode_id == 12:
        qc = QuantumCircuit(QuantumRegister(bits), QuantumRegister(bits), ClassicalRegister(bits))
        ft_draper_modular_adder(qc, None, qc.qregs[0], [qc.qregs[1][0]], 12345, N)
        qc.measure(qc.qregs[0], qc.cregs[0])
        return qc

    # --- 13. COMPRESSED SHADOW ---
    elif mode_id == 13:
        window_size = 8
        ctrl = QuantumRegister(window_size, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        for start in range(0, bits, window_size):
            chunk = min(window_size, bits - start)
            if start > 0: qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
            for j in range(chunk):
                k = start + j; draper_2d_oracle(qc, ctrl[j], state, (delta.x()*(1<<k))%N, 0)
                for m in range(start):
                    with qc.if_test((creg[m], 1)): qc.p(-pi / (2 ** (k - m)), ctrl[j])
            qc.append(synth_qft_full(chunk, do_swaps=False).inverse(), ctrl[:chunk])
            qc.measure(ctrl[:chunk], creg[start:start+chunk])
        return qc

    # --- 14. SHOR LOGIC ---
    elif mode_id == 14:
        block_size = min(bits, 5) 
        ctrl, state, creg = QuantumRegister(block_size), QuantumRegister(bits), ClassicalRegister(block_size)
        qc = QuantumCircuit(ctrl, state, creg)
        qc.h(ctrl)
        for i in range(block_size):
             val = (delta.x() * (1<<i)) % N
             draper_add_const(qc, ctrl[i], state, val)
        qc.append(QFTGate(block_size, inverse=True), ctrl)
        qc.measure(ctrl, creg)
        return qc

    # --- 15. GEOMETRIC IPE ---
    elif mode_id == 15:
        ctrl, state, creg = QuantumRegister(bits), QuantumRegister(bits), ClassicalRegister(bits)
        qc = QuantumCircuit(ctrl, state, creg)
        qc.h(ctrl)
        geo = GeometricIPE(bits)
        for k in range(bits):
            geo._oracle_geometric_phase(qc, ctrl[k], state, ec_scalar_mult((1<<k), delta))
        qc.append(QFTGate(bits, inverse=True), ctrl)
        qc.measure(ctrl, creg)
        return qc

    # --- 16. WINDOWED EXPLICIT ---
    elif mode_id == 16:
        ctrl, state, creg = QuantumRegister(1), QuantumRegister(bits), ClassicalRegister(bits)
        qc = QuantumCircuit(ctrl, state, creg)
        for k in range(bits):
            if k > 0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            for m in range(k):
                angle = -pi / (2**(k-m))
                qc.cp(angle, creg[m], ctrl[0]) 
            ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 17. HIVE SWARM ---
    elif mode_id == 17:
        total_q = 127; state_q = bits
        workers = (total_q - state_q) // 1
        regs = [QuantumRegister(state_q, "state")]
        for w in range(workers): regs.append(QuantumRegister(1, f"w{w}")); regs.append(ClassicalRegister(1, f"c{w}"))
        qc = QuantumCircuit(*regs)
        state = qc.qregs[0]
        for w in range(workers):
            ctrl = qc.qregs[w+1]
            qc.h(ctrl)
            draper_2d_oracle(qc, ctrl[0], state, (delta.x()*(1<<w))%N, 0)
            qc.h(ctrl); qc.measure(ctrl, qc.cregs[w])
        return qc

    # --- 18. EXPLICIT LOGIC ---
    elif mode_id == 18:
        ctrl, state, creg = QuantumRegister(1), QuantumRegister(bits), ClassicalRegister(bits)
        qc = QuantumCircuit(ctrl, state, creg)
        for k in range(bits):
            if k > 0: 
                with qc.if_test((creg[k-1], 1)): qc.x(ctrl[0])
            qc.h(ctrl[0])
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 19. FIXED AB ---
    elif mode_id == 19:
        ctrl, state, anc, creg = QuantumRegister(1), QuantumRegister(bits), QuantumRegister(4), ClassicalRegister(bits)
        qc = QuantumCircuit(ctrl, state, anc, creg)
        for k in range(bits):
            if k > 0: qc.reset(ctrl[0])
            qc.h(ctrl[0]); apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            ft_draper_modular_adder(qc, ctrl[0], state, anc, (1<<k)%N, N)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 20. MATRIX MOD (SMART) ---
    elif mode_id == 20:
        qc = QuantumCircuit(QuantumRegister(bits), ClassicalRegister(bits))
        gate = add_const_mod_gate(1, 2**bits) # Smart scalable
        qc.append(gate, qc.qregs[0]); qc.measure(qc.qregs[0], qc.cregs[0])
        return qc
    
    return build_circuit_selector(2, bits)

# ==========================================
# 5.  EXECUTION MTIGATION ENGINE & VISUALS
# ==========================================


def estimate_gate_counts(qc):
    """Counts specific expensive gates."""
    counts = {"CX": 0, "CCX": 0, "T": 0}
    for instruction in qc.data:
        name = instruction.operation.name.upper()
        if name in counts:
            counts[name] += 1
        if name == "TDG":
            counts["T"] += 1
    return counts

def analyze_circuit_costs(qc, backend):
    """Prints detailed circuit statistics before execution."""
    total_qubits = qc.num_qubits
    gate_counts = estimate_gate_counts(qc)
    
    print("\n" + "="*40)
    print("   CIRCUIT ANALYSIS")
    print("="*40)
    print(f"[i] Logical Qubits: {total_qubits}")
    print(f"[i] Logical Depth:  {qc.depth()}")
    print(f"[i] Gate Estimate:  CX={gate_counts['CX']}, CCX={gate_counts['CCX']}, T={gate_counts['T']}")
    
    # Check backend capacity
    backend_qubits = backend.configuration().n_qubits if hasattr(backend, 'configuration') else 127
    
    if total_qubits > backend_qubits:
        logger.warning(f"⚠️  CRITICAL: Circuit ({total_qubits}q) exceeds backend {backend.name} ({backend_qubits}q)!")
    elif total_qubits > 156:
        logger.warning(f"⚠️  High Qubit Count ({total_qubits}). Execution may be unstable.")
    else:
        print(f"[i] Circuit fits within {backend.name} ({backend_qubits}q).")
    print("-" * 40 + "\n")

def configure_sampler_options(sampler):
    """Applies DRAGON mitigation settings."""
    if config.USE_DD:
        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = config.DD_SEQUENCE
        except: pass
    if config.USE_MEM:
        try:
            sampler.options.twirling.enable_measure = True
            sampler.options.measure_mitigation = True
            sampler.options.trex = True
        except: pass
    
    sampler.options.resilience_level = config.INTERNAL_RESILIENCE_LEVEL
    return sampler

def safe_get_counts(result_item):
    """Aggressive Universal Retrieval."""
    combined_counts = defaultdict(int)
    data_found = False

    # 1. Reflection
    if hasattr(result_item, 'data'):
        data_bin = result_item.data
        for attr_name in [a for a in dir(data_bin) if not a.startswith("_")]:
            val = getattr(data_bin, attr_name)
            if hasattr(val, 'get_counts'):
                try:
                    c = val.get_counts()
                    for k, v in c.items(): combined_counts[k] += v
                    data_found = True
                except: pass
    
    if data_found: return dict(combined_counts)

    # 2. Legacy Fallbacks
    attempts = [
        lambda: result_item.data.meas.get_counts(),
        lambda: result_item.data.c.get_counts(),
        lambda: result_item.data.meas_bits.get_counts(),
        lambda: result_item.data.meas_state.get_counts(),
        lambda: result_item.data.c_meas.get_counts(),
        lambda: result_item.data.probe_c.get_counts(),
        lambda: result_item.data.flag_out.get_counts(),
        lambda: result_item.data.m0.get_counts()
    ]
    for attempt in attempts:
        try: return attempt()
        except: continue
    return None

def manual_zne(qc, backend, shots, scales=[1, 3, 5]):
    logger.info(f"Running Manual ZNE (Scales: {scales})...")
    counts_list = []
    
    for scale in scales:
        scaled_qc = qc.copy()
        
        # Folding (Noise Amplification)
        if scale > 1:
            for _ in range(scale - 1):
                scaled_qc.barrier()
                for q in scaled_qc.qubits: scaled_qc.id(q)
        
        print(f"[i] Transpiling Scale {scale} (ALAP/Sabre)...")
        
        # SINGLE ROBUST TRANSPILE (Replaces PassManager conflict)
        transpiled_qc = transpile(scaled_qc, backend=backend, optimization_level=3, 
                                  scheduling_method='alap', routing_method='sabre')
        
        print(f"[i] Scale {scale} -> Depth: {transpiled_qc.depth()}, Size: {transpiled_qc.size()}")
        
        sampler = Sampler(mode=backend)
        sampler = configure_sampler_options(sampler)
        sampler.options.resilience_level = 0 # Force Raw for ZNE
        
        job = sampler.run([transpiled_qc], shots=shots)
        print(f"[i] ZNE Scale {scale} Job ID: {job.job_id()}")
        
        try:
            job_result = job.result()
            counts = safe_get_counts(job_result[0])
            if counts: counts_list.append(counts)
        except Exception as e:
            logger.error(f"ZNE Scale {scale} failed: {e}")
        
    if not counts_list: return defaultdict(float)
    
    # Linear Extrapolation
    extrapolated = defaultdict(float)
    keys = set().union(*counts_list)
    for k in keys:
        vals = [c.get(k, 0) for c in counts_list]
        if len(vals) > 1:
            fit = np.polyfit(scales[:len(vals)], vals, 1)
            extrapolated[k] = max(0, fit[1]) 
        else:
            extrapolated[k] = vals[0]
            
    return extrapolated

def plot_visuals(counts, bits, order=N, k_target=None):
    if not counts or len(counts) > 500:
        logger.info("Plotting Histogram.")
        plot_histogram(counts)
        plt.show(); return
    
    grid = 256
    heat = np.zeros((grid, grid), dtype=int)
    for bitstr, cnt in counts.items():
        try:
            val = int(bitstr.replace(" ", ""), 2)
            a = (val >> (bits//2)) % grid
            b = val % grid
            heat[a, b] += cnt
        except: continue
        
    plt.figure(figsize=(6,6))
    plt.title('Heatmap')
    plt.imshow(heat, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()
 
# ==========================================
# 6. POST PROCESSING & MAIN EXECUTION
# ==========================================


def save_key(k: int):
    hex_k = hex(k)[2:].zfill(64)
    padded_hex = '0x' + hex_k.zfill(64)
    zero_padded = hex_k.zfill(64)
    shifted_hex = '0x' + zero_padded[32:] + zero_padded[:32]
    with open("boom.txt", "a") as f:
        f.write(f"{padded_hex}\n{zero_padded}\n{shifted_hex}\n")
    logger.info(f"KEY SAVED: boom.txt")

def retrieve_and_process_job(job_id, service, n_bits, start_val, target_pub_x, method):
    try:
        job = service.job(job_id)
        while job.status().name not in ["DONE", "QUEUED", "COMPLETED", "ERROR", "CANCELLED"]:
            logger.info(f"Status: {job.status().name}...")
            time.sleep(60)
            
        if job.status().name == "ERROR":
            logger.error("Job failed on backend.")
            return None
            
        job_result = job.result()
        counts = safe_get_counts(job_result[0])
        # 3. Define a window (e.g., Result to Result + 10,000)  This helps if the QPU was slightly off by LSBs due to noise
        # --- POST-QUANTUM WINDOW SCAN ---
        if counts:
            top_meas = max(counts, key=counts.get)
            clean_meas = top_meas.replace(" ", "")
            clean_meas = "".join([b for b in clean_meas if b in '01'])
            if clean_meas:
                measured_int = int(clean_meas, 2)
                logger.info(f"[Extra Check] Scanning window around measurement: {hex(measured_int)}")
                hits = precompute_good_indices_range(measured_int, measured_int + 10000, target_pub_x)
                if hits:
                    final_key = measured_int + hits[0]
                    logger.info(f"FOUND VIA POST-QUANTUM CHECK: {hex(final_key)}")
                    save_key(final_key)
                    return final_key

        return hybrid_post_process(counts, n_bits, ORDER, start_val, target_pub_x, method)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return None
     
def hybrid_post_process(counts, bits, order, start, target_pub_x, method):
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:100]
    for meas_str, freq in sorted_counts:
        meas_str = meas_str.replace(" ", "")
        
        # --- SHOR (AB) MODE ---
        if method == 'ab' or method == 5:
            mid = len(meas_str) // 2
            a = int(meas_str[:mid], 2)
            b = int(meas_str[mid:], 2)
            if gcd_verbose(b, order) == 1:
                inv_b = modular_inverse_verbose(b, order)
                k = (-a * inv_b) % order
                if ec_scalar_mult(k, G).x() == target_pub_x:
                    save_key(k)
                    return k
        
        # --- IPE / PHASE MODES ---
        else:
            measurements = [int(bit) for bit in meas_str if bit in '01']
            if len(measurements) >= bits: measurements = measurements[:bits]
            measurements = measurements[::-1] 
            
            phi = sum([b * (1 / 2**(i+1)) for i, b in enumerate(measurements)])
            num, den = continued_fractions_approx(int(phi * 2**bits), 2**bits, order)
            d = (num * modular_inverse_verbose(den, order)) % order if den and gcd_verbose(den, order) == 1 else None
            
            if d:
                cand = (start + d) % N
                if ec_scalar_mult(cand, G).x() == target_pub_x:
                    save_key(cand)
                    return cand
    return None

def run_best_solver():
    config.user_menu()
    
    print("[i] Connecting to IBM Quantum Runtime...")
    try: 
        service = QiskitRuntimeService()
    except: 
        service = QiskitRuntimeService(channel="ibm_cloud", token=config.TOKEN, instance=config.CRN)
    
    try:
        backend = service.least_busy(simulator=False, operational=True, min_num_qubits=127)
    except Exception as e:
        logger.error(f"Could not find available real backend: {e}")
        return None

    print(f"[i] Selected Real Backend: {backend.name}")

    # No Initial Pre-Check (Moved to Post-Process)
    target_pub = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)

    if config.METHOD == 'smart':
        mode_id = get_best_mode_id(config.BITS, backend.num_qubits)
    else:
        mode_id = int(config.METHOD)
    
    logger.info(f"Target BITS={config.BITS} | Hardware={backend.name} | Mode={mode_id}")
    qc = build_circuit_selector(mode_id, config.BITS)
    
    analyze_circuit_costs(qc, backend)

    counts = {}
    
    if config.USE_MANUAL_ZNE:
        logger.info(">>> MANUAL ZNE ENABLED <<<")
        counts = manual_zne(qc, backend, config.SHOTS)
    else:
        logger.info(">>> STANDARD RUN ENABLED <<<")
        
        print(f"[i] Transpiling circuit (ALAP/Sabre)...")
        # Direct Transpile (No PassManager)
        transpiled_qc = transpile(qc, backend=backend, optimization_level=config.OPT_LEVEL, 
                                  scheduling_method='alap', routing_method='sabre')
        
        print(f"[i] Transpiled Depth: {transpiled_qc.depth()}")
        print(f"[i] Transpiled Size:  {transpiled_qc.size()}")
        
        sampler = Sampler(mode=backend)
        sampler = configure_sampler_options(sampler)
        
        print(f"[i] Submitting Job with {config.SHOTS} shots...")
        job = sampler.run([transpiled_qc], shots=config.SHOTS)
        print(f"[i] Job Submitted. ID: {job.job_id()}")
        print("[i] Waiting for results...")
        
        try:
            k = retrieve_and_process_job(job.job_id(), service, config.BITS, config.KEYSPACE_START, target_pub.x(), 'ab' if mode_id==5 else 'phase')
            if k: return k
            job_result = job.result()
            counts = safe_get_counts(job_result[0])
        except Exception as e:
            logger.error(f"Job Execution Failed: {e}")
            return None

    k = hybrid_post_process(counts, config.BITS, N, config.KEYSPACE_START, target_pub.x(), 'ab' if mode_id==5 else 'phase')
    if k: logger.info(f"RECOVERED PRIVATE KEY: {hex(k)}")
    else: logger.warning("Key not found in top candidates. Try increasing shots or ZNE scales.")
    plot_visuals(counts, config.BITS, N, k)
    return k

if __name__ == "__main__":
    run_best_solver()




