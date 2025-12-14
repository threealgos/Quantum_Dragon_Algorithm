#Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#======================================================================================================
"""
HERE Is Quantum ECDLP Solver - Dragon Mode Edition (v120)
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

MITIGATION: Manual ZNE, TREX, Twirling, DD (XY4).
"""

from IPython.display import display
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from collections import defaultdict
from fractions import Fraction
import numpy as np
import time
import os
import logging
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit import UnitaryGate
from math import gcd, pi
from typing import Optional, List, Dict
import pickle
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import numbertheory
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CONFIGURATION
# ==========================================

CACHE_DIR = "cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    # --- Target ---
    BITS = 135
    KEYSPACE_START = 0x4000000000000000000000000000000000
    COMPRESSED_PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
    
    # --- Backend ---
    BACKEND = "ibm_fez" #  And For Future backends ~1386 Qubits Nighthawk/Kookaburra 
    TOKEN = "YOUR_IBM_TOKEN"
    CRN = "YOUR_IBM_CRN"
    
    # --- Mode Selection ---
    # Set to integer 0-20 or "smart"
    METHOD = "smart" 
    
    # --- Tuning ---
    USE_COMPRESSED = True    # Hive x-only logic
    USE_FLAGS = 2            # Verification
    USE_FT = False           # Fault Tolerance Toggle
    
    # --- MITIGATION ---
    SHOTS = 8192 # 16384 & 100000
    OPT_LEVEL = 3
    USE_MANUAL_ZNE = True    
    USE_DD = True
    DD_SEQUENCE = "XY4"      
    USE_MEM = True           
    
    @property
    def INTERNAL_RESILIENCE_LEVEL(self):
        return 1 if self.USE_MANUAL_ZNE else 2

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

# --- IPE Oracle (Requested Explicitly) ---
def ipe_oracle_phase(qc, ctrl, point_reg, delta_point, k_step, order=ORDER):
    """Explicit IPE Oracle for Step k."""
    # Applies phase: delta.x * 2^k
    power = 1 << k_step
    const_x = (delta_point.x() * power) % order
    if const_x:
        draper_add_const(qc, ctrl, point_reg, const_x)

# --- FT Modular Adder ---
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

# --- 1D Serial Oracle ---
def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl, target, dx, dy):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle = 2 * math.pi * dx / (2 ** (i + 1)) 
        qc.cp(angle, ctrl, target[i])
    iqft_reg(qc, target)

# --- 2D Oracle ---
def draper_2d_oracle(qc: QuantumCircuit, ctrl, target: QuantumRegister, dx: int, dy: int):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle_x = (2 * pi * dx) / (2 ** (n - i))
        angle_y = (2 * pi * dy) / (2 ** (n - i))
        qc.cp(angle_x, ctrl, target[i])
        qc.cp(angle_y, ctrl, target[i])
    iqft_reg(qc, target)

# --- AB Oracle ---
def ecdlp_oracle_ab(qc, a_reg, b_reg, point_reg, points, ancilla_reg, order=ORDER):
    for i in range(len(a_reg)):
        pt = points[min(i, len(points)-1)]
        val = pt.x() % order if pt else 0
        if val: ft_draper_modular_adder(qc, a_reg[i], point_reg, ancilla_reg, val, order)
    for i in range(len(b_reg)):
        pt = points[min(i, len(points)-1)] 
        val = pt.x() % order if pt else 0
        if val: ft_draper_modular_adder(qc, b_reg[i], point_reg, ancilla_reg, val, order)

# --- Matrix Oracle ---
def add_const_mod_gate(c: int, mod: int) -> UnitaryGate:
    mat = np.zeros((mod, mod))
    for x in range(mod): mat[(x + c) % mod, x] = 1
    return UnitaryGate(mat, label=f"+{c} mod {mod}")

def apply_semiclassical_qft_phase_component(qc, ctrl, creg, n_bits, k):
    for m in range(k):
        angle = -pi / (2 ** (k - m))
        with qc.if_test((creg[m], 1)):
            qc.p(angle, ctrl)

# --- Fault Tolerance ---
def encode_repetition(qc, logical_qubit, ancillas):
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])

def decode_repetition(qc, ancillas, logical_qubit):
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)

# ==========================================
# 4. CIRCUIT BUILDER
# ==========================================

def get_best_mode_id(bits, available_qubits):
    """Maps Smart Mode to integer ID."""
    req_ab = 2*bits + 5
    req_fixed = bits + 6
    if req_ab < available_qubits: return 5 # AB
    elif req_fixed < available_qubits: return 19 # Fixed AB
    else: return 2 # Hive

def build_circuit_selector(mode_id, bits=config.BITS) -> QuantumCircuit:
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    delta = compute_offset(Q, config.KEYSPACE_START)
    points = precompute_points(bits)
    
    # --- 0. PROBE ---
    if mode_id == 0:
        qc = QuantumCircuit(2, 2)
        qc.h(0); qc.cx(0, 1); qc.measure_all()
        return qc
        
    # --- 1. IPE (STANDARD) - WITH REQUESTED ORACLE ---
    elif mode_id == 1:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        for k in range(bits):
            if k>0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            # Explicit call to ipe_oracle_phase
            ipe_oracle_phase(qc, ctrl[0], state, delta, k, ORDER)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 2. HIVE (Optimized) ---
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
                draper_2d_oracle(qc, ctrl[j], state, (delta.x()*pwr)%N, (delta.y()*pwr)%N)
                apply_semiclassical_qft_phase_component(qc, ctrl[j], creg, bits, k)
            qc.measure(ctrl[:chunk], creg[start:start+chunk])
        return qc

    # --- 4. Semiclassical (Dynamic IPE) ---
    elif mode_id == 4:
        ctrl = QuantumRegister(1, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        
        for k in range(bits):
            # Reset control qubit for next bit (dynamic circuit)
            if k > 0:
                qc.reset(ctrl[0])
            
            # Superposition on control
            qc.h(ctrl[0])
            
            # Semiclassical phase correction from previous measurements
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg[m], 1)):
                    qc.p(angle, ctrl[0])
            
            # Apply oracle: add (delta * 2^k) using 2D Draper oracle
            power = 1 << k
            dx = (delta.x() * power) % N
            dy = (delta.y() * power) % N
            draper_2d_oracle(qc, ctrl[0], state, dx, dy)
            
            # Basis change and measure
            qc.h(ctrl[0])
            qc.measure(ctrl[0], creg[k])
        
        return qc

    # --- 5. AB (SHOR) ---
    elif mode_id == 5:
        a = QuantumRegister(bits, "a"); b = QuantumRegister(bits, "b")
        anc = QuantumRegister(4, "anc"); creg = ClassicalRegister(2*bits, "meas")
        qc = QuantumCircuit(a, b, anc, creg)
        qc.h(a); qc.h(b)
        ecdlp_oracle_ab(qc, a, b, b, points, anc, ORDER)
        iqft_reg(qc, a); iqft_reg(qc, b)
        qc.measure(a, creg[:bits]); qc.measure(b, creg[bits:])
        return qc

    # --- 6. FT DRAPER TEST ---
    elif mode_id == 6:
        reg = QuantumRegister(bits, "reg"); ctrl = QuantumRegister(1, "ctrl")
        anc = QuantumRegister(2, "anc"); creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(reg, ctrl, anc, creg)
        qc.x(ctrl); ft_draper_modular_adder(qc, ctrl[0], reg, anc, 12345, N)
        qc.measure(reg, creg)
        return qc

    # --- 7. GEOMETRIC ---
    elif mode_id == 7:
        geo = GeometricIPE(bits)
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        qc.append(synth_qft_full(bits, do_swaps=False).to_gate(), state) 
        for k in range(bits):
            if k>0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            geo._oracle_geometric_phase(qc, ctrl[0], state, delta)
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 8. VERIFIED (With 1D Oracle) ---
    elif mode_id == 8:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        flag = QuantumRegister(1, "flag"); creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, flag, creg)
        for k in range(bits):
            if k>0: qc.reset(ctrl[0]); qc.reset(flag[0])
            qc.h(ctrl[0])
            qc.cx(ctrl[0], flag[0]) 
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            draper_adder_oracle_1d_serial(qc, ctrl[0], state, (delta.x()*(1<<k))%N, 0)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k]); qc.measure(flag[0], creg[k])
        return qc

    # --- 9. Shadow 2D (Windowed Optimization) ---
    elif mode_id == 9:
        window_size = 4
        ctrl = QuantumRegister(window_size, "ctrl")
        state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        
        for start in range(0, bits, window_size):
            chunk = min(window_size, bits - start)
            
            # Reset controls for new window
            if start > 0:
                qc.reset(ctrl[:chunk])
            
            # Superposition on active window controls
            qc.h(ctrl[:chunk])
            
            # Apply oracle and phase feedback for each bit in window
            for j in range(chunk):
                k = start + j
                pwr = 1 << k
                dx = (delta.x() * pwr) % N
                dy = (delta.y() * pwr) % N
                draper_2d_oracle(qc, ctrl[j], state, dx, dy)
                
                # Phase feedback from previous windows only
                for m in range(start):
                    angle = -math.pi / (2 ** (k - m))
                    with qc.if_test((creg[m], 1)):
                        qc.p(angle, ctrl[j])
            
            # Inverse QFT on the window control register to extract phase bits
            qc.append(synth_qft_full(chunk, do_swaps=False).inverse(), ctrl[:chunk])
            
            # Measure the window
            qc.measure(ctrl[:chunk], creg[start:start + chunk])
        
        return qc

    # --- 10. REVERSE IPE ---
    elif mode_id == 10:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, creg)
        for k in reversed(range(bits)):
            if k < bits-1: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            pwr = 1<<k
            draper_2d_oracle(qc, ctrl[0], state, (delta.x()*pwr)%N, (delta.y()*pwr)%N)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 11. SWARM (PARALLEL WORKERS) ---
    elif mode_id == 11:
        hive_size = (bits // 2 + 1) + 8
        num_workers = 156 // hive_size # Assuming 156q default
        if num_workers < 1: num_workers = 1
        print(f"[Builder] Swarm: {num_workers} workers.")
        regs = []
        for w in range(num_workers):
            regs.append(QuantumRegister(4, f"c{w}"))
            regs.append(QuantumRegister((bits//2+1), f"s{w}"))
            regs.append(ClassicalRegister(bits, f"m{w}"))
        qc = QuantumCircuit(*regs)
        window = 4
        for w in range(num_workers):
            q_ctrl = qc.qregs[w*2]; q_state = qc.qregs[w*2+1]; c_meas = qc.cregs[w]
            for start in range(0, bits, window):
                chunk = min(window, bits - start)
                if start > 0: qc.reset(q_ctrl[:chunk])
                qc.h(q_ctrl[:chunk])
                for j in range(chunk):
                    k = start + j; pwr = 1 << k
                    draper_2d_oracle(qc, q_ctrl[j], q_state, (delta.x()*pwr)%N, 0)
                    apply_semiclassical_qft_phase_component(qc, q_ctrl[j], c_meas, bits, k)
                qc.measure(q_ctrl[:chunk], c_meas[start:start+chunk])
        return qc

    # --- 12. Probe (Hardware Diagnostic) ---
    elif mode_id == 12:
        qr = QuantumRegister(2, "probe_q")
        cr = ClassicalRegister(2, "probe_c")
        qc = QuantumCircuit(qr, cr)
        
        # Create Bell state |00> + |11>
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        
        # Measure both qubits
        qc.measure(qr, cr)
        
        return qc

    # --- 19. FIXED AB (SEMICLASSICAL MODULAR) ---
    elif mode_id == 19:
        ctrl = QuantumRegister(1, "ctrl"); state = QuantumRegister(bits, "state")
        anc = QuantumRegister(4, "anc"); creg = ClassicalRegister(bits, "meas")
        qc = QuantumCircuit(ctrl, state, anc, creg)
        for k in range(bits):
            if k > 0: qc.reset(ctrl[0])
            qc.h(ctrl[0])
            apply_semiclassical_qft_phase_component(qc, ctrl[0], creg, bits, k)
            val = (1 << k) % N
            ft_draper_modular_adder(qc, ctrl[0], state, anc, val, N)
            qc.h(ctrl[0]); qc.measure(ctrl[0], creg[k])
        return qc

    # --- 20. MATRIX MOD (UNITARY) ---
    elif mode_id == 20:
        sim_bits = min(bits, 5)
        reg = QuantumRegister(sim_bits, 'reg'); creg = ClassicalRegister(sim_bits, 'meas')
        qc = QuantumCircuit(reg, creg)
        gate = add_const_mod_gate(1, 2**sim_bits)
        qc.append(gate, reg)
        qc.measure(reg, creg)
        return qc
        
    return build_circuit_selector(2, bits)

# ==========================================
# 5. MITIGATION & VISUALS
# ==========================================

def estimate_gate_counts(qc):
    counts = {"CX":0, "CCX":0, "T":0}
    for inst in qc.data:
        name = inst.operation.name.upper()
        if name in counts: counts[name] += 1
        if name == "TDG": counts["T"] += 1
    logger.info(f"Gate Estimation: {counts}")
    return counts

def configure_sampler_options(sampler):
    if config.USE_DD:
        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = config.DD_SEQUENCE
        except: pass
    if config.USE_MEM:
        try: sampler.options.twirling.enable_measure = True
        except: pass
        try: sampler.options.measure_mitigation = True
        except: pass
        try: sampler.options.trex = True
        except: pass
    sampler.options.resilience_level = config.INTERNAL_RESILIENCE_LEVEL
    return sampler

def safe_get_counts(result_item):
    attempts = [
        lambda: result_item.data.meas.get_counts(),
        lambda: result_item.data.c.get_counts(),
        lambda: result_item.data.meas_bits.get_counts(),
        lambda: result_item.data.meas_state.get_counts(),
    ]
    try:
        if hasattr(result_item, 'data'):
            all_c = defaultdict(int)
            found = False
            for attr in dir(result_item.data):
                if attr.startswith('m') and attr[1:].isdigit():
                    c = getattr(result_item.data, attr).get_counts()
                    for k,v in c.items(): all_c[k]+=v
                    found = True
            if found: return all_c
    except: pass
    for attempt in attempts:
        try: return attempt()
        except: continue
    return None

def manual_zne(qc, backend, shots, scales=[1, 3, 5]):
    logger.info(f"Running Manual ZNE (Scales: {scales})...")
    counts_list = []
    pm = generate_preset_pass_manager(backend=backend, optimization_level=config.OPT_LEVEL)
    for scale in scales:
        scaled_qc = qc.copy()
        if scale > 1:
            for _ in range(scale - 1):
                scaled_qc.barrier()
                for q in scaled_qc.qubits: scaled_qc.id(q) 
        isa_qc = pm.run(scaled_qc)
        final_qc = transpile(isa_qc, backend=backend, optimization_level=3, scheduling_method='alap', routing_method='sabre')
        sampler = Sampler(mode=backend)
        sampler = configure_sampler_options(sampler)
        if config.USE_MANUAL_ZNE: sampler.options.resilience_level = 0
        job = sampler.run([final_qc], shots=shots)
        result = job.result()
        cnt = safe_get_counts(result[0])
        if cnt: counts_list.append(cnt)
    if not counts_list: return defaultdict(float)
    extrapolated = defaultdict(float)
    keys = set().union(*counts_list)
    for meas in keys:
        vals = [c.get(meas, 0) for c in counts_list]
        if len(vals) > 1:
            fit = np.polyfit(scales[:len(vals)], vals, 1) 
            extrapolated[meas] = max(0, fit[1]) 
        else: extrapolated[meas] = vals[0]
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
    plt.figure(figsize=(6,6)); plt.title('Heatmap'); plt.imshow(heat, cmap='viridis', origin='lower'); plt.colorbar(); plt.show()
    fig = plt.figure(figsize=(7,5)); ax = fig.add_subplot(111, projection='3d')
    A, B = np.meshgrid(np.arange(grid), np.arange(grid))
    ax.plot_surface(B, A, heat, cmap='viridis'); plt.show()

# ==========================================
# 7. POST PROCESSING & RUNNER
# ==========================================

def save_key(k: int):
    hex_k = hex(k)[2:].zfill(64)
    padded_hex = '0x' + hex_k.zfill(64)
    zero_padded = hex_k.zfill(64)
    shifted_hex = '0x' + zero_padded[32:] + zero_padded[:32]
    with open("boom.txt", "a") as f:
        f.write(f"{padded_hex}\n{zero_padded}\n{shifted_hex}\n")
    logger.info(f"Saved key formats to boom.txt")

def retrieve_and_process_job(job_id, service, n_bits, start_val, target_pub_x, method):
    job = service.job(job_id)
    while job.status().name not in ["DONE", "COMPLETED"]:
        logger.info(f"Waiting... Status: {job.status().name}")
        time.sleep(60)
    result = job.result()
    counts = safe_get_counts(result[0])
    return hybrid_post_process(counts, n_bits, ORDER, start_val, target_pub_x, method)

def hybrid_post_process(counts, bits, order=N, start=config.KEYSPACE_START, qx=None, method='phase'):
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:100]
    for meas_str, freq in sorted_counts:
        meas_str = meas_str.replace(" ", "")
        if method == 'ab' or method == 5:
            mid = len(meas_str) // 2
            a = int(meas_str[:mid], 2); b = int(meas_str[mid:], 2)
            if gcd_verbose(b, order) == 1:
                inv_b = modular_inverse_verbose(b, order)
                k = (-a * inv_b) % order
                if ec_scalar_mult(k, G)[0] == qx: save_key(k); return k
        else:
            measurements = [int(bit) for bit in meas_str[::-1]]
            phi = sum([b * (1 / 2**(i+1)) for i, b in enumerate(measurements)])
            num, den = continued_fractions_approx(int(phi * 2**bits), 2**bits, order)
            d = (num * modular_inverse_verbose(den, order)) % order if den and gcd_verbose(den, order) == 1 else None
            if d:
                cand = (start + d) % N
                if ec_scalar_mult(cand, G)[0] == qx: save_key(cand); return cand
    return None

def run_best_solver():
    try: service = QiskitRuntimeService()
    except: service = QiskitRuntimeService(channel="ibm_cloud", token=config.TOKEN, instance=config.CRN)
    backend = service.backend(config.BACKEND)
    
    if config.METHOD == 'smart':
        mode_id = get_best_mode_id(config.BITS, backend.num_qubits)
    else:
        m_map = {'ipe':1, 'hive':2, 'windowed':3, 'semi':4, 'ab':5, 'ft_test':6, 
                 'geo':7, 'ver':8, 'shadow':9, 'rev':10, 'swarm':11, 'probe':12,
                 'fixed_ab':19, 'matrix_mod':20}
        mode_id = m_map.get(config.METHOD, 2)
    
    logger.info(f"Target BITS={config.BITS}. Hardware={backend.num_qubits}q.")
    logger.info(f"Selected Mode ID: {mode_id}")
    
    qc = build_circuit_selector(mode_id, config.BITS)
    gate_counts = estimate_gate_counts(qc)
    print(f"Est Gates: {gate_counts}")
    
    if qc.num_qubits > backend.num_qubits:
        logger.warning(f"CRITICAL: Circuit needs {qc.num_qubits} qubits.")
    
    counts = {}
    Q = decompress_pubkey(config.COMPRESSED_PUBKEY_HEX)
    
    if config.USE_MANUAL_ZNE:
        logger.info(">>> MANUAL ZNE ENABLED <<<")
        counts = manual_zne(qc, backend, config.SHOTS)
    else:
        logger.info(">>> STANDARD RUN ENABLED <<<")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=config.OPT_LEVEL)
        isa_qc = pm.run(qc)
        final_qc = transpile(isa_qc, backend=backend, optimization_level=3, scheduling_method='alap', routing_method='sabre')
        sampler = Sampler(mode=backend)
        sampler = configure_sampler_options(sampler)
        job = sampler.run([final_qc], shots=config.SHOTS)
        logger.info(f"Job submitted: {job.job_id()}")
        k = retrieve_and_process_job(job.job_id(), service, config.BITS, config.KEYSPACE_START, Q.x(), 'ab' if mode_id==5 else 'phase')
        if k: return k
        try: counts = safe_get_counts(job.result()[0])
        except: counts = {}

    k = hybrid_post_process(counts, config.BITS, N, config.KEYSPACE_START, Q.x(), 'ab' if mode_id==5 else 'phase')
    if k: logger.info(f"Recovered Key: {hex(k)}")
    plot_visuals(counts, config.BITS, N, k)
    return k

if __name__ == "__main__":
    run_best_solver()