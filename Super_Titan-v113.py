import math
import sys
import numpy as np
from fractions import Fraction
from typing import Tuple, Optional, Dict, List, Union

# ==============================================================================
# 0. ROBUST IMPORT & SETUP (HARDWARE EXCLUSIVE)
# ==============================================================================
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import QFT
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError as e:
    print("CRITICAL ERROR: Qiskit Library Version Mismatch.")
    sys.exit(1)

def save_boom(filename, content):
    try:
        with open(filename, "a") as f: f.write(str(content) + "\n")
        print(f"[Disk] >> APPENDED TO {filename}")
    except: pass

# ==============================================================================
# 1. MASTER CONFIGURATION
# ==============================================================================

API_TOKEN = "API_TOKEN"
BACKEND_NAME = "ibm_marrakech" 
SHOTS = 8192 
OPTIMIZATION_LEVEL = 3

# SECP256K1 PARAMETERS
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP_A = 0
SECP_B = 7

# PRESETS
PRESET_135 = {
    "bits": 135,
    "start": "4000000000000000000000000000000000",
    "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
}
PRESET_17 = {
    "bits": 17,
    "start": "10000",
    "pub": "033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3"
}

# ==============================================================================
# 2. CLASSICAL MATH ENGINE
# ==============================================================================

def gcd_verbose(a: int, b: int) -> int:
    while b != 0: a, b = b, a % b
    return a

def extended_euclidean(a: int, b: int) -> Tuple[int,int,int]:
    if b == 0: return (a, 1, 0)
    g, x1, y1 = extended_euclidean(b, a % b)
    x, y = y1, x1 - (a // b) * y1
    return g, x, y

def modular_inverse_verbose(a: int, m: int) -> int:
    try: return pow(a, -1, m)
    except ValueError:
        g, x, y = extended_euclidean(a, m)
        if g != 1: raise ValueError("No inverse")
        return x % m

def tonelli_shanks_sqrt(n: int, p: int) -> int:
    if pow(n, (p - 1) // 2, p) != 1: return 0
    if p % 4 == 3: return pow(n, (p + 1) // 4, p)
    s, e = p - 1, 0
    while s % 2 == 0: s //= 2; e += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != -1: z += 1
    x = pow(n, (s + 1) // 2, p)
    b, g, r = pow(n, s, p), pow(z, s, p), e
    while True:
        t, m = b, 0
        for m in range(r):
            if t == 1: break
            t = pow(t, 2, p)
        if m == 0: return x
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p; x = (x * gs) % p; b = (b * g) % p; r = m

def ec_point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1; x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % SECP_P == 0: return None
        lam = ((3 * x1**2 + SECP_A) * modular_inverse_verbose(2 * y1, SECP_P)) % SECP_P
    else:
        lam = ((y2 - y1) * modular_inverse_verbose(x2 - x1, SECP_P)) % SECP_P
    x3 = (lam**2 - x1 - x2) % SECP_P
    y3 = (lam * (x1 - x3) - y1) % SECP_P
    return (x3, y3)

def ec_scalar_multiply(k, point):
    if k == 0 or point is None: return None
    res, addend = None, point
    for bit in reversed(bin(k)[2:]):
        res = ec_point_add(res, res)
        if bit == '1': res = ec_point_add(res, addend)
    return res

def ec_point_negate(point):
    if point is None: return None
    return (point[0], (-point[1]) % SECP_P)

def ec_point_subtract(p1, p2):
    return ec_point_add(p1, ec_point_negate(p2))

def decompress_pubkey(hex_key):
    hex_key = hex_key.replace("0x", "").lower().strip()
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    y_sq = (pow(x, 3, SECP_P) + SECP_B) % SECP_P
    y = tonelli_shanks_sqrt(y_sq, SECP_P)
    if prefix == 2 and y % 2 != 0: y = SECP_P - y
    if prefix == 3 and y % 2 == 0: y = SECP_P - y
    return (x, y)

def continued_fractions_approx(num, den, max_den):
    g = gcd_verbose(num, den) 
    if g > 1: num //= g; den //= g
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

# ==============================================================================
# 3. QUANTUM KERNEL & ORACLES (ALL DEFINITIONS)
# ==============================================================================

def prepare_verified_ancilla(qc: QuantumCircuit, qubit, initial_state=0):
    qc.reset(qubit)
    if initial_state == 1: qc.x(qubit)

def qft_gate(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / (2 ** (k - j)), k, j)
    return qc.to_gate(label="QFT")

def qft_reg(qc: QuantumCircuit, qreg: QuantumRegister):
    qc.append(qft_gate(len(qreg)), qreg)

def iqft_gate(n):
    return qft_gate(n).inverse()

def iqft_reg(qc: QuantumCircuit, qreg: QuantumRegister):
    qc.append(iqft_gate(len(qreg)), qreg)

# --- DRAPER CORE ---
def draper_add_const(qc: QuantumCircuit, ctrl: QuantumRegister, target: QuantumRegister, value: int):
    n = len(target)
    for i in range(n):
        angle = 2 * math.pi * value / (2 ** (i + 1))
        qc.cp(angle, ctrl, target[i])

def draper_sub_const_uncontrolled(qc: QuantumCircuit, target: QuantumRegister, value: int):
    n = len(target)
    for i in range(n):
        angle = -2 * math.pi * value / (2 ** (i + 1))
        qc.p(angle, target[i])

# --- ORACLE VARIATIONS (SELECTION TARGETS) ---

def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl_qubit, target_reg, dx, dy):
    """Standard Serial Oracle (Bit-by-Bit)"""
    n = len(target_reg)
    qft_reg(qc, target_reg)
    for i in range(n):
        angle = 2 * math.pi * dx / (2 ** (i + 1))
        qc.cp(angle, ctrl_qubit, target_reg[i])
    iqft_reg(qc, target_reg)

def draper_adder_oracle_2d(qc: QuantumCircuit, ctrl: QuantumRegister, target: QuantumRegister, dx: int, dy: int):
    """2D Oracle (Simultaneous X/Y) - The 'Mini-Mod' for Parallel Modes"""
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle_x = 2 * math.pi * dx / (2 ** (i + 1))
        qc.cp(angle_x, ctrl, target[i])
        angle_y = 2 * math.pi * dy / (2 ** (i + 1))
        qc.cp(angle_y, ctrl, target[i])
    iqft_reg(qc, target)

def draper_adder_oracle_scalar(qc: QuantumCircuit, ctrl: QuantumRegister, target: QuantumRegister, scalar: int):
    """Scalar Oracle - The 'Mini-Mod' for Shor Mode 14"""
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle = 2 * math.pi * scalar / (2 ** (i + 1))
        qc.cp(angle, ctrl, target[i])
    iqft_reg(qc, target)

def eigenvalue_phase_oracle(qc: QuantumCircuit, ctrl_qubit, target_qubit, scalar_val, n_mod):
    """Legacy Shor Oracle"""
    theta = 2 * math.pi * scalar_val / (2**n_mod)
    qc.cp(theta, ctrl_qubit, target_qubit)

# --- FT ADDER ---
def ft_draper_modular_adder(qc: QuantumCircuit, value: int,
                            target_reg: QuantumRegister, modulus: int,
                            ancilla: QuantumRegister, temp_reg: QuantumRegister):
    n = len(target_reg)
    prepare_verified_ancilla(qc, temp_reg[0], 0)
    qft_reg(qc, target_reg)
    draper_add_const(qc, ancilla, target_reg, value)
    iqft_reg(qc, target_reg)
    qft_reg(qc, target_reg)
    draper_sub_const_uncontrolled(qc, target_reg, modulus)
    iqft_reg(qc, target_reg)
    qc.cx(target_reg[n-1], temp_reg[0]) 
    qft_reg(qc, target_reg)
    draper_add_const(qc, temp_reg[0], target_reg, modulus)
    iqft_reg(qc, target_reg)
    prepare_verified_ancilla(qc, temp_reg[0], 0)

# ==============================================================================
# 4. EXPERT SELECTION UTILITIES (v113)
# ==============================================================================

def select_oracle_strategy(mode_num: int) -> str:
    """
    Expert Selection Menu - Determines the arithmetic strategy based on the Mode.
    """
    # Group A: Parallel/Shadow Modes (Can use 2D Oracle)
    if mode_num in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13]:
        print("\n[?] Select Arithmetic Strategy for Parallel/Shadow Mode:")
        print("    1. Standard Serial (Bit-by-Bit Logic)")
        print("    2. 2D Group Action (Simultaneous X/Y Oracle) [Mini-Mod]")
        while True:
            c = input("    >>> ").strip()
            if c == "1": return "SERIAL"
            if c == "2": return "2D"
            print("    Invalid. Enter 1 or 2.")
            
    # Group B: Shor Mode (Can use Scalar Oracle)
    elif mode_num == 14:
        print("\n[?] Select Shor Algorithm Strategy:")
        print("    1. Modified Shor (Eigenvalue Phase Oracle)")
        print("    2. Pure Shor a&b (Scalar Oracle) [Mini-Mod]")
        while True:
            c = input("    >>> ").strip()
            if c == "1": return "SHOR_MOD"
            if c == "2": return "SHOR_PURE"
            print("    Invalid. Enter 1 or 2.")
            
    # Default for others
    return "DEFAULT"

# ==============================================================================
# 5. CIRCUIT BUILDERS (STRATEGY-AWARE & FULLY EXPANDED)
# ==============================================================================

class GeometricIPE:
    def __init__(self, n_bits): self.n = n_bits
    def _oracle_geometric_phase(self, qc, ctrl, state_reg, point_val):
        if point_val is None: return
        vx, vy = point_val
        for i in range(self.n):
            angle_x = 2 * math.pi * vx / (2**(i+1))
            qc.cp(angle_x, ctrl, state_reg[i])

# --- MODE 0: PROBE ---
def build_mode_0_hardware_probe(n_bits: int, g_point: int, delta_point: Tuple[int,int]) -> QuantumCircuit:
    print(f"[i] Building Mode 0 (Probe - 3 Qubits)...")
    reg_ctrl = QuantumRegister(1, 'ctrl')
    reg_state = QuantumRegister(1, 'state')
    reg_flag = QuantumRegister(1, 'flag')
    creg = ClassicalRegister(n_bits, 'meas')
    creg_flag = ClassicalRegister(n_bits, 'flag_meas')
    qc = QuantumCircuit(reg_ctrl, reg_state, reg_flag, creg, creg_flag)
    qc.x(reg_state[0]) 
    for k in range(n_bits):
        qc.reset(reg_ctrl); qc.reset(reg_flag); qc.h(reg_ctrl)
        qc.cz(reg_ctrl[0], reg_state[0])
        qc.cx(reg_ctrl[0], reg_flag[0])
        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((creg[n_bits-1-m], 1)): qc.p(angle, reg_ctrl[0])
        qc.h(reg_ctrl)
        qc.measure(reg_ctrl[0], creg[n_bits-1-k])
        qc.measure(reg_flag[0], creg_flag[n_bits-1-k])
    return qc

# --- MODE 1: PHANTOM PARALLEL ---
def build_mode_1_phantom_parallel(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 1 [Strategy: 2D Group Action]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (g_point * power + dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 1 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 2: SHOR PARALLEL ---
def build_mode_2_shor_parallel(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 2 [Strategy: 2D Group Action]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (g_point * power + dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 2 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 3: GHZ PARALLEL ---
def build_mode_3_ghz_parallel(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 3 [Strategy: 2D GHZ]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.h(reg_count[0])
        for i in range(n_bits-1): qc.cx(reg_count[i], reg_count[i+1])
        qc.x(reg_state[0])
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 3 [Strategy: Serial GHZ]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c) 
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 4: VERIFIED PARALLEL ---
def build_mode_4_verified_parallel(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 4 [Strategy: 2D Group Action]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (g_point * power + dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 4 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 5: HIVE EDITION ---
def build_mode_5_hive_edition(n_bits: int, g_point: int, delta_point: Tuple[int,int], available_qubits: int) -> QuantumCircuit:
    num_workers = available_qubits // (n_bits + 2)
    if num_workers < 1: num_workers = 1
    print(f"[i] Building Mode 5 (Hive: {num_workers} Workers)...")
    regs = []
    for w in range(num_workers):
        regs.append(QuantumRegister(1, f'w{w}_c'))
        regs.append(QuantumRegister(n_bits, f'w{w}_s'))
        regs.append(ClassicalRegister(n_bits, f'w{w}_m'))
    qc = QuantumCircuit(*regs)
    powers = []
    curr = delta_point
    for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
    for w in range(num_workers): qc.x(qc.qubits[w*(n_bits+1) + 1])
    for k in range(n_bits):
        for w in range(num_workers):
            ctrl = qc.qubits[w*(n_bits+1)]
            target_reg = qc.qubits[w*(n_bits+1)+1 : w*(n_bits+1)+1+n_bits]
            qc.reset(ctrl); qc.h(ctrl)
            if powers[k]: draper_adder_oracle_1d_serial(qc, ctrl, target_reg, powers[k][0], powers[k][1])
            qc.h(ctrl)
            qc.measure(ctrl, qc.cregs[w][k])
    return qc

# --- MODE 6: EXTRA SHADOW ---
def build_mode_6_extra_shadow(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 6 [Strategy: 2D Group Action]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (g_point * power + dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 6 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 7: ADVANCED QPE ---
def build_advanced_QPE_MOD_7(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 7 [Strategy: 2D QPE]...")
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_state = QuantumRegister(n_bits, 'state')
        creg_phase = ClassicalRegister(n_bits, 'phase_bits')
        creg_flag = ClassicalRegister(n_bits, 'flag_bits')
        qc = QuantumCircuit(reg_ctrl, reg_state, creg_phase, creg_flag)
        qc.x(reg_state[0])
        dx, dy = delta_point
        for k in range(n_bits):
            qc.reset(reg_ctrl); qc.h(reg_ctrl)
            power = 2 ** (n_bits - 1 - k)
            sx = (dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_state, sx, sy)
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[n_bits-1-m], 1)): qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[n_bits-1-k])
        return qc
    else:
        print(f"[i] Building Mode 7 [Strategy: Standard Feedback]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 8: FULL QUANTUM ---
def build_full_quantum_MOD_8(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 8 [Strategy: 2D Group Action]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(n_bits, 'state')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, creg)
        qc.x(reg_state[0])
        qft_reg(qc, reg_count)
        dx, dy = delta_point
        for i in range(n_bits):
            power = 2 ** i
            sx = (g_point * power + dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_count[i], reg_state, sx, sy)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc
    else:
        print(f"[i] Building Mode 8 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 9: SEMICLASSICAL ---
def build_semiclassical_ecdlp_MOD_9(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 9 [Strategy: 2D Semiclassical]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        dx, dy = delta_point
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            power = 2 ** k
            sx = (dx * power) % SECP_N 
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, sx, sy)
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
        return qc
    else:
        print(f"[i] Building Mode 9 [Strategy: Standard Serial]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 10: VERIFIED SHADOW ---
def build_mode_10_verified_shadow(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 10 [Strategy: 2D Verified]...")
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_state = QuantumRegister(n_bits, 'state')
        reg_flag = QuantumRegister(1, 'flag')
        creg_phase = ClassicalRegister(n_bits, 'phase_bits')
        creg_flag = ClassicalRegister(n_bits, 'flag_meas')
        qc = QuantumCircuit(reg_ctrl, reg_state, reg_flag, creg_phase, creg_flag)
        qc.x(reg_state[0])
        dx, dy = delta_point
        for k in range(n_bits):
            qc.reset(reg_ctrl); qc.reset(reg_flag); qc.h(reg_ctrl)
            power = 2 ** (n_bits - 1 - k)
            sx = (dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_state, sx, sy)
            qc.cx(reg_ctrl[0], reg_flag[0])
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[n_bits-1-m], 1)): qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[n_bits-1-k])
            qc.measure(reg_flag[0], creg_flag[n_bits-1-k])
        return qc
    else:
        print(f"[i] Building Mode 10 [Strategy: Serial Verified]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        qr_f = QuantumRegister(1, "flag") 
        cr = ClassicalRegister(n_bits, "meas")
        cr_f = ClassicalRegister(n_bits, "meas_f")
        qc = QuantumCircuit(qr_c, qr_s, qr_f, cr, cr_f)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.reset(qr_f); qc.h(qr_c)
            qc.cx(qr_c[0], qr_f[0])
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr[k])
            qc.measure(qr_f[0], cr_f[k])
        return qc

# --- MODE 11: VERIFIED ADVANCED ---
def build_mode_11_verified_advanced(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 11 [Strategy: 2D Verified - Dual Flags]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        qr_f1 = QuantumRegister(1, "flag_init")
        qr_f2 = QuantumRegister(1, "flag_op")
        cr_meas = ClassicalRegister(n_bits, "meas")
        cr_f1 = ClassicalRegister(n_bits, "f1")
        cr_f2 = ClassicalRegister(n_bits, "f2")
        qc = QuantumCircuit(qr_c, qr_s, qr_f1, qr_f2, cr_meas, cr_f1, cr_f2)
        qc.x(qr_s[0])
        dx, dy = delta_point
        for k in range(n_bits):
            qc.reset(qr_c); qc.reset(qr_f1); qc.reset(qr_f2)
            qc.h(qr_c)
            qc.cx(qr_c[0], qr_f1[0])
            power = 2 ** k
            sx = (dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, sx, sy)
            qc.cx(qr_c[0], qr_f2[0])
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr_meas[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr_meas[k])
            qc.measure(qr_f1[0], cr_f1[k])
            qc.measure(qr_f2[0], cr_f2[k])
        return qc
    else:
        print(f"[i] Building Mode 11 [Strategy: Standard Serial - Dual Flags]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        qr_f1 = QuantumRegister(1, "flag_init")
        qr_f2 = QuantumRegister(1, "flag_op")
        cr_meas = ClassicalRegister(n_bits, "meas")
        cr_f1 = ClassicalRegister(n_bits, "f1")
        cr_f2 = ClassicalRegister(n_bits, "f2")
        qc = QuantumCircuit(qr_c, qr_s, qr_f1, qr_f2, cr_meas, cr_f1, cr_f2)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits):
            powers.append(curr)
            curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.reset(qr_f1); qc.reset(qr_f2)
            qc.h(qr_c)
            qc.cx(qr_c[0], qr_f1[0])
            pt = powers[k]
            if pt is not None:
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, pt[0], pt[1])
            qc.cx(qr_c[0], qr_f2[0])
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((cr_meas[m], 1)): qc.p(angle, qr_c[0])
            qc.h(qr_c)
            qc.measure(qr_c[0], cr_meas[k])
            qc.measure(qr_f1[0], cr_f1[k])
            qc.measure(qr_f2[0], cr_f2[k])
        return qc

# --- MODE 12: HEAVY DRAPER ---
def build_mode_12_heavy_draper(n_bits: int, g_point: int, delta_point: Tuple[int,int]) -> QuantumCircuit:
    print(f"[i] Building Mode 12 (Heavy Draper)...")
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(n_bits, "state")
    qr_anc = QuantumRegister(1, "anc")
    qr_tmp = QuantumRegister(1, "tmp")
    cr = ClassicalRegister(n_bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, qr_anc, qr_tmp, cr)
    qc.x(qr_s[0])
    powers = []
    curr = delta_point
    for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
    for k in range(n_bits):
        qc.reset(qr_c); qc.h(qr_c)
        if powers[k]: 
            ft_draper_modular_adder(qc, powers[k][0], qr_s, SECP_N, qr_anc, qr_tmp)
        for m in range(k):
            angle = -math.pi / (2**(k-m))
            with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    return qc

# --- MODE 13: COMPRESSED SHADOW ---
def build_mode_13_compressed_shadow(n_bits: int, g_point: int, delta_point: Tuple[int,int], strategy="SERIAL") -> QuantumCircuit:
    if strategy == "2D":
        print(f"[i] Building Mode 13 [Strategy: 2D Compressed]...")
        reg_ctrl = QuantumRegister(1, 'ctrl')
        reg_x = QuantumRegister(n_bits, 'x_coord')
        reg_sign = QuantumRegister(1, 'sign')
        creg_phase = ClassicalRegister(n_bits, 'phase_bits')
        qc = QuantumCircuit(reg_ctrl, reg_x, reg_sign, creg_phase)
        qc.x(reg_x[0])
        dx, dy = delta_point
        for k in range(n_bits):
            qc.reset(reg_ctrl); qc.h(reg_ctrl)
            power = 2 ** (n_bits - 1 - k)
            sx = (dx * power) % SECP_N
            sy = (dy * power) % SECP_N
            draper_adder_oracle_2d(qc, reg_ctrl[0], reg_x, sx, sy)
            qc.cx(reg_ctrl[0], reg_sign[0]) 
            for m in range(k):
                angle = -math.pi / (2 ** (k - m))
                with qc.if_test((creg_phase[n_bits-1-m], 1)): qc.p(angle, reg_ctrl[0])
            qc.h(reg_ctrl)
            qc.measure(reg_ctrl[0], creg_phase[n_bits-1-k])
        return qc
    else:
        print(f"[i] Building Mode 13 [Strategy: Serial Compressed]...")
        qr_c = QuantumRegister(1, "ctrl")
        qr_s = QuantumRegister(n_bits, "state")
        cr = ClassicalRegister(n_bits, "meas")
        qc = QuantumCircuit(qr_c, qr_s, cr)
        qc.x(qr_s[0])
        powers = []
        curr = delta_point
        for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
        for k in range(n_bits):
            qc.reset(qr_c); qc.h(qr_c)
            if powers[k]: draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, powers[k][0], powers[k][1])
            qc.h(qr_c); qc.measure(qr_c[0], cr[k])
        return qc

# --- MODE 14: SHOR (SPLIT BEHAVIOR) ---
def build_mode_14_shor_logic(n_bits: int, g_point: int, delta_point: Tuple[int, int], strategy="SHOR_MOD") -> QuantumCircuit:
    if strategy == "SHOR_PURE":
        print(f"[i] Building Mode 14 [Strategy: Pure Shor a & b - Scalar Oracle]...")
        reg_a = QuantumRegister(n_bits, 'a')
        reg_b = QuantumRegister(n_bits, 'b')
        reg_work = QuantumRegister(n_bits, 'work')
        creg = ClassicalRegister(2 * n_bits, 'meas')
        qc = QuantumCircuit(reg_a, reg_b, reg_work, creg)
        qc.h(reg_a)
        qc.h(reg_b)
        qc.x(reg_work[0])
        for i in range(n_bits):
            val_a = (g_point * (2**i)) % SECP_N
            draper_adder_oracle_scalar(qc, reg_a[i], reg_work, val_a)
        target_scalar = delta_point[0] 
        for i in range(n_bits):
            val_b = (target_scalar * (2**i)) % SECP_N
            draper_adder_oracle_scalar(qc, reg_b[i], reg_work, val_b)
        iqft_reg(qc, reg_a)
        iqft_reg(qc, reg_b)
        qc.measure(reg_a, creg[0:n_bits])
        qc.measure(reg_b, creg[n_bits:2*n_bits])
        return qc
    else:
        print(f"[i] Building Mode 14 [Strategy: Modified Shor - Phase Oracle]...")
        reg_count = QuantumRegister(n_bits, 'count')
        reg_state = QuantumRegister(1, 'state') 
        reg_temp = QuantumRegister(1, 'temp_ph')
        creg = ClassicalRegister(n_bits, 'meas')
        qc = QuantumCircuit(reg_count, reg_state, reg_temp, creg)
        qc.x(reg_state[0]) 
        qft_reg(qc, reg_count)
        dx = delta_point[0]
        for i in range(n_bits):
            power = 2 ** i
            scalar_val = (g_point * power) + (dx * power)
            eigenvalue_phase_oracle(qc, reg_count[i], reg_state[0], scalar_val, n_bits)
        iqft_reg(qc, reg_count)
        qc.measure(reg_count, creg)
        return qc

# --- MODE 15: GEOMETRIC IPE ---
def build_mode_15_ipe(n_bits, delta_point):
    print(f"[Build] Mode 15: Geometric IPE ({n_bits} bits)...")
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(n_bits, "state")
    cr = ClassicalRegister(n_bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, cr)
    qc.append(QFT(n_bits, do_swaps=False), qr_s)
    powers = []
    curr = delta_point
    for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
    engine = GeometricIPE(n_bits)
    for k in range(n_bits):
        qc.reset(qr_c); qc.h(qr_c)
        engine._oracle_geometric_phase(qc, qr_c[0], qr_s, powers[k])
        for m in range(k):
            angle = -math.pi / (2**(k-m))
            with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    return qc

# --- MODE 16: WINDOWED IPE ---
def build_mode_16_windowed(n_bits, delta_point):
    print(f"[Build] Mode 16: Windowed IPE...")
    window_size = 4
    qr_c = QuantumRegister(window_size, "ctrl")
    qr_s = QuantumRegister(n_bits, "state")
    cr = ClassicalRegister(n_bits, "meas")
    qc = QuantumCircuit(qr_c, qr_s, cr)
    qft_reg(qc, qr_s)
    powers = []
    curr = delta_point
    for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
    engine = GeometricIPE(n_bits)
    for i in range(0, n_bits, window_size):
        chunk = min(window_size, n_bits - i)
        qc.reset(qr_c)
        qc.h(qr_c[:chunk])
        for j in range(chunk):
            engine._oracle_geometric_phase(qc, qr_c[j], qr_s, powers[i+j])
        qc.append(QFT(chunk, do_swaps=False).inverse(), qr_c[:chunk])
        for j in range(chunk): qc.measure(qr_c[j], cr[i+j])
    return qc

# --- MODE 17: HIVE SWARM ---
def build_mode_17_hive(n_bits, delta_point, total_qubits):
    print(f"[Build] Mode 17: Hive Swarm...")
    num_workers = total_qubits // (n_bits + 1)
    if num_workers < 2: return build_mode_15_ipe(n_bits, delta_point)
    regs = []
    for w in range(num_workers):
        regs.append(QuantumRegister(1, f"c{w}"))
        regs.append(QuantumRegister(n_bits, f"s{w}"))
        regs.append(ClassicalRegister(n_bits, f"m{w}"))
    qc = QuantumCircuit(*regs)
    powers = []
    curr = delta_point
    for _ in range(n_bits): powers.append(curr); curr = ec_point_add(curr, curr)
    engine = GeometricIPE(n_bits)
    for k in range(n_bits):
        for w in range(num_workers):
            ctrl = qc.qubits[w*(n_bits+1)]
            qc.reset(ctrl); qc.h(ctrl)
        for w in range(num_workers):
            ctrl = qc.qubits[w*(n_bits+1)]
            state = qc.qubits[w*(n_bits+1)+1 : w*(n_bits+1)+1+n_bits]
            engine._oracle_geometric_phase(qc, ctrl, state, powers[k])
        for w in range(num_workers):
            ctrl = qc.qubits[w*(n_bits+1)]
            meas = qc.cregs[w]
            for m in range(k):
                angle = -math.pi / (2**(k-m))
                with qc.if_test((meas[m], 1)): qc.p(angle, ctrl)
            qc.h(ctrl)
            qc.measure(ctrl, meas[k])
    return qc

# --- MODE 18: EXPLICIT ---
def build_mode_18_explicit(n_bits: int, delta_point: Tuple[int,int]) -> QuantumCircuit:
    print(f"[Build] Mode 18: Explicit Logic (FT Adder Topology)...")
    run_len = min(n_bits, 8) 
    print(f"    -> Scaled internal logic to {run_len} bits for hardware safety.")
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(run_len, "state")
    qr_anc = QuantumRegister(1, "anc")
    qr_tmp = QuantumRegister(1, "tmp")
    cr = ClassicalRegister(run_len, "meas")
    qc = QuantumCircuit(qr_c, qr_s, qr_anc, qr_tmp, cr)
    qc.x(qr_s[0])
    scalar_val = delta_point[0]
    for k in range(run_len):
        qc.reset(qr_c); qc.h(qr_c)
        val_shifted = (scalar_val * (2**k)) 
        val_shifted = val_shifted % (2**run_len)
        ft_draper_modular_adder(qc, val_shifted, qr_s, (2**run_len) - 1, qr_anc, qr_tmp)
        for m in range(k):
            angle = -math.pi / (2**(k-m))
            with qc.if_test((cr[m], 1)): qc.p(angle, qr_c[0])
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    return qc

# ==============================================================================
# 6. UNIVERSAL POST-PROCESSING
# ==============================================================================

def _universal_solver(counts: Dict[str, int], n_bits: int, start, qx, mode_name: str, search_depth: int = 8192):
    print(f"    Running {mode_name} Logic (Depth {search_depth})...")

    # ==========================================================================
    # 1. INTERNAL UTILITIES
    # ==========================================================================
    
    def lsb_msb_variants(bitstr):
        """Converts bitstring to int (standard) and int (reversed)."""
        variants = set()
        try: variants.add(int(bitstr, 2))
        except: pass
        try: variants.add(int(bitstr[::-1], 2))
        except: pass
        return variants

    def check_ec_match(candidate):
        """Checks if candidate*G == Qx. Returns True if match."""
        if candidate <= 0 or candidate >= SECP_N: return False
        if gcd_verbose(candidate, SECP_N) > 1: return False
        try:
            pub = ec_scalar_multiply(candidate, (SECP_GX, SECP_GY))
            return pub is not None and pub[0] == qx
        except: return False

    # ==========================================================================
    # 2. DATA CLEANING & SORTING (Moved Up for Raw Check Access)
    # ==========================================================================
    clean_counts = {}
    if counts:
        for bitstr, freq in counts.items():
            raw_clean = bitstr.replace(" ", "")
            # For modes with ancilla/flags, we usually want the last N bits
            if any(m in mode_name for m in ["Mode 4", "Mode 10", "Mode 11", "Mode 12"]):
                target_bits = raw_clean[-n_bits:] if len(raw_clean) >= n_bits else raw_clean
            else:
                target_bits = raw_clean
            clean_counts[target_bits] = clean_counts.get(target_bits, 0) + freq

    # Sort by frequency and apply search_depth immediately
    sorted_valid = sorted(clean_counts.items(), key=lambda x: x[1], reverse=True)[:search_depth]
    print(f"[i] Scanning {len(sorted_valid)} unique candidates...")

    # ==========================================================================
    # 3. RAW SAFETY NET (Iterates Search Depth + Reverse Check)
    # ==========================================================================
    # Now checks ALL top candidates, not just the #1 result.
    for bitstr, freq in sorted_valid:
        # Get Standard and Reversed integer values
        raw_candidates = lsb_msb_variants(bitstr)
        
        for val_raw in raw_candidates:
            # We skip the heavy logging for every fail, just log success
            if check_ec_match(val_raw):
                print(f"\n[!!!] BOOM: RAW MATCH (Depth Scan): {hex(val_raw)}")
                save_boom("FOUND_KEY.txt", hex(val_raw))
                save_boom("boom9_raw.txt", hex(val_raw))
                return

    # ==========================================================================
    # 4. PRE-COMPUTATION (Mathematical Deltas)
    # ==========================================================================
    # Pre-calculate integers and deltas to avoid re-mathing in every loop
    precomputed_data = []
    for bitstr, freq in sorted_valid:
        candidate_ints = lsb_msb_variants(bitstr)
        deltas = set()
        for val in candidate_ints:
            if val == 0: continue
            if gcd_verbose(val, SECP_N) > 1: continue
            
            # A. Direct Phase Scaling: delta = round(val/2^n * N)
            try:
                phase = val / (2 ** n_bits)
                d_dir = int(round(phase * SECP_N))
                deltas.add((d_dir, "Direct"))
                # Neighbors (Basic +/- checks)
                for off in [-2, -1, 1, 2]:
                    deltas.add((d_dir + off, "Neighbor"))
            except: pass
            
            # B. Continued Fractions: delta = numerator of approx(val/2^n)
            try:
                num, den = continued_fractions_approx(val, 2**n_bits, SECP_N)
                deltas.add((num, "CF"))
            except: pass
            
        precomputed_data.append((bitstr, freq, candidate_ints, deltas))

    # ==========================================================================
    # 5. SOLVER LOOP A: SPECIAL MODES & RAW OFFSETS
    # ==========================================================================
    for bitstr, freq, candidate_ints, deltas in precomputed_data:

        # --- Mode 14 Special Handling (Shor a & b) ---
        if "Mode 14" in mode_name:
            if len(bitstr) >= n_bits:
                try:
                    # Strategy 1: Split in half
                    mid = len(bitstr) // 2
                    part_b = int(bitstr[:mid], 2)
                    part_a = int(bitstr[mid:], 2)
                    if part_b != 0:
                        inv_b = modular_inverse_verbose(part_b, SECP_N)
                        k_shor = (-part_a * inv_b) % SECP_N
                        if check_ec_match(k_shor):
                            print(f"\n[!!!] BOOM: SHOR (Mid-Split) MATCH: {hex(k_shor)}")
                            save_boom("FOUND_KEY.txt", hex(k_shor))
                            save_boom("boom_shor.txt", hex(k_shor)); return
                except: pass
                try:
                    # Strategy 2: Exact N bits if string is 2*N
                    if len(bitstr) >= 2 * n_bits:
                        part_b = int(bitstr[:n_bits], 2)
                        part_a = int(bitstr[n_bits:], 2)
                        if part_b != 0:
                            inv_b = modular_inverse_verbose(part_b, SECP_N)
                            k_shor = (-part_a * inv_b) % SECP_N
                            if check_ec_match(k_shor):
                                print(f"\n[!!!] BOOM: SHOR (2N-Split) MATCH: {hex(k_shor)}")
                                save_boom("FOUND_KEY.txt", hex(k_shor)); return
                except: pass

        # --- Raw Integer Offset Matching (k = start +/- val) ---
        for val in candidate_ints:
            # Check: val, start+val, start-val
            offsets = [val, (start + val) % SECP_N, (start - val) % SECP_N]
            for k_guess in offsets:
                if check_ec_match(k_guess):
                    print(f"\n[!!!] BOOM: RAW OFFSET MATCH: {hex(k_guess)}")
                    save_boom("FOUND_KEY.txt", hex(k_guess)); return

    # ==========================================================================
    # 6. SOLVER LOOP B: PHASE & DELTA ANALYSIS
    # ==========================================================================
    # This covers the logic: check (Start + d) and (Start - d)
    for bitstr, freq, candidate_ints, deltas in precomputed_data:
        for d, method in deltas:
            if d <= 0 or d >= SECP_N: continue
            
            fk = (start + d) % SECP_N  # Forward (Start + d)
            rk = (start - d) % SECP_N  # Reverse (Start - d)
            gk = (SECP_N - fk) % SECP_N # Global reflection
            
            for k_final in [fk, rk, gk]:
                try:
                    pub = ec_scalar_multiply(k_final, (SECP_GX, SECP_GY))
                    if pub and pub[0] == qx:
                        print("*" * 60)
                        print(f"\n[!!!] BOOM: KEY FOUND (Phase/Delta): {hex(k_final)}")
                        print("*" * 60)
                        save_boom("FOUND_KEY.txt", hex(k_final))
                        save_boom("boom1.txt", hex(k_final))
                        if method == "CF": save_boom("boom7.txt", hex(k_final))
                        if method == "Neighbor": save_boom("boom10.txt", hex(k_final))
                        if k_final == rk: save_boom("boom2.txt", hex(k_final))
                        if k_final == gk: save_boom("boom_global.txt", hex(k_final))
                        return
                except: continue

    # ==========================================================================
    # 7. SOLVER LOOP C: EXTRA NEGATE EC POINT SAFETY CHECK
    # ==========================================================================
    # Checks if k produces -Q (Negative of target), implying Key = N - k
    for bitstr, freq, candidate_ints, deltas in precomputed_data:
        for val in candidate_ints:
            if val == 0: continue
            try:
                num, den = continued_fractions_approx(val, 2**n_bits, SECP_N)
                direct_scale = int(round((val / (2**n_bits)) * SECP_N))
                
                for d in [num, direct_scale]:
                    fk = (start + d) % SECP_N
                    rk = (start - d) % SECP_N
                    gk = (SECP_N - fk) % SECP_N
                    
                    for k_final in [fk, rk, gk]:
                        try:
                            pub = ec_scalar_multiply(k_final, (SECP_GX, SECP_GY))
                            # Check if pub is the negation of target Q
                            if pub:
                                neg_pub = ec_point_negate(pub)
                                if neg_pub and neg_pub[0] == qx:
                                    k_neg_pub = (SECP_N - k_final) % SECP_N
                                    print(f"\n[!!!] BOOM: KEY FOUND (Extra Negate via EC Point): {hex(k_neg_pub)}")
                                    save_boom("FOUND_KEY.txt", hex(k_neg_pub))
                                    save_boom("boom_negate_extra.txt", hex(k_neg_pub))
                                    return
                        except: continue
            except: continue

    # ==========================================================================
    # 8. SOLVER LOOP D: EXTRA BACKUP PHASE ANALYSIS
    # ==========================================================================
    # A simplified pass strictly on standard CF/Direct deltas acting as sanity check.
    for bitstr, freq, candidate_ints, deltas in precomputed_data:
        for val in candidate_ints:
            if val == 0: continue
            try:
                num, den = continued_fractions_approx(val, 2**n_bits, SECP_N)
                direct_scale = int(round((val / (2**n_bits)) * SECP_N))
                
                for d in [num, direct_scale]:
                    fk = (start + d) % SECP_N
                    rk = (start - d) % SECP_N
                    
                    for k_final in [fk, rk]:
                        if check_ec_match(k_final):
                            print(f"\n[!!!] BOOM: EXTRA PHASE CHECK MATCH: {hex(k_final)}")
                            save_boom("FOUND_KEY.txt", hex(k_final))
                            save_boom("boom_extra_phase.txt", hex(k_final))
                            return
            except: continue

    # ==========================================================================
    # 9. SOLVER LOOP E: LEGACY "LOOP 3" (INJECTED FALLBACK)
    # ==========================================================================
    # Exact implementation of your requested "Injected Original LOOP 3"
    for bitstr, freq in sorted_valid:
        candidates = set()
        try: candidates.add(int(bitstr, 2))
        except: pass
        try: candidates.add(int(bitstr[::-1], 2))
        except: pass

        for val in candidates:
            if val == 0: continue
            try:
                num, den = continued_fractions_approx(val, 2**n_bits, SECP_N)
                direct_scale = int(round((val / (2**n_bits)) * SECP_N))
                cands_to_check = [num, direct_scale]

                for d in cands_to_check:
                    key_guess = (start + d) % SECP_N
                    try:
                        # Check Standard
                        pub = ec_scalar_multiply(key_guess, (SECP_GX, SECP_GY))
                        if pub and pub[0] == qx:
                            print(f"\n[!!!] BOOM: KEY FOUND (Legacy Phase): {hex(key_guess)}")
                            save_boom("FOUND_KEY.txt", hex(key_guess)); return

                        # Check Inverse (k_inv = N - key)
                        k_inv = (SECP_N - key_guess) % SECP_N
                        pub_inv = ec_scalar_multiply(k_inv, (SECP_GX, SECP_GY))
                        if pub_inv and pub_inv[0] == qx:
                            print(f"\n[!!!] BOOM: KEY FOUND (Legacy Inv): {hex(k_inv)}")
                            save_boom("FOUND_KEY.txt", hex(k_inv)); return
                    except: continue
            except: continue

    print(f"[x] {mode_name} Failed.")

# ==============================================================================
# 7. MAIN CONTROLLER
# ==============================================================================

def main():
    print("="*60)
    print("   THE OMEGA SOLVER v113 (SUPER-TITAN EXPERT EDITION)")
    print("="*60)
    
    print("1. 135-bit Preset | 2. 17-bit Preset | 3. Custom")
    ch = input("Select: ").strip()
    if ch == "1":
        bits = 135; start_hex = PRESET_135["start"]; pub_hex = PRESET_135["pub"]
    elif ch == "2":
        bits = 17; start_hex = PRESET_17["start"]; pub_hex = PRESET_17["pub"]
    else:
        bits = int(input("Bits: "))
        start_hex = input("Start Hex: ")
        pub_hex = input("Pub Hex: ")
    start_val = int(start_hex, 16)
    
    try:
        target_pub = decompress_pubkey(pub_hex)
        p_start = ec_scalar_multiply(start_val, (SECP_GX, SECP_GY))
        delta = ec_point_subtract(target_pub, p_start)
        if delta is None:
            print("!!! Trivial Solution !!!"); save_boom("FOUND.txt", hex(start_val)); return
        print(f"[Math] Delta Calculated: X={hex(delta[0])[:10]}...")
    except Exception as e: print(f"[Error] Math: {e}"); return

    print("\n[i] Connecting to IBM Quantum...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=API_TOKEN, overwrite=True)
        service = QiskitRuntimeService()
        backend = service.backend(BACKEND_NAME)
        print(f"    Connected: {backend.name} ({backend.num_qubits} Qubits)")
        max_q = backend.num_qubits
    except Exception as e: 
        print(f"[!] Critical Connection Error: {e}")
        sys.exit(1)

    print("\n[Modes]")
    print("   0: Probe | 1-13: Serial/2D | 14: Shor (Mod/Pure) | 15-18: IPE")
    
    try: mode = int(input("Select Mode [0-18]: ").strip())
    except: mode = 15
    
    # --- EXPERT SELECTION CALL ---
    selected_strategy = select_oracle_strategy(mode)
    
    run_bits = bits
    qc = None
    
    # BUILDERS WITH STRATEGY INJECTION
    if mode == 0: qc = build_mode_0_hardware_probe(run_bits, SECP_GX, delta)
    
    # Modes with Strategy Switching (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13)
    elif mode == 1: qc = build_mode_1_phantom_parallel(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 2: qc = build_mode_2_shor_parallel(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 3: qc = build_mode_3_ghz_parallel(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 4: qc = build_mode_4_verified_parallel(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 6: qc = build_mode_6_extra_shadow(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 7: qc = build_advanced_QPE_MOD_7(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 8: qc = build_full_quantum_MOD_8(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 9: qc = build_semiclassical_ecdlp_MOD_9(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 10: qc = build_mode_10_verified_shadow(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 11: qc = build_mode_11_verified_advanced(run_bits, SECP_GX, delta, selected_strategy)
    elif mode == 13: qc = build_mode_13_compressed_shadow(run_bits, SECP_GX, delta, selected_strategy)
    
    # Mode 14 Special Case
    elif mode == 14: qc = build_mode_14_shor_logic(run_bits, SECP_GX, delta, selected_strategy)
    
    # Standard Modes (No variants needed)
    elif mode == 5: qc = build_mode_5_hive_edition(run_bits, SECP_GX, delta, max_q)
    elif mode == 12: qc = build_mode_12_heavy_draper(run_bits, SECP_GX, delta)
    elif mode == 15: qc = build_mode_15_ipe(run_bits, delta)
    elif mode == 16: qc = build_mode_16_windowed(run_bits, delta)
    elif mode == 17: qc = build_mode_17_hive(run_bits, delta, max_q)
    elif mode == 18: qc = build_mode_18_explicit(run_bits, delta)
    else: qc = build_mode_15_ipe(run_bits, delta)
    
    print(f"\n[Exec] Running Mode {mode} [Strategy: {selected_strategy}]...")
    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=OPTIMIZATION_LEVEL, scheduling_method='alap', routing_method='sabre')
        transpiled_qc = pm.run(qc)
        
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XX"
        sampler.options.twirling.enable_measure = True
        
        job = sampler.run([transpiled_qc], shots=SHOTS)
        print(f"    Job ID: {job.job_id()}")
        
        result = job.result()
        # --- 8 Robust Extraction Attempts ---
        counts = None
        extraction_attempts = [
            lambda result: result[0].data.meas.get_counts(),
            lambda result: result[0].data.c.get_counts(),
            lambda result: result[0].data.meas_bits.get_counts(),
            lambda result: result[0].data.meas_state.get_counts(),
            lambda result: result[0].data.meas_f.get_counts(),
            lambda result: result[0].data.flag_meas.get_counts(),
            lambda result: result[0].data.flag_bits.get_counts(),
            lambda result: result[0].data.meas.get_counts(),  # again as last fallback
        ]
        for i, attempt in enumerate(extraction_attempts):
            try:
                counts = attempt(result)
                print(f"[i] Counts extraction succeeded on attempt {i+1}.")
                break
            except Exception as e:
                continue

        if not counts:
            print("[!] Could not extract counts from result object.")
            return

        print(f"[i] Extracted {len(counts)} unique bitstrings.")

        # 4. Run Solver
        _universal_solver(counts, run_bits, start_val, target_pub[0], f"Mode {mode}", search_depth=8192)
        
    except Exception as e: 
        print(f"[Error] Execution Failed: {e}")

if __name__ == "__main__":
    main()
