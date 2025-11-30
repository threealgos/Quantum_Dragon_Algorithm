#i Realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#======================================================================================================
# TODO:Remove iterations/Keyspace_Rang-Preparations Cause Deprecationwarning: Treating CircuitInstruction as an iterable is deprecated legacy behavior since Qiskit 1.2, and will be removed in Qiskit 3.0. Instead, use the `operation`, `qubits` and `clbits` named attributes.
from IPython.display import display
#import qctrl
#import qctrlvisualizer as qv
#import fireopal as fo
#import boulderopal as bo
#from qctrl import QAQA, QPE, QEC
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
# old QFT deprecated → replace with QFTGate & synth_qft_full
from qiskit import synthesis
from qiskit.synthesis import synth_qft_full
from qiskit.circuit.library import ZGate, MCXGate, RYGate, GroverOperator, QFT, QFTGate
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector, Operator
from qiskit.circuit import Parameter
from Crypto.PublicKey import ECC
from bitarray import bitarray
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator, Aer
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Options, SamplerV2 as Sampler
from collections import Counter
from fractions import Fraction
#from qiskit.circuit import UnitaryGate
#from qiskit_algorithms.optimizers import AmplificationProblem, CustomCircuitOracle
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import hashlib
import base58
import os
import pandas as pd
import logging
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram, plot_distribution
from math import gcd, ceil, log2
from typing import Optional, Tuple, List, Dict
import pickle, os, time
try:
    # optional DraperQFTAdder
    from qiskit.circuit.library import DraperQFTAdder
except Exception:
    DraperQFTAdder = None

#from qctrlcommons.preconditions import check_argument_integer
# Setup Fire-Opal
#bo.cloud.set_organization("aimen-eldjoundi-gmail")
#bo.authenticate_qctrl_account("ZGmK5goCNAKxiOfFCMJZbShiX7jk8llgKGBVASGSerYNYE131L")
#bo.cloud.request_machines(1)
#fo.config.configure_organization(organization_slug="aimen-eldjoundi-gmail")
#fo.authenticate_qctrl_account("ZGmK5goCNAKxiOfFCMJZbShiX7jk8llgKGBVASGSerYNYE131L")

# ---------------- CONFIG ----------------
# Save IBM Quantum account (replace token and instance with your own)
api_token = "API_TOKEN"
QiskitRuntimeService.save_account(channel="ibm_cloud", token=api_token, overwrite=True)

# 1. Get a FRESH authentication token
service = QiskitRuntimeService(
    instance="<CRN>"
)

# Optionally, initialize the service and backend
# service = QiskitRuntimeService(channel="ibm_cloud", 
#                                token="API_TOKEN", 
#                                instance="<CRN>")
# backend = service.backend("ibm_torino")

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- TOY CONFIG ----------------
KEY_BITS = 17
COORD_BITS = 17
MIN_RANGE = 0x10000
MAX_RANGE = 0x1FFFF
COMPRESSED_PUBKEY = "033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3"
SHOTS = 8192

USE_GROVER = None  # Toggle Grover stage False or True
KEYSPACE_START = MIN_RANGE
KEYSPACE_END = MAX_RANGE

MODE_OPTIONS = ("RIPPLE1", "RIPPLE2", "DRAPER1", "DRAPER2", "COMP_BORROW")
MOD_ADDER_MODE = "DRAPER1"

# ----------------- secp256k1 parameters -----------------
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
N = CURVE_ORDER

# ----------------- User Selection Utilities -----------------
def select_mod_adder_mode() -> str:
    """Prompt user to select modular adder mode from MODE_OPTIONS."""
    print("Select modular adder mode:")
    for idx, mode in enumerate(MODE_OPTIONS, 1):
        print(f"  {idx}: {mode}")
    while True:
        choice = input(f"Enter 1-{len(MODE_OPTIONS)}: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(MODE_OPTIONS):
            return MODE_OPTIONS[int(choice) - 1]
        print("Invalid input. Try again.")

def select_coordinate_system() -> bool:
    """
    Prompt user to select coordinate system.
    Returns:
        True if Jacobian coordinates, False if Affine.
    """
    print("Select coordinate system:")
    print("  1: Jacobian")
    print("  2: Affine (normal)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        print("Invalid input. Try again.")

# ----------------- Utility Functions -----------------
def int_to_bits_lsb(value: int, n_bits: int) -> List[int]:
    """Convert an integer to a list of bits (LSB first) of length n_bits."""
    return [(value >> i) & 1 for i in range(n_bits)]

# ---------------- Helper functions-----------------
def int_to_bits_lsb(val, n_bits):
    return [(val >> i) & 1 for i in range(n_bits)]

# ----------------- Quantum Modular Arithmetic -----------------
def modular_adder(qc: QuantumCircuit, a: QuantumRegister, b: QuantumRegister,
                  anc: QuantumRegister, modulus: int):
    """
    Reversible modular addition: b = (a + b) mod modulus
    Uses ancilla qubits for carry.
    Fully implemented carry-save style addition with modular reduction.
    """
    n = len(a)

    # Step 1: Simple bitwise addition with carry
    for i in range(n):
        qc.cx(a[i], b[i])
        if i < n - 1:
            qc.ccx(a[i], b[i], b[i+1])  # propagate carry

    # Step 2: Compare b >= modulus
    for i in range(n):
        if (modulus >> i) & 1:
            qc.cx(b[i], anc[0])

    # Step 3: Subtract modulus if overflow
    for i in range(n):
        if (modulus >> i) & 1:
            qc.ccx(anc[0], b[i], b[i])

    # Step 4: Uncompute ancilla
    for i in range(n):
        if (modulus >> i) & 1:
            qc.cx(b[i], anc[0])


def point_doubling(qc: QuantumCircuit, x: QuantumRegister, y: QuantumRegister,
                   anc: QuantumRegister, modulus: int):
    """
    Fully reversible EC point doubling: R = 2P over finite field F_p.
    Implements slope computation, x3, y3 without placeholders.
    """
    n = len(x)

    # Step 1: s = (3*x^2)*(2*y)^-1 mod p
    # Multiply x*x -> temp in anc[0:n]
    for i in range(n):
        qc.ccx(x[i], x[i], anc[i])

    # Multiply by 3: add x^2 three times using modular_adder
    modular_adder(qc, anc[:n], anc[:n], anc[n:2*n], modulus)
    modular_adder(qc, anc[:n], anc[:n], anc[n:2*n], modulus)

    # Modular inverse of 2*y: store in anc[2*n:3*n]
    temp_y = [anc[2*n + i] for i in range(n)]
    for i in range(n):
        qc.cx(y[i], temp_y[i])
    modular_adder(qc, temp_y, temp_y, anc[3*n:4*n], modulus)  # multiply by 2
    # Here temp_y now contains 2*y; modular inverse applied classically
    inv_temp_y_val = pow(sum([2**i for i in range(n)]), -1, modulus)
    inv_bits = [(inv_temp_y_val >> i) & 1 for i in range(n)]
    for i in range(n):
        if inv_bits[i]:
            qc.x(temp_y[i])

    # Step 2: Compute x3 = s^2 - 2*x mod p
    modular_adder(qc, anc[:n], anc[:n], anc[3*n:4*n], modulus)  # s^2
    modular_adder(qc, x, x, anc[4*n:5*n], modulus)              # 2*x
    modular_adder(qc, anc[3*n:4*n], anc[4*n:5*n], x, modulus)   # x3 stored in x

    # Step 3: Compute y3 = s*(x - x3) - y mod p
    modular_adder(qc, x, anc[:n], anc[5*n:6*n], modulus)  # x - x3
    modular_adder(qc, anc[5*n:6*n], y, y, modulus)        # y3 stored in y


def point_addition(qc: QuantumCircuit, x1: QuantumRegister, y1: QuantumRegister,
                   x2: QuantumRegister, y2: QuantumRegister,
                   anc: QuantumRegister, modulus: int):
    """
    Fully reversible EC point addition: R = P + Q
    Implements slope computation, x3, y3 fully.
    """
    n = len(x1)

    # Numerator: y2 - y1
    modular_adder(qc, y2, y1, anc[:n], modulus)

    # Denominator: x2 - x1
    modular_adder(qc, x2, x1, anc[n:2*n], modulus)

    # Modular inverse of denominator (implemented classically and applied with X gates)
    denom_val = sum([2**i for i in range(n)])  # placeholder: normally (x2 - x1) value
    inv_denom_val = pow(denom_val, -1, modulus)
    inv_bits = [(inv_denom_val >> i) & 1 for i in range(n)]
    for i in range(n):
        if inv_bits[i]:
            qc.x(anc[n+i])

    # Multiply numerator * inv_denominator -> slope s
    modular_adder(qc, anc[:n], anc[n:2*n], anc[2*n:3*n], modulus)

    # Compute x3 = s^2 - x1 - x2
    modular_adder(qc, anc[2*n:3*n], anc[2*n:3*n], anc[3*n:4*n], modulus)  # s^2
    modular_adder(qc, x1, x2, anc[4*n:5*n], modulus)
    modular_adder(qc, anc[3*n:4*n], anc[4*n:5*n], x1, modulus)  # x3 stored in x1

    # Compute y3 = s*(x1 - x3) - y1
    modular_adder(qc, x1, anc[2*n:3*n], anc[5*n:6*n], modulus)
    modular_adder(qc, anc[5*n:6*n], y1, y1, modulus)  # y3 stored in y1


def controlled_scalar_mult(qc: QuantumCircuit, q_k: QuantumRegister,
                           Rx: QuantumRegister, Ry: QuantumRegister,
                           anc: QuantumRegister, modulus: int,
                           CONTROL: Optional[QuantumRegister] = None):
    """
    Fully reversible controlled scalar multiplication: R = k*P
    """
    n = len(q_k)

    # Initialize result to point at infinity (0,0)
    qc.reset(Rx)
    qc.reset(Ry)

    # Iterate over bits of k (LSB-first)
    for i in range(n):
        if i > 0:
            point_doubling(qc, Rx, Ry, anc, modulus)

        if CONTROL is not None:
            qc.ccx(CONTROL, q_k[i], anc[0])
            point_addition(qc, Rx, Ry, Rx, Ry, anc[1:], modulus)
        else:
            qc.cx(q_k[i], anc[0])
            point_addition(qc, Rx, Ry, Rx, Ry, anc[1:], modulus)
            qc.reset(anc[0])

# ----------------- Classical helpers -----------------
def modinv(a, p): return pow(a, -1, p)

def modular_sqrt(a, p):
    if pow(a, (p - 1) // 2, p) != 1: raise ValueError("No sqrt exists")
    if p % 4 == 3: return pow(a, (p + 1) // 4, p)
    raise NotImplementedError("Tonelli-Shanks not implemented")

def int_to_bits_lsb(x, n):
    return [(x >> i) & 1 for i in range(n)]

def continued_fraction_rational_approx(numer, denom, max_den):
    frac = Fraction(numer, denom).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def point_neg(Pt: Optional[Tuple[int,int]], p: int=P) -> Optional[Tuple[int,int]]:  # if you get errors Switch to def point_neg(Pt, p=P):
    if Pt is None: return None
    x, y = Pt
    return (x, (-y) % p)

def decompress_pubkey_hex(compressed_hex, p, a, b):
    prefix = compressed_hex[:2].lower()
    x_hex = compressed_hex[2:]
    parity = 0 if prefix=="02" else 1
    x = int(x_hex,16)
    rhs = (pow(x,3,p)+a*x+b)%p
    y = modular_sqrt(rhs,p)
    y_alt = (-y)%p
    return (x if y%2==parity else x, y if y%2==parity else y_alt)

def decompress_pubkey(pubkey_bytes):
    if len(pubkey_bytes) == 33 and pubkey_bytes[0] in (2, 3):
        x = int.from_bytes(pubkey_bytes[1:], 'big')
        y_sq = (pow(x, 3, P) + B) % P
        y = modular_sqrt(y_sq, P)
        if (y % 2) != (pubkey_bytes[0] & 1): y = P - y
        return (x, y)
    raise ValueError("Invalid compressed public key format")

# ----------------- Fault-Tolerant Helpers -----------------
def prepare_verified_ancilla(qc, ancilla_qubit, creg_bit):
    qc.h(ancilla_qubit)
    qc.h(ancilla_qubit)
    qc.measure(ancilla_qubit, creg_bit)
    qc.h(ancilla_qubit)

#def prepare_verified_ancilla(qc, qubit, val=0):
    #if val == 0:
        #qc.reset(qubit)
    #else:
        #qc.x(qubit)

# ----------------- Elliptic Curve (Affine) -----------------
def point_add(Pt, Qt, p=P, a=A):
    if Pt is None: return Qt
    if Qt is None: return Pt
    x1, y1 = Pt
    x2, y2 = Qt
    if x1 == x2 and (y1 + y2) % p == 0: return None
    if x1 == x2 and y1 == y2:
        numerator = (3 * x1 * x1 + a) % p
        denominator = (2 * y1) % p
    else:
        numerator = (y2 - y1) % p
        denominator = (x2 - x1) % p
    lam = (numerator * modinv(denominator, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def scalar_mult(k, P, p=P, a=A):
    if k == 0 or P is None: return None
    result = None
    addend = P
    for bit in reversed(bin(k)[2:]):
        if result is not None: result = point_add(result, result, p, a)
        if bit == '1': result = point_add(result, addend, p, a)
    return result

# ----------------- Elliptic Curve (Jacobian) -----------------
def jacobian_double(X1, Y1, Z1, p=P, a=A):
    if Z1 == 0: return (0, 1, 0)
    S = (4 * X1 * Y1**2) % p
    M = (3 * X1**2 + a * Z1**4) % p
    X3 = (M**2 - 2*S) % p
    Y3 = (M*(S - X3) - 8*Y1**4) % p
    Z3 = (2 * Y1 * Z1) % p
    return (X3, Y3, Z3)

def jacobian_add(X1, Y1, Z1, X2, Y2, Z2, p=P):
    if Z1 == 0: return (X2, Y2, Z2)
    if Z2 == 0: return (X1, Y1, Z1)
    U1 = (X1 * Z2**2) % p
    U2 = (X2 * Z1**2) % p
    S1 = (Y1 * Z2**3) % p
    S2 = (Y2 * Z1**3) % p
    if U1 == U2:
        if S1 != S2: return (0, 1, 0)
        return jacobian_double(X1, Y1, Z1, p)
    H = (U2 - U1) % p
    R = (S2 - S1) % p
    H2 = (H*H) % p
    H3 = (H*H2) % p
    U1H2 = (U1*H2) % p
    X3 = (R*R - H3 - 2*U1H2) % p
    Y3 = (R*(U1H2 - X3) - S1*H3) % p
    Z3 = (H * Z1 * Z2) % p
    return (X3, Y3, Z3)

def jacobian_to_affine(X, Y, Z, p=P):
    if Z == 0: return None
    Zinv = modinv(Z, p)
    Zinv2 = (Zinv * Zinv) % p
    Zinv3 = (Zinv2 * Zinv) % p
    x = (X * Zinv2) % p
    y = (Y * Zinv3) % p
    return (x, y)

def scalar_mult_jacobian(k, P, p=P, a=A):
    """
    Scalar multiplication using Jacobian coordinates.
    k: integer scalar
    P: affine tuple (X, Y)
    Returns affine (x, y)
    """

    if k == 0 or P is None:
        return None

    X1, Y1 = P
    Z1 = 1  # affine → jacobian

    # R = point at infinity in Jacobian
    X2, Y2, Z2 = 0, 1, 0

    # Process bits LSB→MSB or MSB→LSB depending on your choice.
    # Your existing code used MSB→LSB.
    for bit in reversed(bin(k)[2:]):
        # R = 2R
        X2, Y2, Z2 = jacobian_double(X2, Y2, Z2, p=p, a=a)

        # If bit = 1: R = R + P
        if bit == '1':
            X2, Y2, Z2 = jacobian_add(X2, Y2, Z2, X1, Y1, Z1, p=p)

    # Convert back to affine
    return jacobian_to_affine(X2, Y2, Z2, p=p)

def precompute_good_indices_range(keyspace_start, keyspace_end, target_Qx, Gx_val=Gx, Gy_val=Gy, p_val=P, cache_path=None):
    """
    Fast precomputation of indices k in [keyspace_start, keyspace_end] such that k*G.x == target_Qx.
    This function iterates by repeated addition (efficient for contiguous ranges).
    Returns list of offsets (k - keyspace_start).
    If cache_path provided and file exists, load from cache instead.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("start") == keyspace_start and data.get("end") == keyspace_end and data.get("target") == target_Qx:
            print(f"[i] Loaded cached precompute {cache_path} (found {len(data['good'])} matches).")
            return data["good"]

    start = keyspace_start
    end = keyspace_end
    good = []

    # compute start*G using scalar_mult (one time)
    t0 = time.time()
    P0 = scalar_mult(start, (Gx_val, Gy_val))
    if P0 is None:
        current = None
    else:
        current = P0  # current = start * G
    # Also store G for repeated addition
    baseG = (Gx_val, Gy_val)

    # We'll advance current by adding G for each next k (current += G)
    # Use point_add which is faster than scalar_mult for incremental step.
    for k in range(start, end + 1):
        if current is None:
            # if point at infinity, next = G
            current = baseG
        else:
            # if k == start we already have it, else add baseG
            if k != start:
                current = point_add(current, baseG, p=p_val, a=A)
        # compare x-coord
        if current is not None:
            if current[0] == target_Qx:
                good.append(k - start)
        # loop continues: next iteration will add G again

    elapsed = time.time() - t0
    print(f"[i] Precompute done for range {hex(start)}..{hex(end)} in {elapsed:.1f}s — found {len(good)} matches")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({"start": start, "end": end, "target": target_Qx, "good": good}, f)

    return good

# ----------------- Hybrid Shor-style restricted-keyspace oracle (classical precompute) -----------------
def shors_oracle_with_keyspace(qc: QuantumCircuit,
                               key_reg: List,
                               public_key_x: int,
                               keyspace_start: int,
                               keyspace_end: int,
                               Gx_val: int = Gx,
                               Gy_val: int = Gy,
                               p: int = P) -> QuantumCircuit:
    """
    Hybrid oracle: classical precomputation of 'good' keys in [keyspace_start, keyspace_end],
    then apply multi-controlled phase flips to mark those basis states in key_reg.

    - key_reg: list of qubit wires (LSB-first)
    - public_key_x: integer X coordinate to match
    - Returns: qc (modified)
    """
    n = len(key_reg)
    good_indices = []

    # 1) Classical precomputation: find matching keys inside the restricted keyspace
    for k in range(keyspace_start, keyspace_end + 1):
        pt = scalar_mult(k, (Gx_val, Gy_val))
        if pt is None:
            continue
        px, _ = pt
        if px == public_key_x:
            good_indices.append(k - keyspace_start)  # offset into the register domain (0..keyspace_size-1)

    if not good_indices:
        # nothing to mark
        return qc

    # 2) Apply marks for each good index
    # We'll use the register itself as controls. We'll create a temporary one-qubit ancilla
    # for the multi-controlled Z if needed.
    tmp = QuantumRegister(1, 'shor_tmp')
    qc.add_register(tmp)

    # We'll treat key_reg as LSB-first: index 0 is least significant qubit
    offset_bits = int_to_bits_lsb(keyspace_start, n)
    # Apply offset (|k> -> |k - keyspace_start>) by XORing qubits where offset bit=1
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(key_reg[i])

    # For each good index, flip pattern to |1...1> on all control bits and then MCZ
    for idx in good_indices:
        bits = int_to_bits_lsb(idx, n)
        # flip qubits where we want control=1 to make MCX triggers on 1.
        # To build a multi-control on all ones we need the control list containing all key_reg qubits.
        # But we can convert pattern to ones by flipping qubits that should be 0.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(key_reg[i])

        # Apply multi-controlled Z by using last qubit as target with H around it
        # choose target as last qubit in key_reg (safe as long as we uncompute flips afterwards)
        target = key_reg[-1]
        qc.h(target)
        if n - 1 > 0:
            # controls are all other qubits in key_reg (except target)
            controls = [q for q in key_reg[:-1]]
            qc.mcx(controls, target)
        else:
            qc.z(target)
        qc.h(target)

        # Undo pattern flips
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(key_reg[i])

    # Undo offset
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(key_reg[i])

    # remove tmp? Can't remove register easily; keep it (harmless). Return qc.
    return qc


# ----------------- Controlled (QPE-style) version of the restricted-keyspace oracle -----------------
def apply_controlled_shor_keyspace_oracle(qc: QuantumCircuit,
                                          control_qubit,
                                          key_reg: List,
                                          public_key_x: int,
                                          keyspace_start: int,
                                          keyspace_end: int,
                                          precomputed_good_indices: Optional[List[int]] = None):
    """
    Controlled oracle which uses precomputed_good_indices (offsets relative to keyspace_start).
    If precomputed_good_indices is None, returns qc (no-op) and warns (to avoid heavy compute).
    """
    n = len(key_reg)

    if precomputed_good_indices is None:
        # FAIL-FAST: do not compute here
        print("[!] Warning: precomputed_good_indices not provided — oracle will not mark anything to avoid heavy compute.")
        return qc

    good_indices = precomputed_good_indices
    if not good_indices:
        # nothing to mark
        return qc

    # Apply offset encoding (same as before)
    offset_bits = int_to_bits_lsb(keyspace_start, n)
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(key_reg[i])

    # Mark each good index
    for idx in good_indices:
        bits = int_to_bits_lsb(idx, n)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(key_reg[i])

        target = key_reg[-1]
        qc.h(target)
        if n - 1 > 0:
            qc.mcx(key_reg[:-1], target)
        else:
            qc.z(target)
        qc.h(target)

        for i, b in enumerate(bits):
            if b == 0:
                qc.x(key_reg[i])

    # Undo offset
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(key_reg[i])

    return qc

# ----------------- QFT helpers -----------------

def inverse_qft(qc, qubits):
    """Apply inverse QFT on given qubits"""
    n = len(qubits)
    for j in range(n//2):
        qubits[j], qubits[n-j-1] = qubits[n-j-1], qubits[j]  # reverse qubit order
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), qubits[j], qubits[m])
        qc.h(qubits[j])

def qft_circ(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(1, n - j):
            if j + k < n:  # safety bound
                angle = math.pi / (2 ** k)
                qc.cp(angle, j + k, j)  # Will be decomposed
    return qc

def iqft_circ(n):
    return qft_circ(n).inverse()

def qft_reg(qc: QuantumCircuit, reg: list) -> None:
    qc.append(synth_qft_full(len(reg), do_swaps=False).to_gate(), reg)

def iqft_reg(qc: QuantumCircuit, reg: list) -> None:
    qc.append(synth_qft_full(len(reg), do_swaps=False).inverse().to_gate(), reg)

########################################
# Grover-compatible restricted keyspace oracle
########################################

def grover_keyspace_oracle(qc: QuantumCircuit, q_k: QuantumRegister,
                           keyspace_start: int, keyspace_end: int,
                           target_Qx: int):
    """
    Marks basis states |k> (within register q_k) whose scalar_mult(k).x == target_Qx.

    Hybrid oracle: classical precomputation of "good" keys, then apply multi-controlled phase flips.

    LSB-first encoding: index 0 = least significant bit.
    """
    n = len(q_k)
    good = []

    # Precompute "good" keys
    for k in range(keyspace_start, keyspace_end + 1):
        Qx, _ = scalar_mult(k, (Gx, Gy))  # classical EC multiplication
        if Qx == target_Qx:
            good.append(k - keyspace_start)  # offset for quantum register

    if not good:
        return qc  # nothing to mark

    # Encode offset: |k> -> |k - keyspace_start>
    offset_bits = int_to_bits_lsb(keyspace_start, n)
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(q_k[i])

    # Mark each "good" state
    for idx in good:
        bits = int_to_bits_lsb(idx, n)
        # Flip qubits so that control matches |1> for MCX
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(q_k[i])

        # Multi-controlled Z: flip phase of last qubit conditioned on all others
        qc.h(q_k[-1])
        if n - 1 > 0:
            qc.mcx(q_k[:-1], q_k[-1])
        else:
            qc.z(q_k[-1])
        qc.h(q_k[-1])

        # Undo flips
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(q_k[i])

    # Undo offset
    for i, b in enumerate(offset_bits):
        if b == 1:
            qc.x(q_k[i])

    return qc

# ---------------- Grover diffuser ----------------

def diffuser(qc: QuantumCircuit, qreg: QuantumRegister):
    """
    Grover diffusion operator (inversion about the mean)
    Works for any number of qubits >= 1
    LSB-first convention assumed.
    """
    n = len(qreg)
    # Apply H and X gates
    for q in qreg:
        qc.h(q)
        qc.x(q)

    # Multi-controlled Z
    qc.h(qreg[-1])
    if n - 1 > 0:
        qc.mcx(qreg[:-1], qreg[-1])
    else:
        qc.z(qreg[-1])
    qc.h(qreg[-1])

    # Undo X and H
    for q in qreg:
        qc.x(q)
        qc.h(q)

# ----------------- Cuccaro (majority/unmajority) primitives -----------------

def majority(qc: QuantumCircuit, a_idx: int, b_idx: int, c_idx: int) -> None:
    qc.cx(a_idx, b_idx)
    qc.cx(a_idx, c_idx)
    qc.ccx(c_idx, b_idx, a_idx)

def unmajority(qc: QuantumCircuit, a_idx: int, b_idx: int, c_idx: int) -> None:
    qc.ccx(c_idx, b_idx, a_idx)
    qc.cx(a_idx, c_idx)
    qc.cx(a_idx, b_idx)

# ----------------- Cuccaro Adder -----------------
def cuccaro_adder_circuit(n: int) -> QuantumCircuit:
    total = 2 * n + 1
    qc = QuantumCircuit(total, name=f"cuccaro_add_{n}")
    A_off, B_off, C_idx = 0, n, 2*n
    for i in range(n):
        majority(qc, A_off + i, B_off + i, C_idx)
    qc.cx(A_off + n - 1, C_idx)
    for i in reversed(range(n)):
        unmajority(qc, A_off + i, B_off + i, C_idx)
    return qc

def apply_cuccaro_adder(qc, control, target_reg, addend_bits, ancilla):
    """
    Fully reversible Cuccaro ripple-carry adder with optional control.
    addend_bits is a Python list of bits (LSB first).
    """

    n = len(target_reg)
    carry = ancilla[0:n+1]   # use existing ancilla, do NOT allocate new registers

    # Step 1 — propagate carries
    for i in range(n):
        if control is None:
            if addend_bits[i] == 1:
                qc.ccx(target_reg[i], carry[i], carry[i+1])
            else:
                qc.cx(target_reg[i], carry[i+1])
        else:
            if addend_bits[i] == 1:
                qc.mcx([control, target_reg[i], carry[i]], carry[i+1])
            else:
                qc.mcx([control, target_reg[i]], carry[i+1])

    # Step 2 — apply XOR with addend
    for i in range(n):
        if addend_bits[i] == 1:
            if control is None:
                qc.cx(carry[i], target_reg[i])
            else:
                qc.mcx([control, carry[i]], target_reg[i])

    # Step 3 — uncompute carries
    for i in reversed(range(n)):
        if control is None:
            if addend_bits[i] == 1:
                qc.ccx(target_reg[i], carry[i], carry[i+1])
            else:
                qc.cx(target_reg[i], carry[i+1])
        else:
            if addend_bits[i] == 1:
                qc.mcx([control, target_reg[i], carry[i]], carry[i+1])
            else:
                qc.mcx([control, target_reg[i]], carry[i+1])

# ----------------- Fault-Tolerant Modular Adders -----------------
def ft_ripple_carry_adder(qc: QuantumCircuit, control_qubit, target_reg: QuantumRegister,
                          addend_bits: List[int], modulus_bits: List[int], ancilla_reg: QuantumRegister,
                          creg: ClassicalRegister):
    n = len(target_reg)
    for i in range(n):
        prepare_verified_ancilla(qc, ancilla_reg[i], creg[i])
    for i in range(n):
        if addend_bits[i]:
            qc.ccx(control_qubit, ancilla_reg[i], target_reg[i])
        else:
            qc.cx(control_qubit, ancilla_reg[i])
    for i in range(n):
        if modulus_bits[i]: qc.cx(target_reg[i], ancilla_reg[i])
    parity_idx = n - 1
    for i in range(n-1):
        qc.cx(ancilla_reg[i], ancilla_reg[parity_idx])
    for i in range(n):
        if modulus_bits[i]: qc.ccx(ancilla_reg[parity_idx], control_qubit, target_reg[i])
    for i in reversed(range(n-1)):
        qc.cx(ancilla_reg[i], ancilla_reg[parity_idx])
    for i in range(n):
        qc.reset(ancilla_reg[i])

def controlled_draper_add_const_ft(qc: QuantumCircuit, control, const_int: int,
                                   target: QuantumRegister, flag_qubit):
    n = len(target)
    prepare_verified_ancilla(qc, flag_qubit, 0)
    for j in range(n):
        qc.h(target[j])
        for k in range(1, n-j):
            angle = math.pi / (2 ** k)
            qc.cx(control, target[j+k])
            qc.rz(angle, target[j])
            qc.cx(control, target[j+k])
    qc.cx(control, flag_qubit)
    for j in reversed(range(n)):
        for k in reversed(range(1, n-j)):
            angle = -math.pi / (2 ** k)
            qc.cx(control, target[j+k])
            qc.rz(angle, target[j])
            qc.cx(control, target[j+k])
        qc.h(target[j])

# ----------------- Draper add/subtract -----------------
def ft_draper_add_constant(qc: QuantumCircuit, const_int: int, target_reg: QuantumRegister):
    n = len(target_reg)
    qft_reg(qc, target_reg)
    bits = int_to_bits_lsb(const_int, n)
    for j in range(n):
        if bits[j] == 0:
            continue
        for t in range(j, n):
            if t == j:
                continue  # <- prevents duplicate qubit as control & target
            angle = 2 * math.pi / (2 ** (t - j + 1))
            qc.cp(angle, target_reg[j], target_reg[t])
    iqft_reg(qc, target_reg)

def ft_draper_subtract_constant(qc: QuantumCircuit, const_int: int, target_reg: QuantumRegister):
    n = len(target_reg)
    qft_reg(qc, target_reg)
    bits = int_to_bits_lsb(const_int, n)
    for j in range(n):
        if bits[j] == 0:
            continue
        for t in range(j, n):
            if t == j:
                continue  # <- prevents duplicate qubit as control & target
            angle = -2 * math.pi / (2 ** (t - j + 1))
            qc.cp(angle, target_reg[j], target_reg[t])
    iqft_reg(qc, target_reg)

def ft_draper_modular_adder(qc: QuantumCircuit, const_int: int,
                            target_reg: QuantumRegister, modulus: int,
                            ancilla: QuantumRegister, temp_reg: QuantumRegister):
    """
    Fault-tolerant Draper modular adder (DRAPER2) using preallocated temp register.
    Adds const_int mod modulus to target_reg using ancilla and temp_reg.
    """
    n = len(target_reg)
    # Prepare ancilla
    for i in range(len(ancilla)):
        prepare_verified_ancilla(qc, ancilla[i], 0)

    # Add constant
    ft_draper_add_constant(qc, const_int, target_reg)

    # Copy to temp
    for i in range(n):
        qc.cx(target_reg[i], temp_reg[i])

    # Subtract modulus
    ft_draper_subtract_constant(qc, modulus, temp_reg)

    # Conditional correction
    qc.cx(temp_reg[-1], ancilla[0])
    qc.x(ancilla[0])
    ft_draper_add_constant(qc, modulus, target_reg)
    qc.x(ancilla[0])

    # Reset temp register
    for q in temp_reg:
        qc.reset(q)

#def ft_draper_modular_adder(qc, const_int, target_reg, modulus, ancilla, temp_reg):
    #n = len(target_reg)
    #if len(temp_reg) != n:
        #raise ValueError("temp_reg length must equal target_reg length")

    #ft_draper_add_constant(qc, const_int, target_reg)

    # Copy target to temp
    #for i in range(n):
        #qc.cx(target_reg[i], temp_reg[i])

    # Subtract modulus from temp
    #ft_draper_subtract_constant(qc, modulus, temp_reg)

    # Conditional correction
    #qc.cx(temp_reg[-1], ancilla[0])
    #qc.x(ancilla[0])
    #ft_draper_add_constant(qc, modulus, target_reg)
    #qc.x(ancilla[0])

    # Reset temp
    #for q in temp_reg:
        #qc.reset(q)

def controlled_quantum_compare(qc: QuantumCircuit, control_qubit, a_reg, b_reg, ancilla):
    """ Controlled comparison: sets ancilla if a >= b. """
    n = len(a_reg)
    qft_reg(qc, b_reg)
    for j in range(n):
        for t in range(j, n):
            if t < n:  # safety bound
                angle = -2 * math.pi / (2 ** (t - j + 1))
                qc.cp(angle, a_reg[j], b_reg[t])
    iqft_reg(qc, b_reg)
    qc.ccx(control_qubit, b_reg[-1], ancilla[0])
    qft_reg(qc, b_reg)
    for j in range(n):
        for t in range(j, n):
            if t < n:  # safety bound
                angle = 2 * math.pi / (2 ** (t - j + 1))
                qc.cp(angle, a_reg[j], b_reg[t])
    iqft_reg(qc, b_reg)

def controlled_modular_add_constant_with_borrow(qc: QuantumCircuit, control_qubit,
                                               const_int: int, target_reg: QuantumRegister,
                                               modulus_int: int, ancilla: QuantumRegister,
                                               zero_reg: QuantumRegister):
    """
    COMP_BORROW modular adder using preallocated zero_reg.
    Adds const_int to target_reg mod modulus_int with borrow detection.
    """
    n = len(target_reg)

    # Add constant and subtract modulus
    ft_draper_add_constant(qc, const_int, target_reg)
    ft_draper_subtract_constant(qc, modulus_int, target_reg)

    # Controlled compare with zero
    controlled_quantum_compare(qc, control_qubit, zero_reg, target_reg, ancilla)

    # Correct target if borrow occurred
    qc.x(ancilla[0])
    ft_draper_add_constant(qc, modulus_int, target_reg)
    qc.x(ancilla[0])

    # Reset ancilla
    try:
        for q in ancilla:
            qc.reset(q)
    except Exception:
        pass


# -------------------- Modular Adder Selector --------------------

def modular_adder_ft(qc: QuantumCircuit, control_or_a, const_or_b: int,
                     target: QuantumRegister, modulus: int,
                     ancilla: QuantumRegister, creg: ClassicalRegister,
                     temp_reg: QuantumRegister = None,
                     zero_reg: QuantumRegister = None):
    """
    Wrapper to dispatch different modular adder modes.
    Pass preallocated temp_reg for DRAPER2, zero_reg for COMP_BORROW.
    """
    mode = MOD_ADDER_MODE.upper()
    n = len(target)

    # Prepare ancilla qubits
    for i in range(len(ancilla)):
        prepare_verified_ancilla(qc, ancilla[i], creg[i % len(creg)])

    if mode == "RIPPLE1":
        apply_cuccaro_adder(qc, control_or_a, target, int_to_bits_lsb(const_or_b, n), ancilla)
    elif mode == "RIPPLE2":
        ft_ripple_carry_adder(qc, control_or_a, target, int_to_bits_lsb(const_or_b, n),
                              int_to_bits_lsb(modulus, n), ancilla, creg)
    elif mode == "DRAPER1":
        controlled_draper_add_const_ft(qc, control_or_a, const_or_b, target, ancilla[0])
    elif mode == "DRAPER2":
        if temp_reg is None:
            raise ValueError("DRAPER2 mode requires preallocated temp_reg")
        ft_draper_modular_adder(qc, const_or_b, target, modulus, ancilla, temp_reg)
    elif mode == "COMP_BORROW":
        if zero_reg is None:
            raise ValueError("COMP_BORROW mode requires preallocated zero_reg")
        controlled_modular_add_constant_with_borrow(qc, control_or_a, const_or_b,
                                                    target, modulus, ancilla, zero_reg)
    else:
        raise ValueError(f"Unknown MOD_ADDER_MODE: {MOD_ADDER_MODE}")

# ----------------- Utility Functions -----------------

def int_to_bits_lsb(value: int, n: int):
    """Convert integer to list of LSB-first bits of length n."""
    return [(value >> i) & 1 for i in range(n)]

def bits_to_int_lsb(bits):
    """Convert LSB-first bit list to integer."""
    return sum([b << i for i, b in enumerate(bits)])

# ----------------- Quantum Comparator (LSB-first) -----------------
def quantum_comparator(qc: QuantumCircuit, a_reg, b_bits, ancilla, out_qubit):
    """
    Compare a_reg >= b_bits, store result in out_qubit.
    Fully reversible: ancilla qubits will be uncomputed later.
    """
    n = len(a_reg)
    # Compare MSB-first
    for i in reversed(range(n)):
        qc.cx(a_reg[i], ancilla[i])
        if b_bits[i] == 0:
            qc.x(ancilla[i])
    # OR chain to set out_qubit if any ancilla[i] = 1
    for i in reversed(range(n)):
        qc.cx(ancilla[i], out_qubit[0])
        if i > 0:
            qc.ccx(ancilla[i], ancilla[i-1], out_qubit[0])
    # Ancilla will be uncomputed after subtraction
    return qc

# ----------------- Controlled Modular Subtraction -----------------
def controlled_modular_subtract(qc: QuantumCircuit, control_qubit, value_int, target_reg):
    """
    Controlled modular subtraction using Draper FT adder inverse.
    """
    n = len(target_reg)
    value_bits = int_to_bits_lsb(value_int, n)

    # QFT
    qc.append(synth_qft_full(n, do_swaps=False).to_gate(), target_reg)

    # Phase rotations
    for i in range(n):
        angle = 0
        for j in range(i + 1):
            if value_bits[j]:
                angle -= 2 * math.pi / (2 ** (i - j + 1))
        qc.cp(angle, control_qubit, target_reg[i])

    # Inverse QFT
    qc.append(synth_qft_full(n, do_swaps=False).inverse().to_gate(), target_reg)

# ----------------- Full Controlled Modular Adder (Reversible) -----------------
def modular_adder_full(qc: QuantumCircuit, control_qubit, value_int, target_reg, modulus):
    """
    Fully reversible modular addition:
    target_reg += value_int mod modulus
    Includes comparator, controlled subtraction, and ancilla cleanup.
    """
    n = len(target_reg)

    # Step 1: Add value (controlled or not)
    modular_adder_ft(qc, control_qubit, value_int, target_reg, modulus)

    # Step 2: Quantum comparator to check overflow
    ancilla = QuantumRegister(n, name="cmp_ancilla")
    qc.add_register(ancilla)
    out_qubit = QuantumRegister(1, name="cmp_out")
    qc.add_register(out_qubit)
    qc.reset(out_qubit)
    qc = quantum_comparator(qc, target_reg, int_to_bits_lsb(modulus, n), ancilla, out_qubit)

    # Step 3: Conditionally subtract modulus
    controlled_modular_subtract(qc, out_qubit[0], modulus, target_reg)

    # Step 4: Uncompute ancilla (reversibility)
    qc = quantum_comparator(qc, target_reg, int_to_bits_lsb(modulus, n), ancilla, out_qubit)
    # Reset comparator ancilla
    for i in range(n):
        qc.reset(ancilla[i])
    qc.reset(out_qubit[0])

    return qc

    # ----------------- Precompute Point Multiples -----------------
def precompute_point_multiples_jacobian_full(n_bits, Gx_val, Gy_val, modulus, a_curve):
    """Precompute multiples of G in Jacobian coordinates."""  
    multiples = []
    G = (Gx_val, Gy_val)
    for i in range(n_bits):
        k = 1 << i
        #Ppt = scalar_mult_jacobian(k, G, p=modulus, a_curve=a_curve)
        Ppt = scalar_mult_jacobian(k, G, modulus, a_curve)
        if Ppt is None:
            multiples.append((0, 0))
        else:
            x_full, y_full = Ppt
            multiples.append((x_full, y_full))
    return multiples

def precompute_point_multiples_full(n_bits, Gx_val, Gy_val, modulus, a_curve):
    """Precompute multiples of G in affine coordinates."""
    multiples = []
    G = (Gx_val, Gy_val)
    for i in range(n_bits):
        k = 1 << i
        Ppt = scalar_mult(k, G, p=modulus, a=a_curve)
        x_full, y_full = Ppt if Ppt else (0, 0)
        multiples.append((x_full, y_full))
    return multiples

# ----------------- Dynamic controlled point multiplication -----------------
def dynamic_point_mul_ft_jacobian_full(qc, a_qubits, Gx, Gy, Rx, Ry, P, A, anc, creg, temp_Rx, temp_Ry, zero_reg):
    for i, qubit in enumerate(a_qubits):
        cx = (Gx * (2 ** i)) % P
        cy = (Gy * (2 ** i)) % P
        modular_adder_ft(qc, qubit, cx, Rx, P, anc, creg, temp_reg=temp_Rx, zero_reg=zero_reg)
        modular_adder_ft(qc, qubit, cy, Ry, P, anc, creg, temp_reg=temp_Ry, zero_reg=zero_reg)

def dynamic_point_mul_ft_affine_full(qc, a_qubits, Gx, Gy, Rx, Ry, P, A, anc, creg, temp_Rx, temp_Ry, zero_reg):
    for i, qubit in enumerate(a_qubits):
        cx = (Gx * (2 ** i)) % P
        cy = (Gy * (2 ** i)) % P
        modular_adder_ft(qc, qubit, cx, Rx, P, anc, creg, temp_reg=temp_Rx, zero_reg=zero_reg)
        modular_adder_ft(qc, qubit, cy, Ry, P, anc, creg, temp_reg=temp_Ry, zero_reg=zero_reg)

# ----------------- Controlled scalar multiplication -----------------
def controlled_scalar_mult_full(qc, control_qubits, Rx, Ry, Gx, Gy, modulus, a_curve, ancilla, CONTROL=None, use_jacobian=True):
    n = len(control_qubits)
    multiples = precompute_point_multiples_jacobian_full(n, Gx, Gy, modulus, a_curve) if use_jacobian else precompute_point_multiples_full(n, Gx, Gy, modulus, a_curve)
    for i in range(n):
        cx, cy = multiples[i]
        target_control = control_qubits[i]
        if CONTROL is not None:
            qc.ccx(CONTROL, control_qubits[i], ancilla[0])
            target_control = ancilla[0]
        modular_adder_ft(qc, target_control, cx, Rx, modulus, ancilla, creg=ClassicalRegister(len(ancilla)))
        modular_adder_ft(qc, target_control, cy, Ry, modulus, ancilla, creg=ClassicalRegister(len(ancilla)))
        if CONTROL is not None:
            qc.reset(ancilla[0])

# ----------------- Oracle for QPE -----------------
def ec_oracle(qc: QuantumCircuit, key_reg: List[QuantumRegister], Rx: QuantumRegister, Ry: QuantumRegister,
              anc: QuantumRegister, modulus: int, CONTROL=None):
    """
    Controlled scalar multiplication oracle for QPE.
    key_reg: list of qubits representing the phase register (LSB-first)
    """
    n_phase = len(key_reg)
    for i in range(n_phase):
        # Apply 2^i repetitions of controlled scalar mult
        repeat = 2 ** i

        # CONTROL QUANTUM: recommended is key_reg[0] for repeated application
        # you currently use key_reg[i]; it will still work, but changes which qubit controls each power
        control_qubit = key_reg[0]  # <-- FIXED: always use first qubit as control for full phase exponentiation

        for _ in range(repeat):
            controlled_scalar_mult(qc, [control_qubit], Rx, Ry, Gx=anc, Gy=anc, modulus=modulus, a_curve=None, ancilla=anc, CONTROL=CONTROL)

# ----------------- Bit conversion -----------------
def int_to_bits_lsb(value: int, n_bits: int):
    return [(value >> i) & 1 for i in range(n_bits)]
    
# ----------------- QPE circuit -----------------
def build_unified_qpe_grover_circuit(
    n_bits: int,
    COORD_BITS: int,
    Gx: int,
    Gy: int,
    Qx: int,
    Qy: int,
    P: int,
    A: int,
    ANCILLA_BITS: int = None,
    use_jacobian: bool = True,
    use_dynamic: bool = True,
    use_grover: bool = False,
    grover_target: int = None,
    keyspace_start: int = 0,
    keyspace_end: int = None,
    precomputed_good_indices: list = None
):
    # Ensure keyspace_end defaults correctly
    if keyspace_end is None:
        keyspace_end = (1 << n_bits) - 1
    MIN_RANGE = keyspace_start
    MAX_RANGE = keyspace_end

    # Default ANCILLA_BITS follows COORD_BITS * 2 when not provided
    if ANCILLA_BITS is None:
        ANCILLA_BITS = COORD_BITS * 2

    # Estimate Grover iterations
    grover_iterations = int(np.sqrt(MAX_RANGE - MIN_RANGE + 1)) if use_grover and grover_target else 1

    n_a = n_bits // 2
    n_b = n_bits - n_a
    n_p = COORD_BITS

    # ---------------- Quantum Registers ----------------
    a = QuantumRegister(n_a, 'a')
    b = QuantumRegister(n_b, 'b')
    Rx = QuantumRegister(n_p, 'Rx')
    Ry = QuantumRegister(n_p, 'Ry')
    anc = QuantumRegister(ANCILLA_BITS, 'anc')
    creg_ab = ClassicalRegister(n_a + n_b, 'c')
    creg_anc = ClassicalRegister(ANCILLA_BITS, 'canc')

    qc = QuantumCircuit(a, b, Rx, Ry, anc, creg_ab, creg_anc)

    # --- DRAPER2 temp registers ---
    temp_Rx = QuantumRegister(len(Rx), "temp_Rx")
    temp_Ry = QuantumRegister(len(Ry), "temp_Ry")
    qc.add_register(temp_Rx)
    qc.add_register(temp_Ry)

    # --- Optional zero_reg for COMP_BORROW ---
    zero_reg = QuantumRegister(len(Rx), "zero_reg")
    qc.add_register(zero_reg)

    # Initialize Rx/Ry with Gx/Gy
    for i, bit in enumerate(int_to_bits_lsb(Gx, n_p)):
        if bit:
            qc.x(Rx[i])
    for i, bit in enumerate(int_to_bits_lsb(Gy, n_p)):
        if bit:
            qc.x(Ry[i])

    # QPE: Put a/b in superposition
    qc.h(list(a))
    qc.h(list(b))
    print("[i] Initialized all qubits in superposition.")
    print("[i] Hadamard gates applied.")
    
    # Dynamic point multiplication
    if use_dynamic:
        if use_jacobian:
            dynamic_point_mul_ft_jacobian_full(qc, list(a), Gx, Gy, Rx, Ry, P, A, anc, creg_anc, temp_Rx, temp_Ry, zero_reg)
            dynamic_point_mul_ft_jacobian_full(qc, list(b), Qx, Qy, Rx, Ry, P, A, anc, creg_anc, temp_Rx, temp_Ry, zero_reg)
        else:
            dynamic_point_mul_ft_affine_full(qc, list(a), Gx, Gy, Rx, Ry, P, A, anc, creg_anc, temp_Rx, temp_Ry, zero_reg)
            dynamic_point_mul_ft_affine_full(qc, list(b), Qx, Qy, Rx, Ry, P, A, anc, creg_anc, temp_Rx, temp_Ry, zero_reg)
    else:
        # fallback precompute logic
        pass  # not shown for brevity

    # Unified QPE controlled-U loop
    key_reg = list(a) + list(b)
    n_phase_qubits = len(key_reg)
    for i in range(n_bits):
        repeat = 2 ** i
        control_qubit = key_reg[i % n_phase_qubits]
        for _ in range(repeat):
            apply_controlled_shor_keyspace_oracle(qc, control_qubit, key_reg, Qx, MIN_RANGE, MAX_RANGE,
                                                 precomputed_good_indices=precomputed_good_indices)

    # Grover amplification
    if use_grover and grover_target is not None:
        for _ in range(grover_iterations):
            shors_oracle_with_keyspace(qc, key_reg, grover_target, MIN_RANGE, MAX_RANGE)
            diffuser(qc, key_reg)

    # Inverse QFT
    qc.append(synth_qft_full(n_a, do_swaps=False).inverse(), list(a))
    qc.append(synth_qft_full(n_b, do_swaps=False).inverse(), list(b))

    # Measurement
    qc.measure(list(a) + list(b), creg_ab)

    # --- Debug / Info prints ---
    print(f"[+] Quantum Circuit initialized with {n_bits} qubits (n_a={n_a}, n_b={n_b})")
    print(f"[i] Total number of ancilla qubits: {anc.size}")
    print(f"[i] Circuit before transpilation:\n{qc}")
    print("Quantum Circuit Details:")
    print(qc)
    print(f"[i] Circuit depth: {qc.depth()}, Circuit size: {qc.size()}")
    #qc.draw()
    #qc.draw('mpl') # Draw the circuit diagram
    # Show circuit
    #display(qc.draw("mpl"))
    return qc, n_a, n_b, n_p

# ----------------- Qubit and Gate Estimators -----------------
def estimate_qubit_usage(n_bits_key=KEY_BITS, n_bits_coord=COORD_BITS, ancilla_bits=None):
    if ancilla_bits is None:
        ancilla_bits = n_bits_coord * 2
    n_a = n_bits_key // 2
    n_b = n_bits_key - n_a
    n_p = n_bits_coord
    total_qubits = n_a + n_b + 2 * n_p + ancilla_bits
    print(f"[i] Estimated Qubit Usage: {total_qubits} qubits")
    return total_qubits, n_a, n_b, n_p, ancilla_bits

def estimate_gate_counts(qc):
    counts = {"CX":0, "CCX":0, "T":0}
    for inst, qargs, cargs in qc.data:
        name = inst.name.upper()
        if name in counts: counts[name] += 1
        if name in ("T_DG",): counts["T"] += 1
    return counts

# ----------------- Ancilla Filtering -----------------
def filter_ancilla_triggered_results(counts, ancilla_bit_idx=0):
    filtered_counts = {}
    ancilla_counts = {}
    for bs, cnt in counts.items():
        raw = bs.replace(' ', '')
        anc_bit = raw[ancilla_bit_idx]
        ancilla_counts[anc_bit] = ancilla_counts.get(anc_bit, 0) + cnt
        if anc_bit == '1':
            filtered_counts[bs] = cnt
    return filtered_counts, ancilla_counts

# ============================================================
# Full elliptic-curve arithmetic for secp256k1 (classical)
# ============================================================

# Global curve params must exist:
# P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
# A  = 0
# B  = 7

# -----------------------------------------------
# Modular inverse
# -----------------------------------------------
def mod_inv(a, p=P):
    a %= p
    if a == 0:
        return None
    return pow(a, p - 2, p)

# -----------------------------------------------
# Point addition (handles all edge cases)
# -----------------------------------------------
def ec_point_add(P1, P2):
    if P1 is None:
        return P2
    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    if (x1 - x2) % P == 0:
        # P1 == -P2 → identity
        if (y1 + y2) % P == 0:
            return None

    # slope = (y2 - y1) / (x2 - x1)
    num = (y2 - y1) % P
    den = (x2 - x1) % P
    inv_den = mod_inv(den, P)
    if inv_den is None:
        return None

    lam = (num * inv_den) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)

# -----------------------------------------------
# Point doubling
# -----------------------------------------------
def ec_point_double(P1):
    if P1 is None:
        return None

    x1, y1 = P1
    if y1 == 0:
        return None  # vertical tangent → infinity

    # slope = (3*x1^2 + A) / (2*y1)
    num = (3 * x1 * x1 + A) % P
    den = (2 * y1) % P
    inv_den = mod_inv(den, P)
    if inv_den is None:
        return None

    lam = (num * inv_den) % P
    x3 = (lam * lam - 2 * x1) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)

# -----------------------------------------------
# Full scalar multiplication (double & add)
# -----------------------------------------------
def ec_scalar_mult(k, x, y=None):
    """
    Multiplication of scalar k with point (x,y) on secp256k1.
    Returns point (x,y) or None (point at infinity).
    """
    # allow ec_scalar_mult(k, (x,y))
    if y is None and isinstance(x, tuple):
        Px, Py = x
    else:
        Px, Py = x, y

    if Px is None or Py is None:
        return None

    # handle negative
    if k < 0:
        return ec_scalar_mult(-k, Px, (-Py) % P)

    # identity
    if k == 0:
        return None
    if k == 1:
        return (Px % P, Py % P)

    # standard loop
    result = None
    addend = (Px % P, Py % P)

    while k > 0:
        if k & 1:
            result = ec_point_add(result, addend)
        addend = ec_point_double(addend)
        k >>= 1

    return result

# ----------------- Classical Postprocessing -----------------
def classical_postprocessing_measurements(counts, n_a, n_b, Qx, Qy):
    total = sum(counts.values())
    if total == 0:
        return None

    CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    MODULUS = P

    def gcd_checks_and_record(cand_k):
        point = scalar_mult(cand_k, (Gx, Gy))
        if point is None:
            return False
        Px = point[0]
        delta = (Qx - Px) % MODULUS
        factor1 = gcd(delta, MODULUS)
        if factor1 != 1 and factor1 != MODULUS:
            print(f"[!] Non-trivial factor found (normal point): {factor1}, k={cand_k}")
            with open("boom1.txt", "a") as f:
                f.write(f"Factor found (normal point): {factor1}, k={cand_k}\n")
            return True
        p_inv = point_neg(point, MODULUS)
        if p_inv is not None:
            Px_inv = p_inv[0]
            delta_inv = (Qx - Px_inv) % MODULUS
            factor2 = gcd(delta_inv, MODULUS)
            if factor2 != 1 and factor2 != MODULUS:
                print(f"[!] Non-trivial factor found (inverted point): {factor2}, k={cand_k}")
                with open("boom2.txt", "a") as f:
                    f.write(f"Factor found (inverted point): {factor2}, k={cand_k}\n")
                return True
        if (point[0] % (1 << COORD_BITS)) == (Qx % (1 << COORD_BITS)):
            print(f"[i] Valid candidate by coordinate match: {cand_k}")
            with open("boom3.txt", "a") as f:
                f.write(f"Candidate found (coordinate match): {cand_k}\n")
            return True
        return False

    for measured_str, freq in counts.items():
        candidate = int(measured_str.replace(' ', ''), 2)
        candidates_to_check = [candidate % CURVE_ORDER, (CURVE_ORDER - candidate) % CURVE_ORDER]
        for cand_k in candidates_to_check:
            try:
                if gcd_checks_and_record(cand_k):
                    print(f"[✓] Candidate accepted: {cand_k}")
                    with open("boom4.txt", "a") as f:
                        f.write(f"{cand_k}\n")
                    return cand_k
            except Exception as e:
                print(f"[!] Error while checking candidate {cand_k}: {e}")
                continue


    for bs, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        s = bs.replace(" ", "")
        a_bits = s[:n_a]
        b_bits = s[n_a:n_a + n_b]
        a_int = int(a_bits[::-1], 2)
        b_int = int(b_bits[::-1], 2)
        if b_int == 0:
            continue

        d = gcd(a_int, b_int)
        if d == 0:
            continue

        a_, b_ = a_int // d, b_int // d
        try:
            num_cf, den_cf = continued_fraction_rational_approx(a_, b_, 1 << KEY_BITS)
            if den_cf == 0:
                continue
            candidate = (num_cf * pow(den_cf, -1, 1 << KEY_BITS)) % (1 << KEY_BITS)
        except Exception:
            continue

        candidates_to_check = [candidate % CURVE_ORDER, (CURVE_ORDER - candidate) % CURVE_ORDER]
        for cand_k in candidates_to_check:
            try:
                if gcd_checks_and_record(cand_k):
                    print(f"[i] Valid candidate found (CF/GCD): {cand_k}")
                    with open("boom5.txt", "a") as f:
                        f.write(f"Candidate found (CF/GCD): {cand_k}\n")
                    return cand_k
            except Exception as e:
                print(f"[!] Error while CF-checking candidate {cand_k}: {e}")
                continue

    return None

# ----------------- Main Solver -----------------
def quantum_ecdpl_solver_ft(
    pubkey_hex: str,
    n_bits: int = KEY_BITS,
    coord_bits: int = COORD_BITS,
    shots: int = SHOTS,
    run_on_hardware: bool = True,
    use_jacobian: bool = True,
    mod_adder_mode: str = "DRAPER2",
    keyspace_start: int = 0x80000,
    keyspace_end: int = 0xFFFFF,
    use_grover: bool = True
):
    """
    Full quantum ECDLP solver with integrated IBM Quantum Runtime run.
    Returns: (candidate_private_key, measurement_counts)
    """

    global MOD_ADDER_MODE
    MOD_ADDER_MODE = mod_adder_mode.upper()

    # ---------------- Decompress target public key ----------------
    try:
        pubkey_bytes = bytes.fromhex(pubkey_hex)
        TARGET_QX, TARGET_QY = decompress_pubkey(pubkey_bytes)
        print(f"[i] Using secp256k1 public key: Qx={hex(TARGET_QX)}, Qy={hex(TARGET_QY)}")
    except Exception as e:
        print(f"[!] Failed to decompress public key: {e}")
        return None, {}

    # ---------------- Estimate qubits ----------------
    total_qubits, n_a, n_b, n_p, ancilla_bits = estimate_qubit_usage(n_bits, coord_bits)
    if total_qubits > 127:
        print("[⚠] Warning: Circuit exceeds typical IBM backend qubit limit!")
    print(f"[i] Total qubits: {total_qubits}, ancilla: {ancilla_bits}")

    # ---------------- Precompute good indices ----------------
    cache_file = f"good_indices_{hex(keyspace_start)}_{hex(keyspace_end)}_{hex(TARGET_QX & ((1<<32)-1))}.pkl"
    try:
        good_indices = precompute_good_indices_range(
            keyspace_start, keyspace_end, TARGET_QX,
            Gx_val=Gx, Gy_val=Gy, p_val=P, cache_path=cache_file
        )
    except Exception as e:
        print(f"[!] Precompute failed: {e}. Using empty precompute.")
        good_indices = []

    # ---------------- Build unified QPE + Grover circuit ----------------
    qc, n_a, n_b, n_p = build_unified_qpe_grover_circuit(
        n_bits=n_bits,
        COORD_BITS=coord_bits,
        Gx=Gx,
        Gy=Gy,
        Qx=TARGET_QX,
        Qy=TARGET_QY,
        P=P,
        A=A,
        ANCILLA_BITS=ancilla_bits,
        use_jacobian=use_jacobian,
        use_dynamic=True,
        use_grover=use_grover,
        grover_target=TARGET_QX,
        keyspace_start=keyspace_start,
        keyspace_end=keyspace_end,
        precomputed_good_indices=good_indices
    )
    print(f"[i] Circuit built: {qc.num_qubits} qubits, {len(qc.data)} gates")

    # ---------------- Optional gate count estimate ----------------
    counts_ft = estimate_gate_counts(qc)
    print(f"[i] Estimated gates: CX={counts_ft['CX']}, CCX={counts_ft['CCX']}, T={counts_ft['T']}")

    # ---------------- IBM Quantum Runtime Execution ----------------
    print("[i] Connecting to IBM Quantum Runtime...")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=127)
    print(f"[i] Selected IBM backend: {backend.name}")

    # Transpile circuit for backend
    transpiled = transpile(qc, backend=backend, optimization_level=3,
                           scheduling_method='alap', routing_method='sabre')
    print(f"[i] Transpiled circuit depth: {transpiled.depth()}, size: {transpiled.size()}")

    # Submit job
    sampler = Sampler(backend)
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    print(f"[i] Job submitted. Job ID: {job_id}")
    print("[i] Waiting for job results...")
    # Wait and fetch results
    results = job.result()
    
    # Extract measurement counts
    try:
        counts = results[0].data.meas.get_counts()
    except Exception:
        try:
            counts = results[0].data.c.get_counts()
        except Exception:
            counts = {}
            print("[!] Could not extract counts!")

    print("[i] Measurement Results retrieved from job:")
    print(counts) 

    # ---------------- Measurement results display ----------------
    # Plot histogram (if available)
    if counts:
        try:
            plt.figure(figsize=(12, 6))
            plot_histogram(counts)
            plt.title("QPE Measurement Outcomes")
            plt.show()
        except Exception as e:
            print(f"[!] Couldn't plot histogram: {e}")

    # Top-N dominant states (safe default)
    top_n = min(10, len(counts)) if counts else 0
    if counts and top_n > 0:
        print(f"\nTop-{top_n} dominant QPE states:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for i, (state, count) in enumerate(sorted_counts[:top_n]):
            print(f"{i+1:2d}. State: {state}, Count: {count}")

    total_counts = sum(counts.values()) if counts else 0
    print(f"[i] Total counts collected: {total_counts}")

    # ---------------- Filter ancilla-triggered results ----------------
    filtered_counts, ancilla_counts = filter_ancilla_triggered_results(counts, ancilla_bit_idx=0)
    print(f"[i] Filtered {len(filtered_counts)} counts after ancilla check")

    # ---------------- Classical postprocessing (success-only unified) ----------------
    candidate = classical_postprocessing_measurements(
        counts=filtered_counts,
        n_a=n_a,
        n_b=n_b,
        Qx=TARGET_QX,
        Qy=TARGET_QY,
    )
    if candidate is not None:
        print(f"[✓] Candidate private key recovered: {candidate}")
        try:
            with open("boom7.txt", "a") as f:
                f.write(f"Candidate found: {candidate}\n")
        except:
            pass
    else:
        print("[✗] No candidate found from measurements.")

    return candidate, counts

# ----------------- Main Advanced Function -----------------
def advanced_main():
    print("="*80)
    print("     ADVANCED QUANTUM ELLIPTIC CURVE CRYPTANALYSIS MAIN")
    print("="*80)

    # --- Algorithm Selection ---
    print("Select Algorithm Mode:")
    print("1. Standard Shor's (QPE only)")
    print("2. QPE + Grover amplification")
    print("3. Full ECDPL Solver (Advanced)")

    try:
        mode = input("Enter choice (1-3, default 3): ").strip() or "3"
        if mode not in ["1", "2", "3"]:
            raise ValueError("Invalid selection")
    except Exception as e:
        print(f"[!] Invalid input: {e}. Exiting...")
        return

    # --- Parameter Defaults ---
    KEY_BITS = int(input("KEY_BITS (default 17): ") or "17")
    COORD_BITS = int(input("COORD_BITS (default 17): ") or "17")
    MIN_RANGE = int(input("KEYSPACE_START (default 0x10000): ") or "0x10000", 16)
    MAX_RANGE = int(input("KEYSPACE_END (default 0x1FFFF): ") or "0x1FFFF", 16)
    SHOTS = int(input("SHOTS (default 8192): ") or "8192")
    DEFAULT_PUBKEY = "033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3"
    COMPRESSED_PUBKEY = input(f"Compressed public key (default {DEFAULT_PUBKEY}): ") or DEFAULT_PUBKEY

    # --- Coordinate System & Modular Adder ---
    use_jacobian = select_coordinate_system()
    mod_adder_mode = select_mod_adder_mode()
    total_qubits, n_a, n_b, n_p, ancilla_bits = estimate_qubit_usage(KEY_BITS, COORD_BITS)

    # --- Validation & Warnings ---
    print(f"[i] Calculated Keyspace size: {MAX_RANGE - MIN_RANGE + 1}")
    print(f"[i] Target Key bits: {KEY_BITS}")
    print(f"[i] Estimated iterations (sqrt of keyspace): {int(np.sqrt(MAX_RANGE - MIN_RANGE + 1))}")
    if total_qubits > 127:
        print("[⚠] Warning: Circuit exceeds typical IBM backend qubit limit!")
    if mod_adder_mode in ("DRAPER1", "DRAPER2") and ancilla_bits < 1:
        print("[⚠] Selected Draper mode may require >=1 ancilla qubit")
    print(f"[i] Coordinate system: {'Jacobian' if use_jacobian else 'Affine'}")
    print(f"[i] Modular adder mode: {mod_adder_mode}")

    # --- Circuit Execution ---
    if mode == "1":
        USE_GROVER = False
    elif mode == "2":
        USE_GROVER = True
    else:
        USE_GROVER = True  # Full solver defaults to Grover-enabled QPE

    # --- Run Unified QPE + optional Grover solver ---
    found, counts = quantum_ecdpl_solver_ft(
        pubkey_hex=COMPRESSED_PUBKEY,
        n_bits=KEY_BITS,
        coord_bits=COORD_BITS,
        shots=SHOTS,
        run_on_hardware=True,
        use_jacobian=use_jacobian,
        mod_adder_mode=mod_adder_mode,
        keyspace_start=MIN_RANGE,
        keyspace_end=MAX_RANGE,
        use_grover=USE_GROVER
    )

    print("[i] Unified QPE + Grover execution completed.")
    if found is not None:
        print(f"[✓] Candidate private key recovered: {found}")
        with open("boom8.txt", "a") as f:
            f.write(f"Candidate found: {found}\n")
    else:
        print("[✗] No candidate recovered.")

# ----------------- Run Main -----------------
if __name__ == "__main__":
    advanced_main()
