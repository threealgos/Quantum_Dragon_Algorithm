"""
DRAGON JOB RETRIEVER (v120 COMPATIBLE) - FULLY INTERACTIVE EDITION
----------------------------------------------------------
Purpose: Recover Private Keys from completed IBM Quantum Jobs.
Logic: Hybrid Post-Processing (Shor AB + IPE Phase + Window Scan).
Updates: interactive inputs for JobID, CRN, Bits, PubKey, and Keyspace_Start.
"""

import math
import sys
import time
import logging
from collections import defaultdict
from fractions import Fraction
from typing import Tuple, Optional, Dict, List, Union

# ==============================================================================
# 1. HARDCODED DEFAULTS (SUGGESTIONS)
# ==============================================================================
# These values are used if you press ENTER at the prompts.

DEFAULT_JOB_ID = "YOUR_DEFAULT_JOB_ID_HERE"
DEFAULT_BITS = 135
DEFAULT_PUBKEY = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
DEFAULT_START_HEX = "0x4000000000000000000000000000000000"

# REQUIRED: Your IBM Cloud API Token (Must be edited in code)
API_TOKEN = "YOUR_IBM_TOKEN_HERE"

# ==============================================================================
# 2. SECP256K1 CONSTANTS & SETUP
# ==============================================================================
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
ORDER = N
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
A = 0
B = 7

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from ecdsa.ellipticcurve import Point, CurveFp
    from ecdsa import numbertheory
except ImportError:
    print("CRITICAL: Libraries Missing. Install qiskit-ibm-runtime and ecdsa.")
    sys.exit(1)

CURVE = CurveFp(P, A, B)
G = Point(CURVE, Gx, Gy)

# ==============================================================================
# 3. CLASSICAL MATH ENGINE
# ==============================================================================

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

def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    return point * k

def continued_fractions_approx(num, den, max_den):
    if den == 0: return 0, 1
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

def precompute_good_indices_range(start, end, target_qx, gx=G.x(), gy=G.y()):
    """
    Classical Window Check (The 'Extra Check' from Dragon Mode)
    """
    if end - start > 200000: return [] 
    good = []
    
    P0 = G * start
    baseG = G
    current = P0
    
    for k in range(start, end + 1):
        if current.x() == target_qx:
            good.append(k - start)
        current = current + baseG
    return good

def save_key(k: int):
    hex_k = hex(k)[2:].zfill(64)
    padded_hex = '0x' + hex_k.zfill(64)
    zero_padded = hex_k.zfill(64)
    shifted_hex = '0x' + zero_padded[32:] + zero_padded[:32]
    filename = "boom_retrieved.txt"
    try:
        with open(filename, "a") as f:
            f.write(f"--------------------------------------------------\n")
            f.write(f"FOUND KEY: {padded_hex}\n")
            f.write(f"{padded_hex}\n{zero_padded}\n{shifted_hex}\n")
            f.write(f"--------------------------------------------------\n")
        logger.info(f"KEY SAVED: {filename}")
        print(f"\n[!!!] KEY FOUND: {padded_hex}")
    except: pass

# ==============================================================================
# 4. DRAGON RETRIEVAL LOGIC
# ==============================================================================

def safe_get_counts(result_item):
    """
    Aggressive Universal Retrieval (Reflection + Legacy).
    Matches Dragon Mode v120 logic exactly.
    """
    combined_counts = defaultdict(int)
    data_found = False

    # 1. Reflection (Modern SamplerV2)
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
        lambda: result_item.data.m0.get_counts(),
        lambda: result_item.get_counts()
    ]
    for attempt in attempts:
        try: return attempt()
        except: continue
    return None

def hybrid_post_process(counts, bits, order, start, target_pub_x, target_pub_y):
    """
    Dragon Mode Hybrid Solver:
    - Checks Shor (a, b) pattern.
    - Checks IPE (Phase) pattern.
    - Checks Y-coordinate symmetry.
    """
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:500] 
    
    print(f"[i] Analyzing top {len(sorted_counts)} measurement patterns...")

    for meas_str, freq in sorted_counts:
        meas_str = meas_str.replace(" ", "")
        
        # --- ATTEMPT 1: SHOR (AB) MODE LOGIC ---
        if len(meas_str) >= 2:
            mid = len(meas_str) // 2
            try:
                a = int(meas_str[:mid], 2)
                b = int(meas_str[mid:], 2)
                if gcd_verbose(b, order) == 1:
                    inv_b = modular_inverse_verbose(b, order)
                    k = (-a * inv_b) % order
                    
                    cand_pub = G * ((start + k) % order)
                    if cand_pub.x() == target_pub_x:
                        key = (start + k) % order if cand_pub.y() == target_pub_y else (order - (start + k) % order)
                        logger.info(f"FOUND VIA SHOR LOGIC: {hex(key)}")
                        save_key(key); return key
            except: pass
        
        # --- ATTEMPT 2: IPE / PHASE MODE LOGIC ---
        measurements = [int(bit) for bit in meas_str if bit in '01']
        if len(measurements) >= bits: measurements = measurements[:bits]
        
        if len(measurements) > 0:
            measurements = measurements[::-1] 
            
            phi = sum([b * (1 / 2**(i+1)) for i, b in enumerate(measurements)])
            num, den = continued_fractions_approx(int(phi * 2**bits), 2**bits, order)
            d = (num * modular_inverse_verbose(den, order)) % order if den and gcd_verbose(den, order) == 1 else None
            
            if d:
                cand = (start + d) % N
                cand_pub = G * cand
                if cand_pub.x() == target_pub_x:
                    key = cand if cand_pub.y() == target_pub_y else (order - cand)
                    logger.info(f"FOUND VIA IPE PHASE: {hex(key)}")
                    save_key(key); return key
    return None

# ==============================================================================
# 5. MAIN INTERACTIVE EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("   DRAGON JOB RETRIEVER - COMPATIBLE INTERACTIVE MODE")
    print("   Recovering Private Key from IBM Quantum Job")
    print("   Press ENTER to accept the [Default Value]")
    print("="*60 + "\n")

    # -------------------------------------------------------------
    # 1. JOB ID
    # -------------------------------------------------------------
    print(f"Default Job ID: {DEFAULT_JOB_ID}")
    user_job = input(">> Paste JOB ID: ").strip()
    TARGET_JOB_ID = user_job if user_job else DEFAULT_JOB_ID
    print(f"   -> Selected: {TARGET_JOB_ID}\n")

    # -------------------------------------------------------------
    # 2. CRN (OPTIONAL)
    # -------------------------------------------------------------
    print("CRN (Cloud Resource Name) - Optional")
    user_crn = input(">> Paste CRN: ").strip()
    if user_crn:
        print(f"   -> Using Instance: {user_crn[:15]}...\n")
    else:
        print(f"   -> Skipping CRN (Auto/Default Instance)\n")

    # -------------------------------------------------------------
    # 3. BIT SIZE
    # -------------------------------------------------------------
    print(f"Bit Size [Default: {DEFAULT_BITS}]")
    user_bits = input(">> Bits: ").strip()
    if user_bits:
        try:
            BITS = int(user_bits)
        except ValueError:
            print(f"   [!] Invalid number. Reverting to default {DEFAULT_BITS}")
            BITS = DEFAULT_BITS
    else:
        BITS = DEFAULT_BITS
    print(f"   -> Selected Bits: {BITS}\n")

    # -------------------------------------------------------------
    # 4. PUBLIC KEY
    # -------------------------------------------------------------
    print(f"Target Public Key (Compressed Hex) [Default: ...{DEFAULT_PUBKEY[-10:]}]")
    user_pub = input(">> PubKey: ").strip()
    COMPRESSED_PUBKEY_HEX = user_pub if user_pub else DEFAULT_PUBKEY
    print(f"   -> Selected PubKey: ...{COMPRESSED_PUBKEY_HEX[-16:]}\n")

    # -------------------------------------------------------------
    # 5. KEYSPACE START
    # -------------------------------------------------------------
    print(f"Keyspace Start Hex [Default: {DEFAULT_START_HEX}]")
    user_start = input(">> Start: ").strip()
    if user_start:
        try:
            # Handle with or without '0x'
            KEYSPACE_START = int(user_start, 16)
        except ValueError:
            print(f"   [!] Invalid Hex. Reverting to default.")
            KEYSPACE_START = int(DEFAULT_START_HEX, 16)
    else:
        KEYSPACE_START = int(DEFAULT_START_HEX, 16)
    
    print(f"   -> Selected Start: {hex(KEYSPACE_START)}\n")
    print("-" * 60)

    # =============================================================
    # EXECUTION PHASE
    # =============================================================

    # 1. Setup Targets
    try:
        target_pub = decompress_pubkey(COMPRESSED_PUBKEY_HEX)
        print(f"[i] Target Public Key X: {hex(target_pub.x())[:15]}...")
    except Exception as e: 
        print(f"[!] Target Error (Check PubKey): {e}")
        return

    # 2. Connect
    print(f"[i] Connecting to IBM Quantum...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_cloud", token=API_TOKEN, overwrite=True)
        
        # CRN Logic
        if user_crn:
            service = QiskitRuntimeService(channel="ibm_cloud", token=API_TOKEN, instance=user_crn)
        else:
            service = QiskitRuntimeService(channel="ibm_cloud", token=API_TOKEN)
            
        job = service.job(TARGET_JOB_ID)
        status = job.status()
        print(f"[i] Job Found. Status: {status}")
    except Exception as e: 
        print(f"[!] Connection Error: {e}")
        print("    Tip: Check API Token, Job ID, or CRN.")
        return

    # 3. Check & Retrieve
    if status.name not in ["DONE", "COMPLETED"]:
        print("[!] Job is not ready. Try again later.")
        return

    try:
        print("[i] Downloading Results...")
        result = job.result()
        item = result[0] if isinstance(result, list) else result

        # Dragon Universal Extraction
        counts = safe_get_counts(item)

        if not counts:
            print("[!] Could not extract counts from result object.")
            return

        print(f"[i] Extracted {len(counts)} unique bitstrings.")
        
        # 4. EXTRA CHECK: Post-Quantum Window Scan
        top_meas = max(counts, key=counts.get)
        clean_meas = "".join([b for b in top_meas.replace(" ", "") if b in '01'])
        
        if clean_meas:
            measured_int = int(clean_meas, 2)
            logger.info(f"[Extra Check] Scanning window around measurement: {hex(measured_int)}")
            # Check 10,000 range
            hits = precompute_good_indices_range(measured_int, measured_int + 10000, target_pub.x())
            if hits:
                final_key = measured_int + hits[0]
                logger.info(f"FOUND VIA POST-QUANTUM WINDOW CHECK: {hex(final_key)}")
                save_key(final_key)
                return

        # 5. Hybrid Solver
        print("[i] Running Hybrid Solver...")
        k = hybrid_post_process(counts, BITS, ORDER, KEYSPACE_START, target_pub.x(), target_pub.y())
        
        if not k:
            print("[x] No key found in top candidates.")
            print(f"    - Checked Keyspace Start: {hex(KEYSPACE_START)}")

    except Exception as e:
        print(f"[!] Retrieval/Processing Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
