"""
THE OMEGA RETRIEVER (v22): JOB FETCH & SOLVE
--------------------------------------------
Target: 135-bit SECP256k1
Function: Fetches results from a previous run and applies 
          the full Omega Post-Processing suite.

COMPATIBILITY:
  - Works with Omega v22 (Hive, Phantom, Shor, GHZ, Shadow).
  - Handles SamplerV2 data formats.
  - Includes full 10-Boom Safety System.
"""

import math
import sys
import numpy as np
from math import gcd
from fractions import Fraction
from typing import Tuple, Optional, Dict, List, Union

# --- IMPORT ---
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError as e:
    print("CRITICAL: qiskit-ibm-runtime not installed.")
    sys.exit(1)

# ==============================================================================
# 0. HELPER: BOOM SAVER
# ==============================================================================
def save_boom(filename, content):
    try:
        with open(filename, "w") as f: f.write(str(content))
        print(f"[Disk] >> SAVED TO {filename}")
    except: pass

# ==============================================================================
# 1. CONFIGURATION (PASTE YOUR JOB ID HERE)
# ==============================================================================

# >>> PASTE JOB ID HERE <<<
JOB_ID = "d4d4epcnntuc73aarnl0" 

API_TOKEN = "API_TOKEN"

# Must match the submission bits (Default: 135)
# If the submission said "[!] Scaling to 127 bits", change this to 127.
N_COUNTING_BITS = 135 

# SECP256K1 CONSTANTS
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_A = 0
SECP_B = 7
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

PUZZLE_START_HEX  = "4000000000000000000000000000000000" 
TARGET_PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

# ==============================================================================
# 2. CLASSICAL MATH ENGINE
# ==============================================================================

def extended_euclidean(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0: return b, 0, 1
    g, y, x = extended_euclidean(b % a, a)
    return g, x - (b // a) * y, y

def modular_inverse(a: int, m: int) -> int:
    g, x, y = extended_euclidean(a, m)
    if g != 1: return 0
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
    if x1 == x2 and (y1 + y2) % SECP_P == 0: return None
    if x1 == x2 and y1 == y2:
        if y1 == 0: return None
        lam = ((3 * x1**2 + SECP_A) * modular_inverse(2 * y1, SECP_P)) % SECP_P
    else:
        if x1 == x2: return None
        lam = ((y2 - y1) * modular_inverse(x2 - x1, SECP_P)) % SECP_P
    x3 = (lam**2 - x1 - x2) % SECP_P
    y3 = (lam * (x1 - x3) - y1) % SECP_P
    return (x3, y3)

def ec_scalar_multiply(k, point):
    if k == 0 or point is None: return None
    res, addend = None, point
    k_eff = k % SECP_N
    for bit in reversed(bin(k_eff)[2:]):
        res = ec_point_add(res, res)
        if bit == '1': res = ec_point_add(res, addend)
    return res

def ec_point_negate(point):
    if point is None: return None
    return (point[0], (-point[1]) % SECP_P)

def ec_point_subtract(p1, p2):
    return ec_point_add(p1, ec_point_negate(p2))

def decompress_pubkey(hex_key):
    try:
        hex_key = hex_key.replace("0x", "").lower().strip()
        prefix = int(hex_key[:2], 16)
        x = int(hex_key[2:], 16)
        y_sq = (pow(x, 3, SECP_P) + SECP_B) % SECP_P
        y = tonelli_shanks_sqrt(y_sq, SECP_P)
        if y == 0 and y_sq != 0: raise ValueError
        if prefix == 2 and y % 2 != 0: y = SECP_P - y
        if prefix == 3 and y % 2 == 0: y = SECP_P - y
        return (x, y)
    except: raise ValueError("Invalid Hex Key")

def continued_fractions_approx(num, den, max_den):
    if den == 0: return 0, 1
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

# ==============================================================================
# 3. DATA EXTRACTION (ROBUST)
# ==============================================================================

def extract_counts_from_result(result) -> Dict[str, int]:
    """Attempts to pull counts from whatever object structure IBM returned."""
    if isinstance(result, dict): return result
    
    # Try SamplerV2 PubResult
    try:
        if hasattr(result[0], 'data'):
            d = result[0].data
            # Check all known registers from Omega v22
            if hasattr(d, 'meas'): return d.meas.get_counts()
            if hasattr(d, 'meas_count'): return d.meas_count.get_counts()
            if hasattr(d, 'phase_bits'): return d.phase_bits.get_counts()
            if hasattr(d, 'c'): return d.c.get_counts()
            
            # Mode 5 (Hive) might store it in w0_ph
            if hasattr(d, 'w0_ph'): return d.w0_ph.get_counts()
    except: pass

    # Try Generic
    try:
        if hasattr(result[0], 'get_counts'): return result[0].get_counts()
    except: pass
    
    return {}

# ==============================================================================
# 4. THE UNIVERSAL SOLVER (WITH HIVE + BOOMS)
# ==============================================================================

def _universal_solver(counts: Dict[str, int], n_bits: int, start, qx, mode_name):
    print(f"    Running {mode_name} Logic...")
    
    # [BOOM 9] RAW SAFETY NET
    if counts:
        # Get the most frequent string
        top = max(counts, key=counts.get).replace(" ", "")
        
        # Handle Mode 5 (Hive) concatenation: grab the first worker's phase
        if "Mode 5" in mode_name and len(top) > n_bits:
            top = top[-n_bits:]
        elif len(top) > n_bits: 
            # Default slice for other modes if extra bits present
            top = top[-n_bits:]
            
        try: save_boom("boom9_raw.txt", f"Raw: {int(top, 2)}")
        except: pass

    clean = {k.replace(" ", ""): v for k, v in counts.items()}
    valid_aggregated = {} 
    
    # --- DATA PARSING & FILTERING ---
    for bitstr, freq in clean.items():
        # MODE 5: HIVE AGGREGATION
        if "Mode 5" in mode_name:
            # Scan the massive bitstring for [Verify][Phase] chunks
            chunk_size = 2 * n_bits
            num_chunks = len(bitstr) // chunk_size
            for i in range(num_chunks):
                chunk = bitstr[i*chunk_size : (i+1)*chunk_size]
                p_v = chunk[0:n_bits]
                p_c = chunk[n_bits:]
                if p_v == p_c: 
                    valid_aggregated[p_c] = valid_aggregated.get(p_c, 0) + freq
            continue
            
        # MODE 4 FILTER (Verified Shadow)
        if "Mode 4" in mode_name:
            if len(bitstr) >= n_bits + 2:
                # [State][Verify][Count] or [Verify][State][Count]
                # Qiskit V2 usually puts 'meas_state' (2 bits) then 'meas_count'
                if bitstr[0:2] not in ["00", "11"]: continue
                kept = bitstr[2:]
                valid_aggregated[kept] = valid_aggregated.get(kept, 0) + freq
            continue
        
        # MODE 2/3 (Advanced/Shor) usually output clean counts or [Check][Phase]
        # If Check exists, verify it.
        if len(bitstr) >= 2 * n_bits and ("Mode 2" in mode_name or "Mode 3" in mode_name):
             # Assuming [Check][Phase]
             p_chk = bitstr[:n_bits]
             p_phs = bitstr[n_bits:]
             if p_chk == p_phs:
                 valid_aggregated[p_phs] = valid_aggregated.get(p_phs, 0) + freq
             continue

        # DEFAULT (Mode 1, etc)
        valid_aggregated[bitstr] = valid_aggregated.get(bitstr, 0) + freq

    if not valid_aggregated:
        print("[!] Filter too strict (or raw data). Using Raw.")
        valid_aggregated = clean

    # --- SOLVER CORE ---
    sorted_valid = sorted(valid_aggregated.items(), key=lambda x:x[1], reverse=True)
    
    for bitstr, freq in sorted_valid[:25]:
        int_cands = []
        try: int_cands.append((int(bitstr, 2), "Big")) 
        except: pass
        try: int_cands.append((int(bitstr[::-1], 2), "Little")) 
        except: pass
        
        for val, endian in set(int_cands):
            phase = val / (2 ** n_bits)
            if phase == 0: continue
            
            d_list = []
            d_dir = int(round(phase * SECP_N))
            
            # [Boom 10 Candidates]
            for off in [-2,-1,1,2]: d_list.append((d_dir+off, "Neighbor"))
            
            # [Boom 7 Candidate]
            num, _ = continued_fractions_approx(val, 2**n_bits, SECP_N)
            d_list.append((num, "CF"))
            
            # [Boom 1 Candidate]
            d_list.append((d_dir, "Direct"))
            
            for d, method in d_list:
                if d <= 0 or d >= SECP_N: continue
                fk = (start + d) % SECP_N
                try:
                    pub = ec_scalar_multiply(fk, (SECP_GX, SECP_GY))
                    if pub:
                        if pub[0] == qx:
                            print(f"\n[SUCCESS] MATCH: {hex(fk)}")
                            
                            # --- SAVE ALL RELEVANT BOOMS ---
                            save_boom("boom1.txt", hex(fk)) # Always save Boom 1
                            
                            if endian == "Little": 
                                save_boom("boom6.txt", hex(fk))
                            
                            if method == "CF": 
                                save_boom("boom7.txt", hex(fk))
                            
                            if method == "Neighbor": 
                                save_boom("boom10.txt", hex(fk))
                                
                            if "Mode 5" in mode_name:
                                save_boom("boom_hive.txt", hex(fk))
                                
                            return fk
                            
                        neg = ec_point_negate(pub)
                        if neg[0] == qx:
                            rk = (SECP_N - fk) % SECP_N
                            print(f"\n[SUCCESS] MATCH (INV): {hex(rk)}")
                            # [Boom 2]
                            save_boom("boom2.txt", hex(rk))
                            return rk
                except: continue
    print(f"[x] {mode_name} Failed.")
    return None

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    print("="*60)
    print("   THE OMEGA RETRIEVER (v22)")
    print("   Job Fetch & Solve Utility")
    print("="*60)

    # 1. Setup Targets
    try:
        start_val = int(PUZZLE_START_HEX, 16)
        qx, qy = decompress_pubkey(TARGET_PUBKEY_HEX)
    except Exception as e: print(f"[!] Target Error: {e}"); return

    # 2. Connect & Fetch
    print(f"\n[i] Connecting to Job {JOB_ID}...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=API_TOKEN, overwrite=True)
        service = QiskitRuntimeService()
        job = service.job(JOB_ID)
        status = job.status()
        print(f"[i] Job Status: {status}")
    except Exception as e: print(f"[!] Connection Error: {e}"); return

    # 3. Check Status
    completed = ["DONE", "COMPLETED", "SUCCEEDED"]
    status_str = getattr(status, "name", str(status)).upper()
    
    if status_str not in completed:
        print("[!] Job is not finished yet.")
        return

    # 4. Retrieve
    try:
        result = job.result()
        counts = extract_counts_from_result(result)
        if not counts: print("[!] No counts found."); return
        print(f"[i] Extracted {len(counts)} unique outcomes.")
        
        # 5. Mode Selection (Matching Omega v22 Submission)
        print("\nSelect Original Submission Mode:")
        print(" [1] PHANTOM")
        print(" [2] SHOR")
        print(" [3] GHZ")
        print(" [4] SHADOW")
        print(" [5] HIVE")
        
        mode = input("Choice [1-5]: ").strip()
        
        # Run Solver
        final_key = _universal_solver(counts, N_COUNTING_BITS, start_val, qx, f"Mode {mode}")
        
        if final_key:
            save_boom("boom5.txt", str(final_key))
            
    except Exception as e:
        print(f"[!] Retrieval Error: {e}")

if __name__ == "__main__":
    main()