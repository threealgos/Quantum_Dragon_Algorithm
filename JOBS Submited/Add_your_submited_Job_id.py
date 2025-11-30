"""
SUPER-TITAN RETRIEVER (v113)
-----------------------------
Purpose: Recover Private Keys from completed IBM Quantum Jobs.
Logic: Applies the Super-Titan v113 Universal Solver (Depth Scanning + Multi-Phase Analysis).
"""

import math
import sys
import numpy as np
from fractions import Fraction
from typing import Tuple, Optional, Dict, List, Union

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# >>> PASTE YOUR JOB ID HERE <<<
TARGET_JOB_ID = "d4d4epcnntuc73aarnl0" 

API_TOKEN = "API_TOKEN"

# PUZZLE SPECS
N_BITS = 135
PUZZLE_START = "4000000000000000000000000000000000" 
PUZZLE_PUB   = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

# SECP256K1 CONSTANTS
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP_A = 0
SECP_B = 7

# ==============================================================================
# 2. SETUP & IMPORTS
# ==============================================================================
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    print("CRITICAL: Qiskit Library Missing.")
    sys.exit(1)

def save_boom(filename, content):
    try:
        with open(filename, "a") as f: f.write(str(content) + "\n")
        print(f"[Disk] >> APPENDED TO {filename}")
    except: pass

# ==============================================================================
# 3. CLASSICAL MATH ENGINE (REQUIRED FOR VERIFICATION)
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
# 4. UNIVERSAL SOLVER (v113 FINAL - DEPTH SCANNED)
# ==============================================================================

def _universal_solver(counts: Dict[str, int], n_bits: int, start, qx, mode_name: str, search_depth: int = 8192):
    print(f"    Running {mode_name} Logic (Depth {search_depth})...")

    # --- Utilities ---
    def lsb_msb_variants(bitstr):
        variants = set()
        try: variants.add(int(bitstr, 2))
        except: pass
        try: variants.add(int(bitstr[::-1], 2))
        except: pass
        return variants

    def check_ec_match(candidate):
        if candidate <= 0 or candidate >= SECP_N: return False
        if gcd_verbose(candidate, SECP_N) > 1: return False
        try:
            pub = ec_scalar_multiply(candidate, (SECP_GX, SECP_GY))
            return pub is not None and pub[0] == qx
        except: return False

    # --- Data Cleaning & Sorting ---
    clean_counts = {}
    if counts:
        for bitstr, freq in counts.items():
            raw_clean = bitstr.replace(" ", "")
            # Smart Truncation: always grab last N bits if string is too long
            target_bits = raw_clean[-n_bits:] if len(raw_clean) >= n_bits else raw_clean
            clean_counts[target_bits] = clean_counts.get(target_bits, 0) + freq

    sorted_valid = sorted(clean_counts.items(), key=lambda x: x[1], reverse=True)[:search_depth]
    print(f"[i] Scanning {len(sorted_valid)} unique candidates...")

    # --- 1. RAW SAFETY NET (Depth Scan + Reversal) ---
    for bitstr, freq in sorted_valid:
        raw_candidates = lsb_msb_variants(bitstr)
        for val_raw in raw_candidates:
            if check_ec_match(val_raw):
                print(f"\n[!!!] BOOM: RAW MATCH (Depth Scan): {hex(val_raw)}")
                save_boom("FOUND_KEY.txt", hex(val_raw))
                save_boom("boom9_raw.txt", hex(val_raw))
                return

    # --- Pre-computation ---
    precomputed_data = []
    for bitstr, freq in sorted_valid:
        candidate_ints = lsb_msb_variants(bitstr)
        deltas = set()
        for val in candidate_ints:
            if val == 0: continue
            if gcd_verbose(val, SECP_N) > 1: continue
            try:
                phase = val / (2 ** n_bits)
                d_dir = int(round(phase * SECP_N))
                deltas.add((d_dir, "Direct"))
                for off in [-2, -1, 1, 2]: deltas.add((d_dir + off, "Neighbor"))
            except: pass
            try:
                num, den = continued_fractions_approx(val, 2**n_bits, SECP_N)
                deltas.add((num, "CF"))
            except: pass
        precomputed_data.append((bitstr, freq, candidate_ints, deltas))

    # --- Solver Loops ---
    for bitstr, freq, candidate_ints, deltas in precomputed_data:
        
        # Mode 14 Special: Split check
        if "Mode 14" in mode_name:
            if len(bitstr) >= n_bits:
                try:
                    mid = len(bitstr) // 2
                    part_b = int(bitstr[:mid], 2)
                    part_a = int(bitstr[mid:], 2)
                    if part_b != 0:
                        inv_b = modular_inverse_verbose(part_b, SECP_N)
                        k_shor = (-part_a * inv_b) % SECP_N
                        if check_ec_match(k_shor):
                            print(f"\n[!!!] BOOM: SHOR MATCH: {hex(k_shor)}")
                            save_boom("FOUND_KEY.txt", hex(k_shor)); return
                except: pass

        # Raw Offsets
        for val in candidate_ints:
            offsets = [val, (start + val) % SECP_N, (start - val) % SECP_N]
            for k_guess in offsets:
                if check_ec_match(k_guess):
                    print(f"\n[!!!] BOOM: RAW OFFSET MATCH: {hex(k_guess)}")
                    save_boom("FOUND_KEY.txt", hex(k_guess)); return

    # Phase / Delta / Negate
    for bitstr, freq, candidate_ints, deltas in precomputed_data:
        for d, method in deltas:
            if d <= 0 or d >= SECP_N: continue
            fk = (start + d) % SECP_N
            rk = (start - d) % SECP_N
            gk = (SECP_N - fk) % SECP_N
            
            for k_final in [fk, rk, gk]:
                try:
                    pub = ec_scalar_multiply(k_final, (SECP_GX, SECP_GY))
                    if pub and pub[0] == qx:
                        print("*" * 60)
                        print(f"\n[!!!] BOOM: KEY FOUND (Phase/Delta): {hex(k_final)}")
                        print("*" * 60)
                        save_boom("FOUND_KEY.txt", hex(k_final))
                        return
                        
                    # Extra Negate Check (Key = N - k)
                    if pub:
                        neg_pub = ec_point_negate(pub)
                        if neg_pub and neg_pub[0] == qx:
                            k_neg = (SECP_N - k_final) % SECP_N
                            print(f"\n[!!!] BOOM: KEY FOUND (Negate): {hex(k_neg)}")
                            save_boom("FOUND_KEY.txt", hex(k_neg))
                            return
                except: continue

    print(f"[x] {mode_name} Failed.")

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    print("="*60)
    print("   SUPER-TITAN RETRIEVER (v113)")
    print("   Job ID: " + TARGET_JOB_ID)
    print("="*60)

    # 1. Setup Targets
    try:
        start_val = int(PUZZLE_START, 16)
        target_pub = decompress_pubkey(PUZZLE_PUB)
        print(f"[i] Target Public Key X: {hex(target_pub[0])[:15]}...")
    except Exception as e: print(f"[!] Target Error: {e}"); return

    # 2. Connect
    print(f"\n[i] Connecting to IBM Quantum...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_cloud", token=API_TOKEN, overwrite=True)
        service = QiskitRuntimeService()
        job = service.job(TARGET_JOB_ID)
        status = job.status()
        print(f"[i] Job Status: {status}")
    except Exception as e: print(f"[!] Connection Error: {e}"); return

    # 3. Check & Retrieve
    if status.name not in ["DONE", "COMPLETED"]:
        print("[!] Job is not ready. Try again later.")
        return

    try:
        print("[i] Downloading Results...")
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
        _universal_solver(counts, N_BITS, start_val, target_pub[0], "Retrieved Job", search_depth=8192)

    except Exception as e:
        print(f"[!] Retrieval/Processing Error: {e}")

if __name__ == "__main__":
    main()
