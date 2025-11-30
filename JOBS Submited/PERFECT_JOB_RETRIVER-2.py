"""
IBM QUANTUM JOB RETRIEVAL & DUAL-MODE POST-PROCESSING
-----------------------------------------------------
Job ID: d4d4epcnntuc73aarnl0
Status: Retrieval Only (No Submission)
"""

import math
from math import gcd
from fractions import Fraction
from typing import Tuple, Optional, Dict
from qiskit_ibm_runtime import QiskitRuntimeService

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

JOB_ID = "d4d4epcnntuc73aarnl0"
API_TOKEN = "API_TOKEN"

# PUZZLE SPECS
N_COUNTING_BITS = 135 
PUZZLE_START_HEX  = "4000000000000000000000000000000000" 
TARGET_PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

# SECP256K1 CONSTANTS
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_A = 0
SECP_B = 7
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# ==============================================================================
# 2. CLASSICAL MATH ENGINE
# ==============================================================================

def extended_euclidean(a, b):
    if a == 0: return b, 0, 1
    g, y, x = extended_euclidean(b % a, a)
    return g, x - (b // a) * y, y

def modular_inverse(a, m):
    g, x, y = extended_euclidean(a, m)
    if g != 1: return 0
    return x % m

def tonelli_shanks_sqrt(n, p):
    if pow(n, (p-1)//2, p) != 1: return 0
    if p % 4 == 3: return pow(n, (p+1)//4, p)
    s, e = p - 1, 0
    while s % 2 == 0: s //= 2; e += 1
    z = 2
    while pow(z, (p-1)//2, p) != -1: z += 1
    x = pow(n, (s+1)//2, p)
    b, g, r = pow(n, s, p), pow(z, s, p), e
    while True:
        t, m = b, 0
        for m in range(r):
            if t == 1: break
            t = pow(t, 2, p)
        if m == 0: return x
        gs = pow(g, 2**(r-m-1), p)
        g = (gs*gs)%p; x = (x*gs)%p; b = (b*g)%p; r = m

def ec_point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1; x2, y2 = p2
    if x1 == x2 and (y1 + y2) % SECP_P == 0: return None
    if x1 == x2 and y1 == y2:
        lam = ((3*x1**2 + SECP_A) * modular_inverse(2*y1, SECP_P)) % SECP_P
    else:
        lam = ((y2-y1) * modular_inverse(x2-x1, SECP_P)) % SECP_P
    x3 = (lam**2 - x1 - x2) % SECP_P
    y3 = (lam*(x1-x3) - y1) % SECP_P
    return (x3, y3)

def ec_scalar_multiply(k, point):
    if k == 0 or point is None: return None
    res, addend = None, point
    for bit in reversed(bin(k % SECP_N)[2:]):
        res = ec_point_add(res, res)
        if bit == '1': res = ec_point_add(res, addend)
    return res

def ec_point_negate(p):
    if p is None: return None
    return (p[0], (-p[1])%SECP_P)

def decompress_pubkey(hex_key):
    try:
        hex_key = hex_key.replace("0x", "").lower().strip()
        prefix = int(hex_key[:2], 16)
        x = int(hex_key[2:], 16)
        y_sq = (pow(x, 3, SECP_P) + SECP_B) % SECP_P
        y = tonelli_shanks_sqrt(y_sq, SECP_P)
        if prefix == 2 and y % 2 != 0: y = SECP_P - y
        if prefix == 3 and y % 2 == 0: y = SECP_P - y
        return (x, y)
    except: raise ValueError("Invalid Key")

def continued_fractions_approx(num, den, max_den):
    if den == 0: return 0, 1
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

# ==============================================================================
# 3. DATA EXTRACTION
# ==============================================================================

def extract_counts(result) -> Dict[str, int]:
    try:
        # Attempt various extraction paths for robustness
        if hasattr(result[0], 'data'):
            d = result[0].data
            if hasattr(d, 'phase_bits'): return d.phase_bits.get_counts()
            if hasattr(d, 'meas'): return d.meas.get_counts()
            if hasattr(d, 'c'): return d.c.get_counts()
        if hasattr(result[0], 'get_counts'): return result[0].get_counts()
    except: pass
    # Fallback for SamplerV1
    try:
        bp = result[0].data.eigenstate.binary_probabilities()
        return {k: int(v * 10000) for k, v in bp.items()}
    except: return {}

# ==============================================================================
# 4. DUAL MODE SOLVERS
# ==============================================================================

def solve_ecdlp(counts, n_bits, start, qx):
    print(f"\n[Analysis] Mode 1: ECDLP Verification...")
    clean = {k.replace(" ", ""): v for k, v in counts.items()}
    valid = {}
    
    # Symmetry Filter (if available)
    for k, v in clean.items():
        if len(k) >= 2*n_bits:
            if k[:n_bits] == k[n_bits:2*n_bits]: valid[k[n_bits:]] = v
        else: valid[k] = v
        
    if not valid: valid = clean
    sorted_valid = sorted(valid.items(), key=lambda x:x[1], reverse=True)
    
    for bitstr, _ in sorted_valid[:20]:
        try: val = int(bitstr, 2)
        except: 
            try: val = int(bitstr[::-1], 2)
            except: continue
            
        phase = val / (2**n_bits)
        if phase == 0: continue
        
        d_cands = [int(round(phase * SECP_N))]
        num, _ = continued_fractions_approx(val, 2**n_bits, SECP_N)
        d_cands.append(num)
        
        for d in set(d_cands):
            if d <= 0: continue
            fk = (start + d) % SECP_N
            try:
                pub = ec_scalar_multiply(fk, (SECP_GX, SECP_GY))
                if pub and pub[0] == qx:
                    print(f"\n[SUCCESS] ECDLP CRACKED: {hex(fk)}")
                    return fk
                if pub:
                    neg = ec_point_negate(pub)
                    if neg[0] == qx:
                        print(f"\n[SUCCESS] ECDLP CRACKED (Inverted): {hex((SECP_N-fk)%SECP_N)}")
                        return (SECP_N-fk)%SECP_N
            except: continue
    print("[x] ECDLP Failed.")
    return None

def solve_shor(counts, n_bits, N, g):
    print(f"\n[Analysis] Mode 2: Shor's Verification (N={N})...")
    clean = {k.replace(" ", ""): v for k, v in counts.items()}
    for bitstr, _ in sorted(clean.items(), key=lambda x:x[1], reverse=True)[:20]:
        try: val = int(bitstr[-n_bits:], 2)
        except: continue
        if val == 0: continue
        
        _, r = continued_fractions_approx(val, 2**n_bits, N)
        if r % 2 != 0: continue
        
        half_r = r // 2
        try: x = pow(g, half_r, N)
        except: continue
        if x == 1 or x == N-1: continue
        
        p = gcd(x-1, N)
        q = gcd(x+1, N)
        if p > 1 and p < N:
            print(f"\n[SUCCESS] FACTOR: {p}")
            return p
        if q > 1 and q < N:
            print(f"\n[SUCCESS] FACTOR: {q}")
            return q
    print("[x] Shor's Failed.")
    return None

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    print("="*60)
    print("   IBM QUANTUM JOB RETRIEVER & SOLVER")
    print("="*60)
    
    # Setup Targets
    try:
        start_val = int(PUZZLE_START_HEX, 16)
        qx, qy = decompress_pubkey(TARGET_PUBKEY_HEX)
    except Exception as e: print(f"[!] Target Error: {e}"); return

    # Connect & Fetch
    print(f"\n[i] Connecting to Job {JOB_ID}...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=API_TOKEN, overwrite=True)
        service = QiskitRuntimeService()
        job = service.job(JOB_ID)
        status = job.status()
        print(f"[i] Job Status: {status}")
    except Exception as e: print(f"[!] Connection Error: {e}"); return

    # Check if ready
    completed = ["DONE", "COMPLETED", "SUCCEEDED"]
    status_str = getattr(status, "name", str(status)).upper()
    
    if status_str not in completed:
        print("[!] Job is not finished yet.")
        return

    # Retrieve
    try:
        result = job.result()
        counts = extract_counts(result)
        if not counts: print("[!] No counts found."); return
        print(f"[i] Extracted {len(counts)} outcomes.")
        
        # Selector
        print("\n[1] ECDLP (Bitcoin)")
        print("[2] Shor's (Factoring)")
        mode = input("Select Mode: ").strip()
        
        if mode == '1':
            solve_ecdlp(counts, N_COUNTING_BITS, start_val, qx)
        else:
            try:
                N = int(input("Enter N: "))
                g = int(input("Enter g: "))
                solve_shor(counts, N_COUNTING_BITS, N, g)
            except: print("[!] Invalid Input")
            
    except Exception as e:
        print(f"[!] Retrieval Error: {e}")

if __name__ == "__main__":
    main()