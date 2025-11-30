"""
IBM QUANTUM JOB RETRIEVAL & DUAL-MODE POST-PROCESSING (FINAL PRECISE VERSION)
-----------------------------------------------------------------------------
Job ID: d4d4epcnntuc73aarnl0
Use this script to retrieve results from a job submitted previously.
It handles:
 1. Authentication & Connection
 2. Job Status Checking (Queued/Running/Done)
 3. Data Extraction (Robust against Sampler versions)
 4. Mathematical Solving (ECDLP or Shor's Logic)
"""

import math
from math import gcd
from fractions import Fraction
from typing import Tuple, Optional, Dict, List
from qiskit_ibm_runtime import QiskitRuntimeService

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

JOB_ID = "d4d4epcnntuc73aarnl0"
API_TOKEN = "API_TOKEN"

# --- PUZZLE SPECS ---
# This MUST match the 'op_bits' used in the submission.
# If the submission scaled down to 127 bits, set this to 127.
# Defaulting to 135 as per original target.
N_COUNTING_BITS = 135 

PUZZLE_START_HEX  = "4000000000000000000000000000000000" 
TARGET_PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

# --- SECP256K1 CONSTANTS ---
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_A = 0
SECP_B = 7
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# ==============================================================================
# 2. CLASSICAL MATH ENGINE (HIGH PRECISION)
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

def ec_point_add(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
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

def ec_scalar_multiply(k: int, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if k == 0 or point is None: return None
    res, addend = None, point
    k_eff = k % SECP_N
    for bit in reversed(bin(k_eff)[2:]):
        res = ec_point_add(res, res)
        if bit == '1': res = ec_point_add(res, addend)
    return res

def ec_point_negate(point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if point is None: return None
    return (point[0], (-point[1]) % SECP_P)

def decompress_pubkey(hex_key: str) -> Tuple[int, int]:
    try:
        hex_key = hex_key.replace("0x", "").lower().strip()
        prefix = int(hex_key[:2], 16)
        x = int(hex_key[2:], 16)
        y_sq = (pow(x, 3, SECP_P) + SECP_B) % SECP_P
        y = tonelli_shanks_sqrt(y_sq, SECP_P)
        if y == 0 and y_sq != 0: raise ValueError("Point not on curve")
        if prefix == 2 and y % 2 != 0: y = SECP_P - y
        if prefix == 3 and y % 2 == 0: y = SECP_P - y
        return (x, y)
    except: raise ValueError("Invalid Hex Key")

def continued_fractions_approx(num, den, max_den):
    """Returns the best rational approximation numerator/denominator."""
    if den == 0: return 0, 1
    f = Fraction(num, den).limit_denominator(max_den)
    return f.numerator, f.denominator

# ==============================================================================
# 3. DATA EXTRACTION HELPERS
# ==============================================================================

def extract_counts_from_result(result) -> Dict[str, int]:
    if isinstance(result, dict): return result
    try:
        # SamplerV2 PubResult
        if hasattr(result[0], 'data'):
            if hasattr(result[0].data, 'phase_bits'): return result[0].data.phase_bits.get_counts()
            if hasattr(result[0].data, 'meas'): return result[0].data.meas.get_counts()
            if hasattr(result[0].data, 'c'): return result[0].data.c.get_counts()
    except Exception: pass
    try:
        # SamplerV1
        if hasattr(result[0].data, 'eigenstate'):
            bp = result[0].data.eigenstate.binary_probabilities()
            return {k: int(v * 10000) for k, v in bp.items()}
    except Exception: pass
    try: return result.get_counts()
    except: return {}

# ==============================================================================
# 4. MODE 1: ECDLP / BITCOIN SOLVER
# ==============================================================================

def solve_verified_ecdlp(counts: dict, n_bits: int, start_val: int, qx_target: int, qy_target: Optional[int] = None) -> Optional[int]:
    """
    Solves ECDLP by converting Phase -> Private Key 'd'.
    Optimized for SECP256k1 (Prime Order): Removes redundant GCD checks.
    """
    print(f"\n[Mode 1] Running ECDLP Verification...")
    
    valid_counts = {}
    clean_counts = {k.replace(" ", ""): v for k, v in counts.items()}
    
    # Filter logic (Symmetry check)
    for bitstr, freq in clean_counts.items():
        expected_len = 2 * n_bits
        if len(bitstr) >= expected_len:
            # Split into Check/Phase registers
            part_a = bitstr[0 : n_bits]
            part_b = bitstr[n_bits : 2*n_bits]
            if part_a != part_b: continue 
            valid_counts[part_a] = valid_counts.get(part_a, 0) + freq
        else:
            # Fallback for non-verified circuits
            valid_counts[bitstr] = valid_counts.get(bitstr, 0) + freq

    if not valid_counts:
        print("[!] No shots passed filtering. Using raw data.")
        valid_counts = clean_counts

    sorted_valid = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"[i] Checking top {min(len(sorted_valid), 20)} candidates...")

    for best_bitstring, frequency in sorted_valid[:20]:
        
        # Attempt both Endian interpretations
        int_candidates = []
        try: int_candidates.append(int(best_bitstring, 2)) 
        except: pass
        try: int_candidates.append(int(best_bitstring[::-1], 2)) 
        except: pass
        
        for integer_meas in set(int_candidates):
            phase = integer_meas / (2 ** n_bits)
            if phase == 0: continue
            
            d_candidates = []
            
            # Strategy A: Direct Phase Mapping (d = Phase * Order)
            d_candidates.append(int(round(phase * SECP_N)))
            
            # Strategy B: Continued Fractions (Rational Approximation)
            frac_num, frac_den = continued_fractions_approx(integer_meas, 2**n_bits, SECP_N)
            d_candidates.append(frac_num)
            
            # Verification: Elliptic Curve Multiplication
            for delta_d in set(d_candidates):
                if delta_d <= 0: continue
                full_key = (start_val + delta_d) % SECP_N
                
                try:
                    pub = ec_scalar_multiply(full_key, (SECP_GX, SECP_GY))
                    if pub:
                        # Positive Case
                        if pub[0] == qx_target:
                            if qy_target and pub[1] != qy_target: continue
                            print(f"\n[SUCCESS] ECDLP CRACKED: {hex(full_key)}")
                            return full_key
                        
                        # Negative/Inverse Case (N - d)
                        neg = ec_point_negate(pub)
                        if neg[0] == qx_target:
                            real_k = (SECP_N - full_key) % SECP_N
                            print(f"\n[SUCCESS] ECDLP CRACKED (Inverted): {hex(real_k)}")
                            return real_k
                except: continue
                
    print("[x] ECDLP Solver failed to find key in top candidates.")
    return None

# ==============================================================================
# 5. MODE 2: SHOR'S FACTORING (GCD Logic)
# ==============================================================================

def solve_shors_factoring(counts, n_bits, N_to_factor, guess_g):
    """
    Shor's Algorithm Logic (Video Method).
    Finds period 'r' via Continued Fractions, then factors via GCD.
    """
    print(f"\n[Mode 2] Running Shor's Factoring Logic (N={N_to_factor}, g={guess_g})...")

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    for bitstring_raw, freq in sorted_counts[:10]:
        clean_bitstring = bitstring_raw.replace(" ", "")
        
        try:
            # Standard LSB parsing for Shor's
            integer_meas = int(clean_bitstring[-n_bits:], 2)
        except: continue
            
        if integer_meas == 0: continue

        # 1. Find Period 'r' (The Denominator)
        frac_num, r = continued_fractions_approx(integer_meas, 2**n_bits, N_to_factor)
        
        # 2. Shor's Requirements
        if r % 2 != 0: continue # Period must be even
        
        half_r = r // 2
        try:
            x = pow(guess_g, half_r, N_to_factor)
        except ValueError: continue
            
        if x == 1 or x == (N_to_factor - 1): continue # Trivial
        
        # 3. The GCD Extraction (Correct usage for Factoring)
        cand_1 = gcd(x - 1, N_to_factor)
        cand_2 = gcd(x + 1, N_to_factor)
        
        if cand_1 not in [1, N_to_factor]:
            print(f"\n[SUCCESS] FACTOR FOUND: {cand_1}")
            return cand_1
        if cand_2 not in [1, N_to_factor]:
            print(f"\n[SUCCESS] FACTOR FOUND: {cand_2}")
            return cand_2
            
    print("[x] Shor's Solver failed.")
    return None

# ==============================================================================
# 6. MAIN EXECUTION & SELECTOR
# ==============================================================================

def main():
    print("="*80)
    print("     IBM JOB RETRIEVAL & DUAL ANALYSIS")
    print(f"     Job ID: {JOB_ID}")
    print("="*80)

    # 1. Setup Targets
    try:
        start_val = int(PUZZLE_START_HEX, 16)
        target_Qx, target_Qy = decompress_pubkey(TARGET_PUBKEY_HEX)
        print(f"[i] Target Public Key Decompressed.")
        print(f"    Qx: {hex(target_Qx)}")
    except Exception as e:
        print(f"[!] Error parsing target info: {e}")
        return

    # 2. Connect & Fetch
    print(f"[i] Authenticating & Fetching Job...")
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=API_TOKEN, overwrite=True)
        service = QiskitRuntimeService()
        job = service.job(JOB_ID)
        status = job.status()
        print(f"[i] Job Status: {status}")
    except Exception as e:
        print(f"[!] Connection failed: {e}")
        return

    # 3. Process Results
    completed_states = ["DONE", "COMPLETED", "SUCCEEDED"]
    
    # Check if job is finished
    if str(status).upper() not in completed_states and hasattr(status, "name"):
         if status.name not in completed_states:
             print(f"[!] Job is not complete yet. Current state: {status}")
             return
             
    # Fetch Data
    try:
        print("[i] Downloading Results...")
        result = job.result()
        counts = extract_counts_from_result(result)
        
        if not counts:
            print("[!] No counts extracted.")
            return
            
        print(f"[i] Extracted {len(counts)} unique bitstrings.")
        
        # ==============================================================
        # SELECT MODE
        # ==============================================================
        print("\n" + "="*40)
        print(" SELECT POST-PROCESSING MODE")
        print("="*40)
        print(" [1] ECDLP / Bitcoin Mode (Finds 'd')")
        print("     -> Direct Phase Mapping")
        print("     -> Continued Fractions")
        print("     -> No Redundant GCD Checks")
        print(" [2] Shor's Factoring Mode (Finds Factors)")
        print("     -> Period Finding")
        print("     -> GCD(g^(r/2) +/- 1, N)")
        print("="*40)
        
        choice = input("Enter choice [1 or 2]: ").strip()
        
        if choice == '1':
            key = solve_verified_ecdlp(
                counts, N_COUNTING_BITS, start_val, target_Qx, target_Qy
            )
            if key:
                with open(f"KEY_{JOB_ID[-4:]}.txt", "w") as f: f.write(hex(key))
        
        elif choice == '2':
            try:
                user_N = int(input("    Enter N (Number to factor): "))
                user_g = int(input("    Enter g (The guess used in circuit): "))
                solve_shors_factoring(counts, N_COUNTING_BITS, user_N, user_g)
            except ValueError:
                print("[!] Invalid input.")
        else:
            print("[!] Invalid Selection.")

    except Exception as e:
        print(f"[!] Error processing result: {e}")

if __name__ == "__main__":
    main()