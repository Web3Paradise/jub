#!/usr/bin/env python3
"""
gpu_vanity_main.py

Hybrid GPU-CPU Solana Vanity Generator.

The GPU is used as a fast candidate filter using a simplified derivation.
When a candidate passes the GPU filter, the CPU fully derives the Ed25519 public key
(using PyNaCl) and checks if the final (real) public key matches the vanity pattern.
If it does, the key is output.
"""

import pyopencl as cl
import numpy as np
import argparse
import sys
import os
import time
import json
import base58
from nacl.signing import SigningKey

def main():
    parser = argparse.ArgumentParser(description="Hybrid GPU-CPU Solana Vanity Generator")
    parser.add_argument("--prefix", type=str, default="", help="Base58 prefix to match")
    parser.add_argument("--suffix", type=str, default="", help="Base58 suffix to match")
    parser.add_argument("--gpu-id", type=int, default=0, help="Which GPU device to use")
    parser.add_argument("--global-size", type=int, default=2**20, help="Number of threads per kernel launch (increased from 2^16)")
    parser.add_argument("--output", type=str, default="", help="File path to store the found key")
    parser.add_argument("--output-dir", type=str, default="./wallets", help="Directory for output if no --output given")
    parser.add_argument("--kernel-file", type=str, default="ed25519_kernel.cl", help="Path to the OpenCL kernel source")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--max-false-positives", type=int, default=10, help="Max false positives to print before quitting (0=unlimited)")
    parser.add_argument("--timeout", type=int, default=0, help="Stop after this many seconds (0=run indefinitely)")
    parser.add_argument("--status-interval", type=int, default=5, help="How often to update status (seconds)")
    args = parser.parse_args()

    if not args.prefix and not args.suffix:
        print("Error: must specify at least one of --prefix or --suffix.")
        sys.exit(1)
    if not os.path.exists(args.kernel_file):
        print(f"Error: kernel file not found: {args.kernel_file}")
        sys.exit(1)
        
    # Open the kernel file with UTF-8 encoding and replace errors
    with open(args.kernel_file, "r", encoding="utf-8", errors="replace") as f:
        kernel_src = f.read()

    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found.")
        sys.exit(1)
        
    all_gpus = []
    for pf in platforms:
        try:
            devs = pf.get_devices(device_type=cl.device_type.GPU)
            all_gpus.extend(devs)
        except Exception:
            pass
            
    if not all_gpus:
        print("No GPU devices found.")
        sys.exit(1)
        
    if args.gpu_id >= len(all_gpus):
        print(f"GPU ID {args.gpu_id} is invalid. Using 0.")
        args.gpu_id = 0
        
    device = all_gpus[args.gpu_id]
    print("[INFO] Using GPU:", device.name)
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    try:
        program = cl.Program(ctx, kernel_src).build(options=["-cl-std=CL1.2"])
    except Exception as e:
        print("[ERROR] Build failed:", e)
        sys.exit(1)

    prefix_bytes = args.prefix.encode("ascii")
    suffix_bytes = args.suffix.encode("ascii")
    prefix_len = np.int32(len(prefix_bytes))
    suffix_len = np.int32(len(suffix_bytes))
    
    mf = cl.mem_flags
    b58_prefix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prefix_bytes) if prefix_len > 0 else None
    b58_suffix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=suffix_bytes) if suffix_len > 0 else None

    out_seed = np.zeros(32, dtype=np.uint8)
    out_b58pub = np.zeros(50, dtype=np.uint8)
    found_flag = np.array([0], dtype=np.int32)
    
    out_seed_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out_seed)
    out_b58pub_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out_b58pub)
    found_flag_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_flag)
    
    kernel = program.vanity_search

    random_state = np.random.RandomState()
    global_size = args.global_size
    attempts = 0
    false_positives = 0
    start_time = time.time()
    last_status_time = start_time

    print(f"[INFO] Searching for vanity with prefix='{args.prefix}' suffix='{args.suffix}'.")
    print(f"[INFO] Global size per launch = {global_size}")
    print(f"[INFO] Status updates every {args.status_interval} seconds")
    
    if args.timeout > 0:
        print(f"[INFO] Will stop after {args.timeout} seconds if no match found")

    while True:
        # Check timeout
        current_time = time.time()
        if args.timeout > 0 and current_time - start_time > args.timeout:
            print(f"\n[INFO] Timeout after {args.timeout} seconds. Stopping.")
            break
            
        seed1 = np.uint64(random_state.randint(0, 2**63, dtype=np.int64))
        seed2 = np.uint64(random_state.randint(0, 2**63, dtype=np.int64))
        
        out_seed.fill(0)
        out_b58pub.fill(0)
        found_flag[0] = 0
        
        cl.enqueue_copy(queue, out_seed_buf, out_seed)
        cl.enqueue_copy(queue, out_b58pub_buf, out_b58pub)
        cl.enqueue_copy(queue, found_flag_buf, found_flag)
        
        kernel.set_args(
            seed1,
            seed2,
            b58_prefix_buf if prefix_len > 0 else cl.LocalMemory(0),
            prefix_len,
            b58_suffix_buf if suffix_len > 0 else cl.LocalMemory(0),
            suffix_len,
            out_seed_buf,
            out_b58pub_buf,
            found_flag_buf
        )
        
        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), None)
        queue.finish()
        attempts += global_size
        
        cl.enqueue_copy(queue, found_flag, found_flag_buf)
        
        # Only copy results if we found a candidate
        if found_flag[0] == 1:
            cl.enqueue_copy(queue, out_seed, out_seed_buf)
            cl.enqueue_copy(queue, out_b58pub, out_b58pub_buf)
            queue.finish()

            # When the GPU filter finds a candidate, perform full derivation on CPU.
            seed_bytes = bytes(out_seed)
            signing_key = SigningKey(seed_bytes)
            verify_key = signing_key.verify_key
            real_pub_bytes = verify_key.encode()
            solana_pub = base58.b58encode(real_pub_bytes).decode()
            
            # Get the candidate string from GPU
            candidate_b58 = "".join(chr(x) for x in out_b58pub if x != 0)
            
            # Only accept if the full derived public key matches the desired pattern.
            if ((args.prefix and not solana_pub.startswith(args.prefix)) or
                (args.suffix and not solana_pub.endswith(args.suffix))):
                # This is a false positive
                false_positives += 1
                
                if args.debug or (args.max_false_positives > 0 and false_positives <= args.max_false_positives):
                    print(f"\n[DEBUG] False positive #{false_positives}:")
                    print(f"  GPU candidate: {candidate_b58}")
                    print(f"  Real address: {solana_pub}")
                
                # Show where we found false matching characters to help diagnose
                if args.debug:
                    if args.prefix:
                        prefix_match = min(len(args.prefix), len(solana_pub))
                        matching = 0
                        for i in range(prefix_match):
                            if solana_pub[i] == args.prefix[i]:
                                matching += 1
                            else:
                                break
                        print(f"  Prefix matches first {matching}/{len(args.prefix)} characters")
                        
                    if args.suffix:
                        suffix_match = min(len(args.suffix), len(solana_pub))
                        matching = 0
                        for i in range(suffix_match):
                            if solana_pub[-(i+1)] == args.suffix[-(i+1)]:
                                matching += 1
                            else:
                                break
                        print(f"  Suffix matches last {matching}/{len(args.suffix)} characters")
                
                # If we've reached the maximum false positives limit, exit
                if args.max_false_positives > 0 and false_positives >= args.max_false_positives:
                    print(f"\n[ERROR] Reached {false_positives} false positives without finding a match.")
                    print("This suggests the GPU filter is not well-correlated with actual Ed25519 keys.")
                    print("Try using an improved kernel or switch to a CPU-only approach.")
                    sys.exit(1)
                    
                continue  # false positive from GPU; keep searching
            
            # We found a real match!
            elapsed = time.time() - start_time
            rate = attempts / elapsed if elapsed > 0 else 0
            print(f"\n[FOUND] After {attempts:,} attempts in {elapsed:.2f}s (~{rate:,.2f} H/s)")
            print(f"[INFO] Found after {false_positives} false positives")
            print(f"[GPU] Candidate Base58 (from GPU filter) = {candidate_b58}")
            print(f"[REAL] Ed25519 public key = {solana_pub}")
            
            secret_64 = seed_bytes + real_pub_bytes
            phantom_key = base58.b58encode(secret_64).decode()
            print(f"[REAL] Phantom-compatible secret key = {phantom_key}")
            
            if args.output:
                out_path = args.output
                out_dir = os.path.dirname(out_path)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            else:
                pat = []
                if args.prefix:
                    pat.append(f"prefix_{args.prefix}")
                if args.suffix:
                    pat.append(f"suffix_{args.suffix}")
                pat_str = "_".join(pat)
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, f"solana_{pat_str}_{int(time.time())}.json")
                
            data = {
                "public_key": solana_pub,
                "phantom_key": phantom_key,
                "raw_private_key": base58.b58encode(seed_bytes).decode(),
                "found_b58_gpu": candidate_b58,
                "attempts": attempts,
                "elapsed_sec": elapsed,
                "prefix": args.prefix,
                "suffix": args.suffix,
                "false_positives": false_positives,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"[INFO] Saved result to: {out_path}")
            return
        
        # Update status at regular intervals
        if current_time - last_status_time >= args.status_interval:
            dt = current_time - start_time
            speed = attempts/dt if dt > 0 else 0
            print(f"\r[INFO] Attempts: {attempts:,}  Speed: {speed:,.2f} H/s  False positives: {false_positives}", end="")
            last_status_time = current_time

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)