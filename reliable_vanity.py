#!/usr/bin/env python3
"""
reliable_vanity.py - A simple but reliable Solana vanity address generator
"""

import argparse
import base58
import json
import os
import sys
import time
import secrets
import multiprocessing
from nacl.signing import SigningKey

def generate_address(seed=None):
    """Generate a Solana address from a random or provided seed"""
    if seed is None:
        seed = secrets.token_bytes(32)
    
    # Derive Ed25519 key
    signing_key = SigningKey(seed)
    verify_key = signing_key.verify_key
    public_key_bytes = verify_key.encode()
    address = base58.b58encode(public_key_bytes).decode()
    
    return seed, address

def worker_task(args):
    """Worker function that generates and checks addresses"""
    worker_id, prefix, suffix, batch_size = args
    
    attempts = 0
    start_time = time.time()
    
    try:
        while True:
            # Generate and check a batch of addresses
            for _ in range(batch_size):
                seed, address = generate_address()
                attempts += 1
                
                # Check if the address matches our criteria
                if ((not prefix or address.startswith(prefix)) and 
                    (not suffix or address.endswith(suffix))):
                    # Found a match!
                    elapsed = time.time() - start_time
                    rate = attempts / elapsed if elapsed > 0 else 0
                    return {
                        "found": True,
                        "seed": seed,
                        "address": address,
                        "attempts": attempts,
                        "worker_id": worker_id,
                        "rate": rate
                    }
                
                # Periodically report back (every 1000 attempts)
                if attempts % 1000 == 0:
                    # Check if we should report progress
                    elapsed = time.time() - start_time
                    # Only report if we've been running for at least 5 seconds
                    if elapsed >= 5:
                        rate = attempts / elapsed
                        return {
                            "found": False,
                            "attempts": attempts,
                            "worker_id": worker_id,
                            "rate": rate
                        }
    
    except Exception as e:
        # Catch any exceptions and return information about them
        return {
            "found": False,
            "attempts": attempts,
            "worker_id": worker_id,
            "error": str(e)
        }

def estimate_difficulty(pattern, is_prefix=True):
    """Estimate the difficulty of finding a match"""
    pattern_len = len(pattern)
    
    # Base58 has 58 characters
    combinations = 58 ** pattern_len
    
    # Slight adjustment for prefix vs suffix
    multiplier = 1.2 if is_prefix else 1.0
    expected_attempts = combinations * multiplier
    
    # Approximate keys per second based on a modern CPU
    rate_per_core = 5000  # Conservative estimate
    cores = multiprocessing.cpu_count()
    estimated_seconds = expected_attempts / (rate_per_core * cores)
    
    return {
        "combinations": combinations,
        "expected_attempts": expected_attempts,
        "estimated_seconds": estimated_seconds
    }

def format_time(seconds):
    """Format seconds into a human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 604800:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/604800:.1f} weeks"

def main():
    parser = argparse.ArgumentParser(description="Reliable Solana Vanity Address Generator")
    parser.add_argument("--prefix", type=str, default="", 
                        help="Base58 prefix to match")
    parser.add_argument("--suffix", type=str, default="", 
                        help="Base58 suffix to match")
    parser.add_argument("--threads", type=int, default=0, 
                        help="Number of CPU threads (0=auto)")
    parser.add_argument("--batch", type=int, default=500, 
                        help="Keys per batch per worker")
    parser.add_argument("--output", type=str, default="", 
                        help="Output file path")
    parser.add_argument("--output-dir", type=str, default="./wallets", 
                        help="Output directory")
    args = parser.parse_args()
    
    if not args.prefix and not args.suffix:
        print("Error: Must specify at least one of --prefix or --suffix")
        sys.exit(1)
    
    if args.threads <= 0:
        args.threads = multiprocessing.cpu_count()
    
    print(f"[INFO] Searching for: prefix='{args.prefix}' suffix='{args.suffix}'")
    print(f"[INFO] Using {args.threads} CPU threads")
    
    # Estimate difficulty
    pattern_len = max(len(args.prefix or ""), len(args.suffix or ""))
    expected_attempts = 0
    
    if args.prefix:
        estimate = estimate_difficulty(args.prefix, is_prefix=True)
        time_estimate = format_time(estimate["estimated_seconds"])
        print(f"[INFO] Prefix '{args.prefix}' difficulty estimate:")
        print(f"       Approximately 1 in {estimate['combinations']:,} keys")
        print(f"       Estimated time with {args.threads} threads: {time_estimate}")
        expected_attempts = estimate["expected_attempts"]
    
    if args.suffix:
        estimate = estimate_difficulty(args.suffix, is_prefix=False)
        time_estimate = format_time(estimate["estimated_seconds"])
        print(f"[INFO] Suffix '{args.suffix}' difficulty estimate:")
        print(f"       Approximately 1 in {estimate['combinations']:,} keys")
        print(f"       Estimated time with {args.threads} threads: {time_estimate}")
        expected_attempts = estimate["expected_attempts"]
    
    # Set up multiprocessing
    pool = multiprocessing.Pool(processes=args.threads)
    manager = multiprocessing.Manager()
    found_address = manager.Value('i', 0)  # Shared flag for found address
    
    # Prepare worker arguments
    worker_args = [(i, args.prefix, args.suffix, args.batch) for i in range(args.threads)]
    
    start_time = time.time()
    total_attempts = 0
    last_status_time = start_time
    
    try:
        # Start asynchronous processing
        results = [pool.apply_async(worker_task, (arg,)) for arg in worker_args]
        
        # Continue until we find a match
        found_result = None
        while found_result is None:
            # Check results in progress
            all_done = True
            for result in results:
                if not result.ready():
                    all_done = False
                    continue
                
                # Get the result and check if it found a match
                worker_result = result.get()
                total_attempts += worker_result["attempts"]
                
                if worker_result.get("found", False):
                    found_result = worker_result
                    break
                elif "error" in worker_result:
                    print(f"\n[ERROR] Worker {worker_result['worker_id']} failed: {worker_result['error']}")
            
            # If all workers finished without finding anything, restart them
            if all_done:
                # Display status before restarting
                elapsed = time.time() - start_time
                rate = total_attempts / elapsed if elapsed > 0 else 0
                
                # Show progress percentage
                progress_str = ""
                if expected_attempts > 0:
                    progress_pct = (total_attempts / expected_attempts) * 100
                    progress_str = f" (Progress: {progress_pct:.2f}%)"
                
                print(f"\n[INFO] Total: {total_attempts:,} keys, Rate: {rate:.2f} keys/sec{progress_str}")
                print("[INFO] Restarting workers for next round...")
                
                # Start a new round of workers
                results = [pool.apply_async(worker_task, (arg,)) for arg in worker_args]
            
            # Periodically update status
            current_time = time.time()
            if current_time - last_status_time >= 5:
                elapsed = current_time - start_time
                rate = total_attempts / elapsed if elapsed > 0 else 0
                
                # Show progress percentage
                progress_str = ""
                if expected_attempts > 0:
                    progress_pct = (total_attempts / expected_attempts) * 100
                    progress_str = f" (Progress: {progress_pct:.2f}%)"
                
                print(f"\r[INFO] Total: {total_attempts:,} keys, Rate: {rate:.2f} keys/sec{progress_str}", end="", flush=True)
                last_status_time = current_time
            
            # Short sleep to prevent CPU hogging
            time.sleep(0.1)
        
        # We found a match!
        pool.terminate()  # Stop all workers
        
        # Process the found result
        seed_bytes = found_result["seed"]
        solana_pub = found_result["address"]
        
        # Generate required key formats
        signing_key = SigningKey(seed_bytes)
        verify_key = signing_key.verify_key
        public_key_bytes = verify_key.encode()
        secret_64 = seed_bytes + public_key_bytes
        phantom_key = base58.b58encode(secret_64).decode()
        raw_private_key = base58.b58encode(seed_bytes).decode()
        
        elapsed = time.time() - start_time
        print(f"\n\n[FOUND] After {total_attempts:,} attempts in {elapsed:.2f} seconds")
        print(f"[INFO] Found by worker {found_result['worker_id']}")
        print(f"[INFO] Address: {solana_pub}")
        print(f"[INFO] Phantom-compatible secret key: {phantom_key}")
        print(f"[INFO] Raw private key (base58): {raw_private_key}")
        
        # Save result
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
            "raw_private_key": raw_private_key,
            "attempts": total_attempts,
            "elapsed_sec": elapsed,
            "keys_per_second": total_attempts / elapsed if elapsed > 0 else 0,
            "prefix": args.prefix,
            "suffix": args.suffix,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Saved result to: {out_path}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping workers...")
        pool.terminate()
        elapsed = time.time() - start_time
        print(f"[INFO] Stopped after {total_attempts:,} attempts in {elapsed:.2f} seconds")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        pool.terminate()
        sys.exit(1)
    
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()