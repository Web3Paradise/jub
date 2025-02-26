#!/usr/bin/env python3
"""
cpu_vanity_solana.py

CPU-only multithreaded Solana vanity address generator.
Less overhead, no false positives, might be more effective for short patterns.
"""

import argparse
import base58
import json
import os
import sys
import time
import secrets
from nacl.signing import SigningKey
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def generate_key_with_pattern(prefix, suffix, thread_id):
    """Generate random keys until one matches the desired pattern"""
    attempts = 0
    
    while True:
        # Generate random seed (private key)
        seed_bytes = secrets.token_bytes(32)
        
        # Derive Ed25519 key
        signing_key = SigningKey(seed_bytes)
        verify_key = signing_key.verify_key
        public_key_bytes = verify_key.encode()
        
        # Get Solana address in Base58
        solana_address = base58.b58encode(public_key_bytes).decode()
        
        # Check if it matches our pattern
        if ((not prefix or solana_address.startswith(prefix)) and
            (not suffix or solana_address.endswith(suffix))):
            return {
                "seed_bytes": seed_bytes,
                "public_key": solana_address,
                "attempts": attempts,
                "thread_id": thread_id
            }
            
        attempts += 1
        if attempts % 10000 == 0:
            print(f"\rThread {thread_id}: {attempts:,} attempts", end="", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="CPU Multithreaded Solana Vanity Generator")
    parser.add_argument("--prefix", type=str, default="", help="Base58 prefix to match")
    parser.add_argument("--suffix", type=str, default="", help="Base58 suffix to match")
    parser.add_argument("--threads", type=int, default=0, 
                       help="Number of CPU threads (0 = auto-detect)")
    parser.add_argument("--output", type=str, default="", help="Output file path")
    parser.add_argument("--output-dir", type=str, default="./wallets", help="Output directory")
    args = parser.parse_args()
    
    if not args.prefix and not args.suffix:
        print("Error: Must specify at least one of --prefix or --suffix")
        sys.exit(1)
    
    if args.threads <= 0:
        args.threads = multiprocessing.cpu_count()
    
    print(f"[INFO] Searching for: prefix='{args.prefix}' suffix='{args.suffix}'")
    print(f"[INFO] Using {args.threads} CPU threads")
    
    start_time = time.time()
    total_attempts = 0
    
    try:
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            # Submit the search to each worker
            futures = [
                executor.submit(generate_key_with_pattern, args.prefix, args.suffix, i) 
                for i in range(args.threads)
            ]
            
            # Wait for the first worker to find a match
            import concurrent.futures
            done, not_done = concurrent.futures.wait(
                futures, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Cancel other workers
            for future in not_done:
                future.cancel()
                
            # Get the result from the successful worker
            for future in done:
                result = future.result()
                break
    
        # Calculate statistics        
        elapsed = time.time() - start_time
        
        # Generate the output
        seed_bytes = result["seed_bytes"]
        solana_pub = result["public_key"]
        signing_key = SigningKey(seed_bytes)
        verify_key = signing_key.verify_key
        public_key_bytes = verify_key.encode()
        secret_64 = seed_bytes + public_key_bytes
        phantom_key = base58.b58encode(secret_64).decode()
        
        print(f"\n[FOUND] After {result['attempts']:,} attempts in thread {result['thread_id']}")
        print(f"[INFO] Total time: {elapsed:.2f} seconds")
        print(f"[REAL] Ed25519 public key = {solana_pub}")
        print(f"[REAL] Phantom-compatible secret key = {phantom_key}")
        
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
            "raw_private_key": base58.b58encode(seed_bytes).decode(),
            "attempts": result["attempts"],
            "thread_id": result["thread_id"],
            "elapsed_sec": elapsed,
            "prefix": args.prefix,
            "suffix": args.suffix,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"[INFO] Saved result to: {out_path}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()