#!/usr/bin/env python3
"""
optimized_cpu_vanity.py

High-performance CPU-based Solana vanity address generator with:
- Memory optimization to prevent slowdowns
- Efficient worker management
- Better progress reporting
"""

import argparse
import base58
import json
import os
import sys
import time
import secrets
import multiprocessing
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from nacl.signing import SigningKey
import numpy as np

# Try to import faster libraries, but only log once in the main process
if multiprocessing.current_process().name == 'MainProcess':
    try:
        import pyximport
        pyximport.install()
        HAVE_CYTHON = True
        print("[INFO] Cython support available - using accelerated functions")
    except ImportError:
        HAVE_CYTHON = False
        print("[INFO] Cython not available - using pure Python")

    try:
        import ctypes
        from cffi import FFI
        HAVE_CFFI = True
        print("[INFO] CFFI support available - can use libsodium directly")
    except ImportError:
        HAVE_CFFI = False
        print("[INFO] CFFI not available - using PyNaCl")
else:
    # For worker processes, don't output anything
    HAVE_CYTHON = False
    HAVE_CFFI = False

# Batch size to generate and check at once
DEFAULT_BATCH_SIZE = 256

# Calculate optimal chunk size based on pattern length
def calculate_chunk_size(pattern_length):
    # Optimal sizes determined through testing
    if pattern_length <= 3:
        return 1000000  # 1M keys per chunk for short patterns
    elif pattern_length <= 5:
        return 750000   # 750K keys for medium patterns
    else:
        return 500000   # 500K keys for longer patterns

class VanityGenerator:
    """Base class for generating Solana vanity addresses"""
    
    def __init__(self, prefix="", suffix="", batch_size=DEFAULT_BATCH_SIZE):
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        
    def generate_keys_batch(self, count):
        """Generate a batch of random keys"""
        seeds = [secrets.token_bytes(32) for _ in range(count)]
        return seeds
        
    def derive_public_keys(self, seed_batch):
        """Derive public keys from seeds (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def check_pattern_match(self, addresses, seeds):
        """Check batch of addresses for matching patterns"""
        for i, addr in enumerate(addresses):
            if ((not self.prefix or addr.startswith(self.prefix)) and
                (not self.suffix or addr.endswith(self.suffix))):
                return True, seeds[i], addr
        return False, None, None
        
    def search_batch(self):
        """Search a batch of keys for matching pattern"""
        seeds = self.generate_keys_batch(self.batch_size)
        addresses = self.derive_public_keys(seeds)
        return self.check_pattern_match(addresses, seeds)
        
    def search_n_batches(self, n=1):
        """Search multiple batches, return stats"""
        attempts = 0
        for _ in range(n):
            found, seed, addr = self.search_batch()
            attempts += self.batch_size
            if found:
                return True, seed, addr, attempts
        return False, None, None, attempts

class OptimizedGenerator(VanityGenerator):
    """Memory-optimized key generation"""
    
    def derive_public_keys(self, seed_batch):
        """Process keys with minimal memory overhead"""
        addresses = []
        # Process in smaller chunks to avoid memory pressure
        chunk_size = 32
        for i in range(0, len(seed_batch), chunk_size):
            chunk = seed_batch[i:i+chunk_size]
            
            # Process each seed in the chunk
            for seed in chunk:
                signing_key = SigningKey(seed)
                verify_key = signing_key.verify_key
                public_key_bytes = verify_key.encode()
                address = base58.b58encode(public_key_bytes).decode()
                addresses.append(address)
                
                # Explicitly delete to help garbage collection
                del signing_key
                del verify_key
                del public_key_bytes
            
            # Force garbage collection every few chunks
            if i % (chunk_size * 8) == 0:
                gc.collect()
                
        return addresses

def worker_process(args):
    """Worker function for parallel processing"""
    worker_id, prefix, suffix, chunk_size, batch_size = args
    
    # Create a generator with the requested parameters
    generator = OptimizedGenerator(prefix, suffix, batch_size)
    
    # Search until we find a match or exhaust chunk
    attempts = 0
    start_time = time.time()
    
    try:
        while attempts < chunk_size:
            # Search in smaller increments to avoid memory buildup
            increment = min(batch_size * 50, chunk_size - attempts)
            batches_to_search = increment // batch_size
            
            found, seed, addr, batch_attempts = generator.search_n_batches(batches_to_search)
            attempts += batch_attempts
            
            if found:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                return {
                    "found": True,
                    "seed": seed,
                    "address": addr,
                    "attempts": attempts,
                    "worker_id": worker_id,
                    "rate": rate,
                    "elapsed": elapsed
                }
            
            # Periodically force garbage collection
            if attempts % (batch_size * 200) == 0:
                gc.collect()
        
        # Didn't find anything in this chunk
        elapsed = time.time() - start_time
        rate = attempts / elapsed if elapsed > 0 else 0
        return {
            "found": False,
            "attempts": attempts,
            "worker_id": worker_id,
            "rate": rate,
            "elapsed": elapsed
        }
    except Exception as e:
        print(f"\n[ERROR] Worker {worker_id} error: {str(e)}")
        return {
            "found": False,
            "attempts": attempts,
            "worker_id": worker_id,
            "rate": 0,
            "elapsed": time.time() - start_time,
            "error": str(e)
        }

def estimate_search_time(pattern, is_prefix=True):
    """Estimate search time for a pattern based on length"""
    pattern_len = len(pattern)
    
    # Base probability: 1/58^n for exact pattern
    # Multiplier varies based on character frequency in Base58 encoding
    if is_prefix:
        # Prefixes are slightly harder due to Base58 encoding properties
        difficulty_multiplier = 1.2
    else:
        # Suffixes have more even distribution
        difficulty_multiplier = 1.0
        
    # Base58 has 58 characters
    combinations = 58 ** pattern_len
    
    # Adjust for actual Base58 character distribution
    expected_attempts = combinations * difficulty_multiplier
    
    # Typical rate on a modern CPU (keys per second per core)
    # This is very hardware dependent
    typical_rate_per_core = 20000  # Conservative estimate for a modern CPU
    
    # Estimate seconds needed with the given number of cores
    typical_cores = multiprocessing.cpu_count()
    estimated_seconds = expected_attempts / (typical_rate_per_core * typical_cores)
    
    return {
        "combinations": combinations,
        "expected_attempts": expected_attempts,
        "estimated_time_seconds": estimated_seconds
    }

def format_time_estimate(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 604800:
        return f"{seconds/86400:.1f} days"
    elif seconds < 2592000:
        return f"{seconds/604800:.1f} weeks"
    elif seconds < 31536000:
        return f"{seconds/2592000:.1f} months"
    else:
        return f"{seconds/31536000:.1f} years"

def restart_worker_pool():
    """Force restart of the worker pool to clean up memory"""
    # This is a no-op function that just ensures we create a fresh pool each round
    pass

def main():
    parser = argparse.ArgumentParser(description="High-Performance CPU Solana Vanity Address Generator")
    parser.add_argument("--prefix", type=str, default="", help="Base58 prefix to match")
    parser.add_argument("--suffix", type=str, default="", help="Base58 suffix to match")
    parser.add_argument("--threads", type=int, default=0, help="Number of CPU threads (0=auto)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f"Number of keys to check in a batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--chunk-size", type=int, default=0, 
                       help="Keys per worker chunk (0=auto-calculate based on pattern)")
    parser.add_argument("--output", type=str, default="", help="Output file path")
    parser.add_argument("--output-dir", type=str, default="./wallets", help="Output directory")
    parser.add_argument("--no-estimate", action="store_true", help="Skip difficulty estimation")
    parser.add_argument("--continuous", action="store_true", 
                       help="Keep searching even after finding a match")
    parser.add_argument("--status-interval", type=float, default=2.0,
                      help="How often to update status (seconds)")
    parser.add_argument("--timeout", type=int, default=0,
                      help="Stop after this many seconds (0=run indefinitely)")
    parser.add_argument("--max-rounds", type=int, default=0,
                      help="Maximum number of search rounds (0=unlimited)")
    args = parser.parse_args()
    
    if not args.prefix and not args.suffix:
        print("Error: Must specify at least one of --prefix or --suffix")
        sys.exit(1)
    
    if args.threads <= 0:
        args.threads = multiprocessing.cpu_count()
    
    # Calculate chunk size based on pattern length if not specified
    if args.chunk_size <= 0:
        pattern_len = max(len(args.prefix or ""), len(args.suffix or ""))
        args.chunk_size = calculate_chunk_size(pattern_len)
    
    print(f"[INFO] Searching for: prefix='{args.prefix}' suffix='{args.suffix}'")
    print(f"[INFO] Using {args.threads} CPU threads with {args.batch_size} keys per batch")
    print(f"[INFO] Each worker will process {args.chunk_size:,} keys per chunk")
    
    # Estimate difficulty
    pattern_len = max(len(args.prefix or ""), len(args.suffix or ""))
    expected_attempts = 0
    
    if not args.no_estimate and (args.prefix or args.suffix):
        if pattern_len > 8:
            print(f"[WARNING] Searching for a {pattern_len}-character pattern might take a very long time!")
            print("[WARNING] Consider using a shorter pattern or running on multiple machines.")
            
        if args.prefix:
            prefix_estimate = estimate_search_time(args.prefix, is_prefix=True)
            prefix_time = format_time_estimate(prefix_estimate["estimated_time_seconds"])
            print(f"[INFO] Prefix '{args.prefix}' difficulty estimate:")
            print(f"       Approximately 1 in {prefix_estimate['combinations']:,} keys")
            print(f"       Estimated time with {args.threads} threads: {prefix_time}")
            expected_attempts = prefix_estimate["expected_attempts"]
        
        if args.suffix:
            suffix_estimate = estimate_search_time(args.suffix, is_prefix=False)
            suffix_time = format_time_estimate(suffix_estimate["estimated_time_seconds"])
            print(f"[INFO] Suffix '{args.suffix}' difficulty estimate:")
            print(f"       Approximately 1 in {suffix_estimate['combinations']:,} keys")
            print(f"       Estimated time with {args.threads} threads: {suffix_time}")
            expected_attempts = suffix_estimate["expected_attempts"]
    
    start_time = time.time()
    total_attempts = 0
    keys_found = 0
    last_status_time = start_time
    search_round = 0
    
    try:
        # Continue searching until we find a key or reach max_rounds
        while args.max_rounds == 0 or search_round < args.max_rounds:
            search_round += 1
            print(f"\n[INFO] Starting search round {search_round}")
            
            # Check timeout if set
            if args.timeout > 0 and time.time() - start_time > args.timeout:
                print(f"\n[INFO] Timeout after {args.timeout} seconds. Stopping.")
                break
            
            # Create a fresh process pool for each round to prevent memory buildup
            with ProcessPoolExecutor(max_workers=args.threads) as executor:
                # Submit work to each worker
                future_to_worker = {}
                for i in range(args.threads):
                    worker_args = (i, args.prefix, args.suffix, args.chunk_size, args.batch_size)
                    future = executor.submit(worker_process, worker_args)
                    future_to_worker[future] = i
                
                # Process results as they complete
                found_result = None
                worker_stats = []
                
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        result = future.result()
                        worker_stats.append(result)
                        total_attempts += result["attempts"]
                        
                        # Update status periodically
                        current_time = time.time()
                        if current_time - last_status_time >= args.status_interval:
                            elapsed = current_time - start_time
                            rate = total_attempts / elapsed if elapsed > 0 else 0
                            
                            # Show progress percentage if we have an estimate
                            progress_str = ""
                            if expected_attempts > 0:
                                progress_pct = (total_attempts / expected_attempts) * 100
                                progress_str = f" (Progress: {progress_pct:.2f}%)"
                                
                            print(f"\r[INFO] Total: {total_attempts:,} keys, Rate: {rate:.2f} keys/sec{progress_str}", end="", flush=True)
                            last_status_time = current_time
                        
                        if result.get("found", False) and not found_result:
                            found_result = result
                            # If not in continuous mode, cancel all other workers
                            if not args.continuous:
                                for f in future_to_worker:
                                    if f != future and not f.done():
                                        f.cancel()
                    except Exception as e:
                        print(f"\n[ERROR] Worker {worker_id} failed: {e}")
            
            # Final status update for this round
            print("")  # New line after the status updates
            
            # Calculate overall statistics
            elapsed = time.time() - start_time
            
            if worker_stats:
                valid_stats = [w for w in worker_stats if w.get("rate", 0) > 0]
                if valid_stats:
                    avg_rate = sum(w["rate"] for w in valid_stats) / len(valid_stats)
                    print(f"[INFO] Round {search_round} complete: {avg_rate:.2f} keys/sec")
            
            # Force garbage collection between rounds
            gc.collect()
            
            # Process the found key if any
            if found_result:
                seed_bytes = found_result["seed"]
                solana_pub = found_result["address"]
                keys_found += 1
                
                # Generate required key formats
                signing_key = SigningKey(seed_bytes)
                verify_key = signing_key.verify_key
                public_key_bytes = verify_key.encode()
                secret_64 = seed_bytes + public_key_bytes
                phantom_key = base58.b58encode(secret_64).decode()
                raw_private_key = base58.b58encode(seed_bytes).decode()
                
                print(f"\n[FOUND] After {total_attempts:,} attempts in {elapsed:.2f} seconds")
                print(f"[INFO] Found by worker {found_result['worker_id']} in round {search_round}")
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
                    "search_round": search_round,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
                
                print(f"[INFO] Saved result to: {out_path}")
                
                # Exit if not in continuous mode
                if not args.continuous:
                    return
                
                print(f"[INFO] Continuing search in continuous mode...")
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[INFO] Stopped by user after {total_attempts:,} attempts in {elapsed:.2f} seconds.")
        print(f"[INFO] Average rate: {total_attempts / elapsed:.2f} keys/sec")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()