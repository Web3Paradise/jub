#!/usr/bin/env python3
"""
epyc_vanity.py - Optimized Solana vanity address generator for AMD EPYC processors
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
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('epyc_vanity')

# EPYC-specific constants
DEFAULT_PHYSICAL_CORES = 64  # EPYC 7702 has 64 physical cores
DEFAULT_LOGICAL_CORES = 128  # EPYC 7702 has 128 logical cores (with SMT)
DEFAULT_BATCH_SIZE = 2048    # Larger batch size for server-class CPU
DEFAULT_MAX_MEMORY_GB = 32   # Default memory limit in GB
NUMA_NODES = 8               # EPYC 7702 has 8 NUMA nodes

# Calculate optimal chunk size based on pattern length and available memory
def calculate_chunk_size(pattern_length, max_memory_gb):
    # Memory analysis:
    # - Each key operation uses roughly 2KB of memory
    # - Leave 25% overhead for Python runtime
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024 * 0.75
    
    # Shorter patterns need larger chunks for efficiency
    if pattern_length <= 3:
        return min(10000000, int(max_memory_bytes / 2048))  # 10M keys or memory limit
    elif pattern_length <= 5:
        return min(5000000, int(max_memory_bytes / 2048))   # 5M keys or memory limit
    else:
        return min(2000000, int(max_memory_bytes / 2048))   # 2M keys or memory limit

# Process and thread binding utilities
def bind_to_cpu(cpu_id):
    """Bind the current process to a specific CPU (Linux only)"""
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([cpu_id])
        return True
    except (ImportError, AttributeError):
        return False

def worker_task(args):
    """Worker function that generates and checks addresses"""
    worker_id, prefix, suffix, batch_size, bind_cpu, physical_cores_only = args
    
    # CPU affinity binding for NUMA optimization
    if bind_cpu:
        if physical_cores_only:
            # Bind to physical cores only (0, 1, 2, ..., 63)
            cpu_id = worker_id % DEFAULT_PHYSICAL_CORES
        else:
            # Use all logical cores (0-127)
            cpu_id = worker_id % DEFAULT_LOGICAL_CORES
            
        success = bind_to_cpu(cpu_id)
        if success:
            logger.debug(f"Worker {worker_id} bound to CPU {cpu_id}")
    
    # Seed RNG for better distribution
    rng = np.random.RandomState(int(time.time()) ^ (worker_id * 1000003))
    
    attempts = 0
    start_time = time.time()
    report_interval = max(1000, batch_size // 10)
    
    try:
        while True:
            # Generate and check a batch of addresses
            for _ in range(batch_size):
                # Use numpy for faster random generation
                seed_bytes = bytes(rng.bytes(32))
                
                # Derive Ed25519 key
                signing_key = SigningKey(seed_bytes)
                verify_key = signing_key.verify_key
                public_key_bytes = verify_key.encode()
                address = base58.b58encode(public_key_bytes).decode()
                
                attempts += 1
                
                # Check if the address matches our criteria
                if ((not prefix or address.startswith(prefix)) and 
                    (not suffix or address.endswith(suffix))):
                    # Found a match!
                    elapsed = time.time() - start_time
                    rate = attempts / elapsed if elapsed > 0 else 0
                    return {
                        "found": True,
                        "seed": seed_bytes,
                        "address": address,
                        "attempts": attempts,
                        "worker_id": worker_id,
                        "rate": rate
                    }
                
                # Periodically report back
                if attempts % report_interval == 0:
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
    
    # Approximate keys per second based on EPYC performance
    # EPYC 7702 can process approximately 15,000-30,000 keys/sec per core
    rate_per_core = 25000  # Mid-range estimate for EPYC
    cores = DEFAULT_PHYSICAL_CORES  # Use physical core count for estimation
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
    elif seconds < 2592000:
        return f"{seconds/604800:.1f} weeks"
    else:
        return f"{seconds/2592000:.1f} months"

def main():
    parser = argparse.ArgumentParser(description="EPYC-Optimized Solana Vanity Address Generator")
    parser.add_argument("--prefix", type=str, default="", 
                        help="Base58 prefix to match")
    parser.add_argument("--suffix", type=str, default="", 
                        help="Base58 suffix to match")
    parser.add_argument("--threads", type=int, default=DEFAULT_PHYSICAL_CORES, 
                        help=f"Number of worker threads (default: {DEFAULT_PHYSICAL_CORES} physical cores)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f"Keys per batch per worker (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output", type=str, default="", 
                        help="Output file path")
    parser.add_argument("--output-dir", type=str, default="./wallets", 
                        help="Output directory")
    parser.add_argument("--bind-cpu", action="store_true", 
                        help="Bind workers to specific CPUs (improves NUMA performance)")
    parser.add_argument("--use-all-threads", action="store_true",
                        help="Use all logical cores (128) instead of just physical cores (64)")
    parser.add_argument("--max-memory", type=int, default=DEFAULT_MAX_MEMORY_GB,
                        help=f"Maximum memory usage in GB (default: {DEFAULT_MAX_MEMORY_GB})")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.prefix and not args.suffix:
        logger.error("Error: Must specify at least one of --prefix or --suffix")
        sys.exit(1)
    
    # Determine thread count
    if args.threads <= 0:
        if args.use_all_threads:
            args.threads = DEFAULT_LOGICAL_CORES
        else:
            args.threads = DEFAULT_PHYSICAL_CORES
    
    physical_cores_only = not args.use_all_threads
    
    # Calculate chunk size based on pattern length and memory limits
    pattern_len = max(len(args.prefix or ""), len(args.suffix or ""))
    chunk_size = calculate_chunk_size(pattern_len, args.max_memory)
    
    logger.info(f"EPYC-Optimized Vanity Address Generator")
    logger.info(f"Searching for: prefix='{args.prefix}' suffix='{args.suffix}'")
    logger.info(f"Using {args.threads} worker threads")
    logger.info(f"Batch size: {args.batch} keys per worker")
    logger.info(f"CPU binding: {'Enabled' if args.bind_cpu else 'Disabled'}")
    logger.info(f"Thread mode: {'All logical cores' if args.use_all_threads else 'Physical cores only'}")
    
    # Estimate difficulty
    expected_attempts = 0
    
    if args.prefix:
        estimate = estimate_difficulty(args.prefix, is_prefix=True)
        time_estimate = format_time(estimate["estimated_seconds"])
        logger.info(f"Prefix '{args.prefix}' difficulty estimate:")
        logger.info(f"  Approximately 1 in {estimate['combinations']:,} keys")
        logger.info(f"  Estimated time with {args.threads} threads: {time_estimate}")
        expected_attempts = estimate["expected_attempts"]
    
    if args.suffix:
        estimate = estimate_difficulty(args.suffix, is_prefix=False)
        time_estimate = format_time(estimate["estimated_seconds"])
        logger.info(f"Suffix '{args.suffix}' difficulty estimate:")
        logger.info(f"  Approximately 1 in {estimate['combinations']:,} keys")
        logger.info(f"  Estimated time with {args.threads} threads: {time_estimate}")
        expected_attempts = estimate["expected_attempts"]
    
    # Set up multiprocessing - use 'fork' for better performance on Linux
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        # Already set, or not supported
        pass
    
    # Create pool with larger subprocess memory limits
    pool = multiprocessing.Pool(processes=args.threads)
    manager = multiprocessing.Manager()
    
    # Prepare worker arguments
    worker_args = [(i, args.prefix, args.suffix, args.batch, args.bind_cpu, physical_cores_only) 
                  for i in range(args.threads)]
    
    start_time = time.time()
    total_attempts = 0
    last_status_time = start_time
    round_count = 0
    
    try:
        # Start asynchronous processing
        results = [pool.apply_async(worker_task, (arg,)) for arg in worker_args]
        
        # Continue until we find a match
        found_result = None
        while found_result is None:
            # Check results in progress
            all_done = True
            new_results = []
            
            for result in results:
                if not result.ready():
                    all_done = False
                    new_results.append(result)
                    continue
                
                # Get the result and check if it found a match
                try:
                    worker_result = result.get(timeout=1)
                    
                    # Update total attempts
                    total_attempts += worker_result.get("attempts", 0)
                    
                    if worker_result.get("found", False):
                        found_result = worker_result
                        break
                    elif "error" in worker_result:
                        logger.error(f"Worker {worker_result['worker_id']} failed: {worker_result['error']}")
                        # Restart this worker
                        worker_id = worker_result['worker_id']
                        new_results.append(pool.apply_async(worker_task, (worker_args[worker_id],)))
                    else:
                        # Worker completed without finding anything - restart it
                        worker_id = worker_result['worker_id']
                        new_results.append(pool.apply_async(worker_task, (worker_args[worker_id],)))
                except Exception as e:
                    logger.error(f"Error retrieving result: {e}")
                    # Keep checking other results
                    continue
            
            # If we found a match, exit the loop
            if found_result:
                break
                
            # Update our list of results
            results = new_results
            
            # If all workers finished without finding anything, restart them
            if all_done:
                round_count += 1
                # Display status before restarting
                elapsed = time.time() - start_time
                rate = total_attempts / elapsed if elapsed > 0 else 0
                
                # Show progress percentage
                progress_str = ""
                if expected_attempts > 0:
                    progress_pct = (total_attempts / expected_attempts) * 100
                    progress_str = f" (Progress: {progress_pct:.2f}%)"
                
                logger.info(f"Completed round {round_count}: {total_attempts:,} keys, Rate: {rate:.2f} keys/sec{progress_str}")
                logger.info("Starting new search round...")
                
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
            
            # Short sleep to prevent CPU hogging in the monitoring loop
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
        print("\n")  # Clear the status line
        logger.info(f"FOUND match after {total_attempts:,} attempts in {elapsed:.2f} seconds")
        logger.info(f"Found by worker {found_result['worker_id']}")
        logger.info(f"Address: {solana_pub}")
        logger.info(f"Phantom-compatible secret key: {phantom_key}")
        logger.info(f"Raw private key (base58): {raw_private_key}")
        
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
        
        logger.info(f"Saved result to: {out_path}")
    
    except KeyboardInterrupt:
        print("\n")  # Clear the status line
        logger.info("Stopping workers...")
        pool.terminate()
        elapsed = time.time() - start_time
        logger.info(f"Stopped after {total_attempts:,} attempts in {elapsed:.2f} seconds")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        pool.terminate()
        sys.exit(1)
    
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
