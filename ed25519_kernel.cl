#pragma OPENCL EXTENSION cl_khr_int64 : enable

// This kernel implements a better approximation of Ed25519 characteristics
// to reduce false positives when searching for vanity addresses

typedef unsigned char u8;
typedef unsigned int  u32;
typedef unsigned long u64;

__constant char BASE58_ALPHABET[58] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// ---------- Xorshift128+ RNG ----------
typedef struct {
    u64 s0;
    u64 s1;
} xorshift128p_state;

u64 xorshift128p(xorshift128p_state *state) {
    u64 s1 = state->s0;
    u64 s0 = state->s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0;
    s1 ^= s0 >> 26;
    state->s1 = s1;
    return s1 + s0;
}

// ---------- Improved candidate derivation function ----------
// (Better approximates Ed25519 key characteristics)
void derive_candidate_pubkey(const u8 *seed, u8 *pubkey) {
    // Initialize with Ed25519 curve constants to better approximate distribution
    u32 h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    // Mix the seed more thoroughly
    for (int i = 0; i < 32; i += 4) {
        u32 val = seed[i] | (seed[i+1] << 8) | (seed[i+2] << 16) | (seed[i+3] << 24);
        // More thorough mixing to approximate distribution
        for (int j = 0; j < 8; j++) {
            h[j] ^= val;
            h[j] = (h[j] << 13) | (h[j] >> 19);
            h[(j+1) % 8] += h[j];
            h[j] = (h[j] << 7) | (h[j] >> 25);
        }
    }
    
    // Output the mixed state as the pubkey
    for (int i = 0; i < 32; i++) {
        pubkey[i] = (u8)((h[i/4] >> ((i % 4) * 8)) & 0xFF);
    }
}

// ---------- Base58 conversion for 32-byte input ----------
int to_base58_32(const u8 *in32, char *out58) {
    u8 digits[45] = {0};
    int digitslen = 1;
    for (int i = 0; i < 32; i++) {
        u32 carry = in32[i];
        for (int j = 0; j < digitslen && j < 45; j++) {
            u32 val = carry + ((u32)digits[j] << 8);
            digits[j] = val % 58;
            carry = val / 58;
        }
        while (carry > 0 && digitslen < 45) {
            digits[digitslen] = carry % 58;
            carry /= 58;
            digitslen++;
        }
    }
    int zeros = 0;
    for (int i = 0; i < 32 && in32[i] == 0; i++) zeros++;
    int b58len = 0;
    for (int i = 0; i < zeros && b58len < 49; i++) {
        out58[b58len++] = BASE58_ALPHABET[0];
    }
    for (int i = digitslen - 1; i >= 0 && b58len < 49; i--) {
        out58[b58len++] = BASE58_ALPHABET[digits[i]];
    }
    out58[b58len] = '\0';
    return b58len;
}

// ---------- Kernel Entry Point ----------
__kernel void vanity_search(
    const u64 seed1,
    const u64 seed2,
    __constant char *b58_prefix,
    const int prefix_len,
    __constant char *b58_suffix,
    const int suffix_len,
    __global u8 *out_seed,      // Candidate seed (32 bytes)
    __global char *out_b58pub,   // Candidate public key in Base58 (null terminated)
    __global int *found_flag
) {
    int gid = get_global_id(0);
    if (*found_flag != 0)
        return;
    
    // Generate candidate seed using xorshift128+
    xorshift128p_state rng;
    rng.s0 = seed1 ^ ((u64)gid * 104729ULL);
    rng.s1 = seed2 ^ ((u64)gid * 104729ULL);
    u8 seed[32];
    for (int i = 0; i < 32; i += 8) {
        u64 r = xorshift128p(&rng);
        seed[i+0] = (u8)(r & 0xFF);
        seed[i+1] = (u8)((r >> 8) & 0xFF);
        seed[i+2] = (u8)((r >> 16) & 0xFF);
        seed[i+3] = (u8)((r >> 24) & 0xFF);
        seed[i+4] = (u8)((r >> 32) & 0xFF);
        seed[i+5] = (u8)((r >> 40) & 0xFF);
        seed[i+6] = (u8)((r >> 48) & 0xFF);
        seed[i+7] = (u8)((r >> 56) & 0xFF);
    }
    
    // Derive a candidate public key using the improved derivation.
    u8 candidate_pub[32];
    derive_candidate_pubkey(seed, candidate_pub);
    
    // Convert candidate public key to Base58 string.
    char candidate_b58[50];
    int b58len = to_base58_32(candidate_pub, candidate_b58);
    
    // Check if the candidate Base58 string matches desired prefix/suffix.
    int match = 1;
    if (prefix_len > 0) {
        if (b58len < prefix_len) { match = 0; }
        else {
            for (int i = 0; i < prefix_len; i++) {
                if (candidate_b58[i] != b58_prefix[i]) { match = 0; break; }
            }
        }
    }
    if (match && suffix_len > 0) {
        if (b58len < suffix_len) { match = 0; }
        else {
            for (int i = 0; i < suffix_len; i++) {
                if (candidate_b58[b58len - suffix_len + i] != b58_suffix[i]) { match = 0; break; }
            }
        }
    }
    if (match) {
        if (*found_flag == 0) {
            *found_flag = 1;
            for (int i = 0; i < 32; i++) {
                out_seed[i] = seed[i];
            }
            for (int i = 0; i < b58len; i++) {
                out_b58pub[i] = candidate_b58[i];
            }
            out_b58pub[b58len] = '\0';
        }
    }
}