// test_rns_api.m — Test Orion RNS API
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/orion_rns.m \
//     tests/test_rns_api.m -o test_rns_api
//
// Run:
//   ./test_rns_api

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "rns.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ============================================================================
// Test Data
// ============================================================================

static const RNSMod kTinyMod[5] = {
    { 3, "q0" }, { 5, "q1" }, { 7, "q2" }, { 11, "q3" }, { 13, "q4" },
};

static const RNSMod kProdMod[7] = {
    { 97, "p0" }, { 101, "p1" }, { 103, "p2" }, { 107, "p3" },
    { 109, "p4" }, { 113, "p5" }, { 127, "p6" },
};

// ============================================================================
// Tests
// ============================================================================

static void test_extended_gcd(void) {
    printf("\n=== Test: Extended GCD ===\n");

    struct { int64_t a, b; int64_t exp_g; } tests[] = {
        {12, 8, 4},
        {35, 15, 5},
        {7, 5, 1},
        {1, 1, 1},
    };

    for (int i = 0; i < 4; i++) {
        int64_t x, y;
        int64_t g = orion_extended_gcd(tests[i].a, tests[i].b, &x, &y);
        CHECK(g == tests[i].exp_g, "correct gcd");
        // Verify x*a + y*b = g
        int64_t verify = x * tests[i].a + y * tests[i].b;
        CHECK(verify == g, "Bezout identity holds");
    }
}

static void test_crt_simple(void) {
    printf("\n=== Test: CRT Simple ===\n");

    // Test: x = 12345
    uint32_t residues[5];
    uint64_t x = 12345;

    orion_rns_decompose(x, kTinyMod, 5, residues);
    printf("  x=%llu -> residues: ", (unsigned long long)x);
    for (int i = 0; i < 5; i++) printf("%u ", residues[i]);
    printf("\n");

    uint64_t reconstructed = orion_crt_reconstruct(residues, kTinyMod, 5);
    CHECK(reconstructed == x, "CRT reconstructs correctly");
}

static void test_crt_product(void) {
    printf("\n=== Test: RNS Product ===\n");

    uint64_t M = orion_rns_product(kTinyMod, 5);
    printf("  Product of {3,5,7,11,13} = %llu\n", (unsigned long long)M);
    CHECK(M == 15015, "product correct");

    double bits = orion_rns_bits(kTinyMod, 5);
    printf("  Bit width: %.2f bits\n", bits);
    CHECK(bits > 13 && bits < 15, "bit width correct (~14 bits)");

    uint64_t M_prod = orion_rns_product(kProdMod, 7);
    double bits_prod = orion_rns_bits(kProdMod, 7);
    printf("  Product of 7 moduli: %llu (~%.1f bits)\n", (unsigned long long)M_prod, bits_prod);
    CHECK(bits_prod > 47 && bits_prod < 48, "production bit width correct (~47 bits)");
}

static void test_rns_decompose(void) {
    printf("\n=== Test: RNS Decompose ===\n");

    struct { uint64_t x; uint32_t exp[5]; } tests[] = {
        {0, {0, 0, 0, 0, 0}},
        {1, {1, 1, 1, 1, 1}},
        {15014, {2, 4, 6, 10, 12}},  // M-1 case (15014 % mod)
        {12345, {0, 0, 4, 3, 8}},
    };

    for (int i = 0; i < 4; i++) {
        uint32_t residues[5];
        orion_rns_decompose(tests[i].x, kTinyMod, 5, residues);
        bool match = true;
        for (int j = 0; j < 5; j++) {
            if (residues[j] != tests[i].exp[j]) match = false;
        }
        printf("  x=%llu -> [%u,%u,%u,%u,%u]",
               (unsigned long long)tests[i].x,
               residues[0], residues[1], residues[2], residues[3], residues[4]);
        if (match) {
            printf(" PASS\n");
            g_pass++;
        } else {
            printf(" FAIL (expected [%u,%u,%u,%u,%u])\n",
                   tests[i].exp[0], tests[i].exp[1], tests[i].exp[2],
                   tests[i].exp[3], tests[i].exp[4]);
            g_fail++;
        }
    }
}

static void test_tile_layout(void) {
    printf("\n=== Test: Tile Layout ===\n");

    struct { int dim; int max_tile; int exp_tiles; } tests[] = {
        {256, 2048, 1},
        {1024, 1024, 1},
        {2048, 1024, 2},
        {4096, 1024, 4},
        {4096, 2048, 2},
        {8192, 2048, 4},
    };

    for (int i = 0; i < 6; i++) {
        TileLayout tile;
        orion_tile_layout_init(&tile, tests[i].dim, tests[i].max_tile);

        printf("  dim=%d, max_tile=%d -> %d tiles (sizes: %d",
               tests[i].dim, tests[i].max_tile, tile.n_tiles, tile.tile_size);
        if (tile.n_tiles > 1) {
            printf(", %d", tile.last_tile_size);
        }
        printf(")\n");

        CHECK(tile.n_tiles == tests[i].exp_tiles, "tile count correct");

        // Verify offsets
        for (int j = 0; j < tile.n_tiles; j++) {
            CHECK(orion_tile_offset_at(&tile, j) == j * tests[i].max_tile, "offset correct");
        }

        orion_tile_layout_free(&tile);
    }
}

static void test_tile_layout_sizes(void) {
    printf("\n=== Test: Tile Sizes ===\n");

    TileLayout tile;
    orion_tile_layout_init(&tile, 5000, 2048);

    printf("  dim=5000, max_tile=2048 -> %d tiles\n", tile.n_tiles);

    // First tile should be 2048
    CHECK(orion_tile_size_at(&tile, 0) == 2048, "first tile size");
    CHECK(orion_tile_offset_at(&tile, 0) == 0, "first tile offset");

    // Second tile should be 2048
    CHECK(orion_tile_size_at(&tile, 1) == 2048, "second tile size");
    CHECK(orion_tile_offset_at(&tile, 1) == 2048, "second tile offset");

    // Last tile should be 904 (5000 - 2*2048)
    CHECK(orion_tile_size_at(&tile, 2) == 904, "last tile size");
    CHECK(orion_tile_offset_at(&tile, 2) == 4096, "last tile offset");

    // Total should cover full dimension
    int total = 0;
    for (int i = 0; i < tile.n_tiles; i++) {
        total += orion_tile_size_at(&tile, i);
    }
    CHECK(total == 5000, "total size matches dim");

    orion_tile_layout_free(&tile);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    printf("=== Orion RNS API Test ===\n");

    test_extended_gcd();
    test_crt_simple();
    test_crt_product();
    test_rns_decompose();
    test_tile_layout();
    test_tile_layout_sizes();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
