// orion_weight_patch.m — Weight Patching API Implementation

#import "orion_weight_patch.h"
#import "orion_mil_cache.h"
#import <stdio.h>
#import <stdlib.h>

struct OrionWeightPatchCtx {
    char *mil_text;           // Owned
    OrionProgram *donor;      // Owned - master program for patching
    int patch_count;          // Number of patches applied
    char *tag;                // Owned - debug tag
};

OrionWeightPatchCtx* orion_weight_patch_create(
    const char *mil_text,
    NSDictionary *wdict,
    const char *tag
) {
    if (!mil_text || !wdict) {
        fprintf(stderr, "orion_weight_patch_create: invalid args\n");
        return NULL;
    }

    // Compile once to create donor
    OrionProgram *donor = orion_mil_cache_get(mil_text, wdict, tag);
    if (!donor) {
        fprintf(stderr, "orion_weight_patch_create: donor compile failed\n");
        return NULL;
    }

    OrionWeightPatchCtx *ctx = (OrionWeightPatchCtx *)calloc(1, sizeof(OrionWeightPatchCtx));
    ctx->mil_text = strdup(mil_text);
    ctx->donor = donor;
    ctx->tag = tag ? strdup(tag) : NULL;
    ctx->patch_count = 0;

    return ctx;
}

OrionProgram* orion_weight_patch_get(
    OrionWeightPatchCtx *ctx,
    NSDictionary *wdict,
    const char *tag
) {
    if (!ctx || !wdict) {
        fprintf(stderr, "orion_weight_patch_get: invalid args\n");
        return NULL;
    }

    if (!ctx->donor) {
        fprintf(stderr, "orion_weight_patch_get: no donor program\n");
        return NULL;
    }

    // Use orion_program_patch_weights to create patched copy
    OrionProgram *patched = orion_program_patch_weights(
        ctx->donor, ctx->mil_text, wdict, tag
    );

    if (patched) {
        ctx->patch_count++;
    }

    return patched;
}

int orion_weight_patch_count(OrionWeightPatchCtx *ctx) {
    return ctx ? ctx->patch_count : 0;
}

bool orion_weight_patch_near_limit(
    OrionWeightPatchCtx *ctx,
    int warn_threshold
) {
    if (!ctx) return false;

    // Check current compile count vs threshold
    int compile_count = orion_compile_count();
    return (compile_count + ctx->patch_count) >= warn_threshold;
}

void orion_weight_patch_destroy(OrionWeightPatchCtx *ctx) {
    if (!ctx) return;

    if (ctx->donor) {
        orion_release_program(ctx->donor);
    }
    if (ctx->mil_text) {
        free(ctx->mil_text);
    }
    if (ctx->tag) {
        free(ctx->tag);
    }
    free(ctx);
}
