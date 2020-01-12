#ifndef JEMALLOC_INTERNAL_EXTENT_H
#define JEMALLOC_INTERNAL_EXTENT_H

#include "jemalloc/internal/ecache.h"
#include "jemalloc/internal/ehooks.h"
#include "jemalloc/internal/ph.h"
#include "jemalloc/internal/rtree.h"

/*
 * This module contains the page-level allocator.  It chooses the addresses that
 * allocations requested by other modules will inhabit, and updates the global
 * metadata to reflect allocation/deallocation/purging decisions.
 */

/*
 * When reuse (and split) an active extent, (1U << opt_lg_extent_max_active_fit)
 * is the max ratio between the size of the active extent and the new extent.
 */
#define LG_EXTENT_MAX_ACTIVE_FIT_DEFAULT 6
extern size_t opt_lg_extent_max_active_fit;

extern rtree_t extents_rtree;

edata_t *ecache_alloc(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    ecache_t *ecache, void *new_addr, size_t size, size_t pad, size_t alignment,
    bool slab, szind_t szind, bool *zero, bool *commit);
edata_t *ecache_alloc_grow(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    ecache_t *ecache, void *new_addr, size_t size, size_t pad, size_t alignment,
    bool slab, szind_t szind, bool *zero, bool *commit);
void ecache_dalloc(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    ecache_t *ecache, edata_t *edata);
edata_t *ecache_evict(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    ecache_t *ecache, size_t npages_min);

edata_t *extent_alloc_wrapper(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    void *new_addr, size_t size, size_t pad, size_t alignment, bool slab,
    szind_t szind, bool *zero, bool *commit);
void extent_dalloc_gap(tsdn_t *tsdn, arena_t *arena, edata_t *edata);
void extent_dalloc_wrapper(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    edata_t *edata);
void extent_destroy_wrapper(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    edata_t *edata);
bool extent_commit_wrapper(tsdn_t *tsdn, ehooks_t *ehooks, edata_t *edata,
    size_t offset, size_t length);
bool extent_decommit_wrapper(tsdn_t *tsdn, ehooks_t *ehooks, edata_t *edata,
    size_t offset, size_t length);
bool extent_purge_lazy_wrapper(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    edata_t *edata, size_t offset, size_t length);
bool extent_purge_forced_wrapper(tsdn_t *tsdn, arena_t *arena, ehooks_t *ehooks,
    edata_t *edata, size_t offset, size_t length);
edata_t *extent_split_wrapper(tsdn_t *tsdn, edata_cache_t *edata_cache,
    ehooks_t *ehooks, edata_t *edata, size_t size_a, szind_t szind_a,
    bool slab_a, size_t size_b, szind_t szind_b, bool slab_b);
bool extent_merge_wrapper(tsdn_t *tsdn, ehooks_t *ehooks,
    edata_cache_t *edata_cache, edata_t *a, edata_t *b);

bool extent_boot(void);

#endif /* JEMALLOC_INTERNAL_EXTENT_H */
