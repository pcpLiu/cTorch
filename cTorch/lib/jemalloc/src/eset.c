#include "jemalloc/internal/jemalloc_preamble.h"
#include "jemalloc/internal/jemalloc_internal_includes.h"

#include "jemalloc/internal/eset.h"
/* For opt_retain */
#include "jemalloc/internal/extent_mmap.h"

const bitmap_info_t eset_bitmap_info =
    BITMAP_INFO_INITIALIZER(SC_NPSIZES+1);

void
eset_init(eset_t *eset, extent_state_t state) {
	for (unsigned i = 0; i < SC_NPSIZES + 1; i++) {
		edata_heap_new(&eset->heaps[i]);
	}
	bitmap_init(eset->bitmap, &eset_bitmap_info, true);
	edata_list_init(&eset->lru);
	atomic_store_zu(&eset->npages, 0, ATOMIC_RELAXED);
	eset->state = state;
}

size_t
eset_npages_get(eset_t *eset) {
	return atomic_load_zu(&eset->npages, ATOMIC_RELAXED);
}

size_t
eset_nextents_get(eset_t *eset, pszind_t pind) {
	return atomic_load_zu(&eset->nextents[pind], ATOMIC_RELAXED);
}

size_t
eset_nbytes_get(eset_t *eset, pszind_t pind) {
	return atomic_load_zu(&eset->nbytes[pind], ATOMIC_RELAXED);
}

static void
eset_stats_add(eset_t *eset, pszind_t pind, size_t sz) {
	size_t cur = atomic_load_zu(&eset->nextents[pind], ATOMIC_RELAXED);
	atomic_store_zu(&eset->nextents[pind], cur + 1, ATOMIC_RELAXED);
	cur = atomic_load_zu(&eset->nbytes[pind], ATOMIC_RELAXED);
	atomic_store_zu(&eset->nbytes[pind], cur + sz, ATOMIC_RELAXED);
}

static void
eset_stats_sub(eset_t *eset, pszind_t pind, size_t sz) {
	size_t cur = atomic_load_zu(&eset->nextents[pind], ATOMIC_RELAXED);
	atomic_store_zu(&eset->nextents[pind], cur - 1, ATOMIC_RELAXED);
	cur = atomic_load_zu(&eset->nbytes[pind], ATOMIC_RELAXED);
	atomic_store_zu(&eset->nbytes[pind], cur - sz, ATOMIC_RELAXED);
}

void
eset_insert(eset_t *eset, edata_t *edata) {
	assert(edata_state_get(edata) == eset->state);

	size_t size = edata_size_get(edata);
	size_t psz = sz_psz_quantize_floor(size);
	pszind_t pind = sz_psz2ind(psz);
	if (edata_heap_empty(&eset->heaps[pind])) {
		bitmap_unset(eset->bitmap, &eset_bitmap_info,
		    (size_t)pind);
	}
	edata_heap_insert(&eset->heaps[pind], edata);

	if (config_stats) {
		eset_stats_add(eset, pind, size);
	}

	edata_list_append(&eset->lru, edata);
	size_t npages = size >> LG_PAGE;
	/*
	 * All modifications to npages hold the mutex (as asserted above), so we
	 * don't need an atomic fetch-add; we can get by with a load followed by
	 * a store.
	 */
	size_t cur_eset_npages =
	    atomic_load_zu(&eset->npages, ATOMIC_RELAXED);
	atomic_store_zu(&eset->npages, cur_eset_npages + npages,
	    ATOMIC_RELAXED);
}

void
eset_remove(eset_t *eset, edata_t *edata) {
	assert(edata_state_get(edata) == eset->state);

	size_t size = edata_size_get(edata);
	size_t psz = sz_psz_quantize_floor(size);
	pszind_t pind = sz_psz2ind(psz);
	edata_heap_remove(&eset->heaps[pind], edata);

	if (config_stats) {
		eset_stats_sub(eset, pind, size);
	}

	if (edata_heap_empty(&eset->heaps[pind])) {
		bitmap_set(eset->bitmap, &eset_bitmap_info,
		    (size_t)pind);
	}
	edata_list_remove(&eset->lru, edata);
	size_t npages = size >> LG_PAGE;
	/*
	 * As in eset_insert, we hold eset->mtx and so don't need atomic
	 * operations for updating eset->npages.
	 */
	/*
	 * This class is not thread-safe in general; we rely on external
	 * synchronization for all mutating operations.
	 */
	size_t cur_extents_npages =
	    atomic_load_zu(&eset->npages, ATOMIC_RELAXED);
	assert(cur_extents_npages >= npages);
	atomic_store_zu(&eset->npages,
	    cur_extents_npages - (size >> LG_PAGE), ATOMIC_RELAXED);
}

/*
 * Find an extent with size [min_size, max_size) to satisfy the alignment
 * requirement.  For each size, try only the first extent in the heap.
 */
static edata_t *
eset_fit_alignment(eset_t *eset, size_t min_size, size_t max_size,
    size_t alignment) {
        pszind_t pind = sz_psz2ind(sz_psz_quantize_ceil(min_size));
        pszind_t pind_max = sz_psz2ind(sz_psz_quantize_ceil(max_size));

	for (pszind_t i = (pszind_t)bitmap_ffu(eset->bitmap,
	    &eset_bitmap_info, (size_t)pind); i < pind_max; i =
	    (pszind_t)bitmap_ffu(eset->bitmap, &eset_bitmap_info,
	    (size_t)i+1)) {
		assert(i < SC_NPSIZES);
		assert(!edata_heap_empty(&eset->heaps[i]));
		edata_t *edata = edata_heap_first(&eset->heaps[i]);
		uintptr_t base = (uintptr_t)edata_base_get(edata);
		size_t candidate_size = edata_size_get(edata);
		assert(candidate_size >= min_size);

		uintptr_t next_align = ALIGNMENT_CEILING((uintptr_t)base,
		    PAGE_CEILING(alignment));
		if (base > next_align || base + candidate_size <= next_align) {
			/* Overflow or not crossing the next alignment. */
			continue;
		}

		size_t leadsize = next_align - base;
		if (candidate_size - leadsize >= min_size) {
			return edata;
		}
	}

	return NULL;
}

/*
 * Do first-fit extent selection, i.e. select the oldest/lowest extent that is
 * large enough.
 */
static edata_t *
eset_first_fit(eset_t *eset, size_t size, bool delay_coalesce) {
	edata_t *ret = NULL;

	pszind_t pind = sz_psz2ind(sz_psz_quantize_ceil(size));

	if (!maps_coalesce && !opt_retain) {
		/*
		 * No split / merge allowed (Windows w/o retain). Try exact fit
		 * only.
		 */
		return edata_heap_empty(&eset->heaps[pind]) ? NULL :
		    edata_heap_first(&eset->heaps[pind]);
	}

	for (pszind_t i = (pszind_t)bitmap_ffu(eset->bitmap,
	    &eset_bitmap_info, (size_t)pind);
	    i < SC_NPSIZES + 1;
	    i = (pszind_t)bitmap_ffu(eset->bitmap, &eset_bitmap_info,
	    (size_t)i+1)) {
		assert(!edata_heap_empty(&eset->heaps[i]));
		edata_t *edata = edata_heap_first(&eset->heaps[i]);
		assert(edata_size_get(edata) >= size);
		/*
		 * In order to reduce fragmentation, avoid reusing and splitting
		 * large eset for much smaller sizes.
		 *
		 * Only do check for dirty eset (delay_coalesce).
		 */
		if (delay_coalesce &&
		    (sz_pind2sz(i) >> opt_lg_extent_max_active_fit) > size) {
			break;
		}
		if (ret == NULL || edata_snad_comp(edata, ret) < 0) {
			ret = edata;
		}
		if (i == SC_NPSIZES) {
			break;
		}
		assert(i < SC_NPSIZES);
	}

	return ret;
}

edata_t *
eset_fit(eset_t *eset, size_t esize, size_t alignment, bool delay_coalesce) {
	size_t max_size = esize + PAGE_CEILING(alignment) - PAGE;
	/* Beware size_t wrap-around. */
	if (max_size < esize) {
		return NULL;
	}

	edata_t *edata = eset_first_fit(eset, max_size, delay_coalesce);

	if (alignment > PAGE && edata == NULL) {
		/*
		 * max_size guarantees the alignment requirement but is rather
		 * pessimistic.  Next we try to satisfy the aligned allocation
		 * with sizes in [esize, max_size).
		 */
		edata = eset_fit_alignment(eset, esize, max_size, alignment);
	}

	return edata;
}
