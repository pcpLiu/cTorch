#define JEMALLOC_THREAD_EVENT_C_
#include "jemalloc/internal/jemalloc_preamble.h"
#include "jemalloc/internal/jemalloc_internal_includes.h"

#include "jemalloc/internal/thread_event.h"

/*
 * There's no lock for thread_event_active because write is only done in
 * malloc_init(), where init_lock there serves as the guard, and ever since
 * then thread_event_active becomes read only.
 */
static bool thread_event_active = false;

/* TSD event init function signatures. */
#define E(event, condition)						\
static void tsd_thread_##event##_event_init(tsd_t *tsd);

ITERATE_OVER_ALL_EVENTS
#undef E

/* Event handler function signatures. */
#define E(event, condition)						\
static void thread_##event##_event_handler(tsd_t *tsd);

ITERATE_OVER_ALL_EVENTS
#undef E

static void
tsd_thread_tcache_gc_event_init(tsd_t *tsd) {
	assert(TCACHE_GC_INCR_BYTES > 0);
	thread_tcache_gc_event_update(tsd, TCACHE_GC_INCR_BYTES);
}

static void
tsd_thread_prof_sample_event_init(tsd_t *tsd) {
	assert(config_prof && opt_prof);
	prof_sample_threshold_update(tsd);
}

static void
thread_tcache_gc_event_handler(tsd_t *tsd) {
	assert(TCACHE_GC_INCR_BYTES > 0);
	assert(tcache_gc_event_wait_get(tsd) == 0U);
	thread_tcache_gc_event_update(tsd, TCACHE_GC_INCR_BYTES);
	tcache_t *tcache = tcache_get(tsd);
	if (tcache != NULL) {
		tcache_event_hard(tsd, tcache);
	}
}

static void
thread_prof_sample_event_handler(tsd_t *tsd) {
	assert(config_prof && opt_prof);
	assert(prof_sample_event_wait_get(tsd) == 0U);
	uint64_t last_event = thread_allocated_last_event_get(tsd);
	uint64_t last_sample_event = prof_sample_last_event_get(tsd);
	prof_sample_last_event_set(tsd, last_event);
	if (prof_idump_accum(tsd_tsdn(tsd), last_event - last_sample_event)) {
		prof_idump(tsd_tsdn(tsd));
	}
	if (!prof_active_get_unlocked()) {
		/*
		 * If prof_active is off, we reset prof_sample_event_wait to be
		 * the sample interval when it drops to 0, so that there won't
		 * be excessive routings to the slow path, and that when
		 * prof_active is turned on later, the counting for sampling
		 * can immediately resume as normal.
		 */
		thread_prof_sample_event_update(tsd,
		    (uint64_t)(1 << lg_prof_sample));
	}
}

static uint64_t
thread_allocated_next_event_compute(tsd_t *tsd) {
	uint64_t wait = THREAD_EVENT_MAX_START_WAIT;
	bool no_event_on = true;

#define E(event, condition)						\
	if (condition) {						\
		no_event_on = false;					\
		uint64_t event_wait =					\
		    event##_event_wait_get(tsd);			\
		assert(event_wait <= THREAD_EVENT_MAX_START_WAIT);	\
		if (event_wait > 0U && event_wait < wait) {		\
			wait = event_wait;				\
		}							\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E

	assert(no_event_on == !thread_event_active);
	assert(wait <= THREAD_EVENT_MAX_START_WAIT);
	return wait;
}

void
thread_event_assert_invariants_debug(tsd_t *tsd) {
	uint64_t thread_allocated = thread_allocated_get(tsd);
	uint64_t last_event = thread_allocated_last_event_get(tsd);
	uint64_t next_event = thread_allocated_next_event_get(tsd);
	uint64_t next_event_fast = thread_allocated_next_event_fast_get(tsd);

	assert(last_event != next_event);
	if (next_event > THREAD_ALLOCATED_NEXT_EVENT_FAST_MAX ||
	    !tsd_fast(tsd)) {
		assert(next_event_fast == 0U);
	} else {
		assert(next_event_fast == next_event);
	}

	/* The subtraction is intentionally susceptible to underflow. */
	uint64_t interval = next_event - last_event;

	/* The subtraction is intentionally susceptible to underflow. */
	assert(thread_allocated - last_event < interval);

	uint64_t min_wait = thread_allocated_next_event_compute(tsd);

	/*
	 * next_event should have been pushed up only except when no event is
	 * on and the TSD is just initialized.  The last_event == 0U guard
	 * below is stronger than needed, but having an exactly accurate guard
	 * is more complicated to implement.
	 */
	assert((!thread_event_active && last_event == 0U) ||
	    interval == min_wait ||
	    (interval < min_wait && interval == THREAD_EVENT_MAX_INTERVAL));
}

/*
 * Synchronization around the fast threshold in tsd --
 * There are two threads to consider in the synchronization here:
 * - The owner of the tsd being updated by a slow path change
 * - The remote thread, doing that slow path change.
 *
 * As a design constraint, we want to ensure that a slow-path transition cannot
 * be ignored for arbitrarily long, and that if the remote thread causes a
 * slow-path transition and then communicates with the owner thread that it has
 * occurred, then the owner will go down the slow path on the next allocator
 * operation (so that we don't want to just wait until the owner hits its slow
 * path reset condition on its own).
 *
 * Here's our strategy to do that:
 *
 * The remote thread will update the slow-path stores to TSD variables, issue a
 * SEQ_CST fence, and then update the TSD next_event_fast counter. The owner
 * thread will update next_event_fast, issue an SEQ_CST fence, and then check
 * its TSD to see if it's on the slow path.

 * This is fairly straightforward when 64-bit atomics are supported. Assume that
 * the remote fence is sandwiched between two owner fences in the reset pathway.
 * The case where there is no preceding or trailing owner fence (i.e. because
 * the owner thread is near the beginning or end of its life) can be analyzed
 * similarly. The owner store to next_event_fast preceding the earlier owner
 * fence will be earlier in coherence order than the remote store to it, so that
 * the owner thread will go down the slow path once the store becomes visible to
 * it, which is no later than the time of the second fence.

 * The case where we don't support 64-bit atomics is trickier, since word
 * tearing is possible. We'll repeat the same analysis, and look at the two
 * owner fences sandwiching the remote fence. The next_event_fast stores done
 * alongside the earlier owner fence cannot overwrite any of the remote stores
 * (since they precede the earlier owner fence in sb, which precedes the remote
 * fence in sc, which precedes the remote stores in sb). After the second owner
 * fence there will be a re-check of the slow-path variables anyways, so the
 * "owner will notice that it's on the slow path eventually" guarantee is
 * satisfied. To make sure that the out-of-band-messaging constraint is as well,
 * note that either the message passing is sequenced before the second owner
 * fence (in which case the remote stores happen before the second set of owner
 * stores, so malloc sees a value of zero for next_event_fast and goes down the
 * slow path), or it is not (in which case the owner sees the tsd slow-path
 * writes on its previous update). This leaves open the possibility that the
 * remote thread will (at some arbitrary point in the future) zero out one half
 * of the owner thread's next_event_fast, but that's always safe (it just sends
 * it down the slow path earlier).
 */
void
thread_event_recompute_fast_threshold(tsd_t *tsd) {
	if (tsd_state_get(tsd) != tsd_state_nominal) {
		/* Check first because this is also called on purgatory. */
		thread_allocated_next_event_fast_set_non_nominal(tsd);
		return;
	}
	uint64_t next_event = thread_allocated_next_event_get(tsd);
	uint64_t next_event_fast = (next_event <=
	    THREAD_ALLOCATED_NEXT_EVENT_FAST_MAX) ? next_event : 0U;
	thread_allocated_next_event_fast_set(tsd, next_event_fast);

	atomic_fence(ATOMIC_SEQ_CST);
	if (tsd_state_get(tsd) != tsd_state_nominal) {
		thread_allocated_next_event_fast_set_non_nominal(tsd);
	}
}

static void
thread_event_adjust_thresholds_helper(tsd_t *tsd, uint64_t wait) {
	assert(wait <= THREAD_EVENT_MAX_START_WAIT);
	uint64_t next_event = thread_allocated_last_event_get(tsd) + (wait <=
	    THREAD_EVENT_MAX_INTERVAL ? wait : THREAD_EVENT_MAX_INTERVAL);
	thread_allocated_next_event_set(tsd, next_event);
}

static uint64_t
thread_event_trigger_batch_update(tsd_t *tsd, uint64_t accumbytes,
    bool allow_event_trigger) {
	uint64_t wait = THREAD_EVENT_MAX_START_WAIT;

#define E(event, condition)						\
	if (condition) {						\
		uint64_t event_wait = event##_event_wait_get(tsd);	\
		assert(event_wait <= THREAD_EVENT_MAX_START_WAIT);	\
		if (event_wait > accumbytes) {				\
			event_wait -= accumbytes;			\
		} else {						\
			event_wait = 0U;				\
			if (!allow_event_trigger) {			\
				event_wait =				\
				    THREAD_EVENT_MIN_START_WAIT;	\
			}						\
		}							\
		assert(event_wait <= THREAD_EVENT_MAX_START_WAIT);	\
		event##_event_wait_set(tsd, event_wait);		\
		/*							\
		 * If there is a single event, then the remaining wait	\
		 * time may become zero, and we rely on either the	\
		 * event handler or a thread_event_update() call later	\
		 * to properly set next_event; if there are multiple	\
		 * events, then	here we can get the minimum remaining	\
		 * wait time to	the next already set event.		\
		 */							\
		if (event_wait > 0U && event_wait < wait) {		\
			wait = event_wait;				\
		}							\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E

	assert(wait <= THREAD_EVENT_MAX_START_WAIT);
	return wait;
}

void
thread_event_trigger(tsd_t *tsd, bool delay_event) {
	/* usize has already been added to thread_allocated. */
	uint64_t thread_allocated_after = thread_allocated_get(tsd);

	/* The subtraction is intentionally susceptible to underflow. */
	uint64_t accumbytes = thread_allocated_after -
	    thread_allocated_last_event_get(tsd);

	/* Make sure that accumbytes cannot overflow uint64_t. */
	assert(THREAD_EVENT_MAX_INTERVAL <= UINT64_MAX - SC_LARGE_MAXCLASS + 1);

	thread_allocated_last_event_set(tsd, thread_allocated_after);
	bool allow_event_trigger = !delay_event && tsd_nominal(tsd) &&
	    tsd_reentrancy_level_get(tsd) == 0;
	uint64_t wait = thread_event_trigger_batch_update(tsd, accumbytes,
	    allow_event_trigger);
	thread_event_adjust_thresholds_helper(tsd, wait);

	thread_event_assert_invariants(tsd);

#define E(event, condition)						\
	if (condition && event##_event_wait_get(tsd) == 0U) {		\
		assert(allow_event_trigger);				\
		thread_##event##_event_handler(tsd);			\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E

	thread_event_assert_invariants(tsd);
}

void
thread_event_rollback(tsd_t *tsd, size_t diff) {
	thread_event_assert_invariants(tsd);

	if (diff == 0U) {
		return;
	}

	uint64_t thread_allocated = thread_allocated_get(tsd);
	/* The subtraction is intentionally susceptible to underflow. */
	uint64_t thread_allocated_rollback = thread_allocated - diff;
	thread_allocated_set(tsd, thread_allocated_rollback);

	uint64_t last_event = thread_allocated_last_event_get(tsd);
	/* Both subtractions are intentionally susceptible to underflow. */
	if (thread_allocated_rollback - last_event <=
	    thread_allocated - last_event) {
		thread_event_assert_invariants(tsd);
		return;
	}

	thread_allocated_last_event_set(tsd, thread_allocated_rollback);

	/* The subtraction is intentionally susceptible to underflow. */
	uint64_t wait_diff = last_event - thread_allocated_rollback;
	assert(wait_diff <= diff);

#define E(event, condition)						\
	if (condition) {						\
		uint64_t event_wait = event##_event_wait_get(tsd);	\
		assert(event_wait <= THREAD_EVENT_MAX_START_WAIT);	\
		if (event_wait > 0U) {					\
			if (wait_diff >					\
			    THREAD_EVENT_MAX_START_WAIT - event_wait) {	\
				event_wait =				\
				    THREAD_EVENT_MAX_START_WAIT;	\
			} else {					\
				event_wait += wait_diff;		\
			}						\
			assert(event_wait <=				\
			    THREAD_EVENT_MAX_START_WAIT);		\
			event##_event_wait_set(tsd, event_wait);	\
		}							\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E

	thread_event_update(tsd);
}

void
thread_event_update(tsd_t *tsd) {
	uint64_t wait = thread_allocated_next_event_compute(tsd);
	thread_event_adjust_thresholds_helper(tsd, wait);

	uint64_t last_event = thread_allocated_last_event_get(tsd);

	/* Both subtractions are intentionally susceptible to underflow. */
	if (thread_allocated_get(tsd) - last_event >=
	    thread_allocated_next_event_get(tsd) - last_event) {
		thread_event_trigger(tsd, true);
	} else {
		thread_event_assert_invariants(tsd);
	}
}

void thread_event_boot() {
#define E(event, condition)						\
	if (condition) {						\
		thread_event_active = true;				\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E
}

void tsd_thread_event_init(tsd_t *tsd) {
#define E(event, condition)						\
	if (condition) {						\
		tsd_thread_##event##_event_init(tsd);			\
	}

	ITERATE_OVER_ALL_EVENTS
#undef E
}
