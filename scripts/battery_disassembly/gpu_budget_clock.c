/* Kit 110.4 GPU memory-budget history wraps negative wall-clock deltas to uint64.
 * Use an epoch-anchored monotonic clock for that library, retaining wall time for
 * all other callers. This is a process-local, opt-in workaround; no SDK files or
 * system clocks are changed. The ABI below is tested on Linux x86_64.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t epoch, anchor, last, real_max;
static unsigned long calls;

int rsr_budget_clock_version(void) { return 1; }

static int64_t ns(clockid_t id) {
    struct timespec t;
    clock_gettime(id, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void initialize(void) {
    anchor = ns(CLOCK_MONOTONIC);
    epoch = ns(CLOCK_REALTIME);
    last = epoch;
    real_max = epoch;
}

/* libstdc++ returns this trivially-copyable 64-bit time_point in RAX. */
int64_t budget_now(void) __asm__("_ZNSt6chrono3_V212system_clock3nowEv");
int64_t budget_now(void) {
    Dl_info caller;
    if (!dladdr(__builtin_return_address(0), &caller) || !caller.dli_fname ||
        !strstr(caller.dli_fname, "/libgpu.foundation.plugin.so")) {
        return ns(CLOCK_REALTIME);
    }
    pthread_once(&once, initialize);
    pthread_mutex_lock(&lock);
    int64_t real = ns(CLOCK_REALTIME);
    int64_t value = epoch + ns(CLOCK_MONOTONIC) - anchor;
    if (value <= last) value = last + 1;
    if (calls++ == 0)
        dprintf(STDERR_FILENO, "[rsr-budget-clock] active pid=%d caller=%s\n", getpid(), caller.dli_fname);
    if (real < real_max - 1000000LL)
        dprintf(STDERR_FILENO, "[rsr-budget-clock] realtime reversal pid=%d ns=%lld\n", getpid(), (long long)(real - real_max));
    if (real > real_max) real_max = real;
    last = value;
    pthread_mutex_unlock(&lock);
    return value;
}
