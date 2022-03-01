**Write-back cache**: Store modified contents in cache without immediate changes to memory. Only update memory when the line is "evicted". Cache tracks which lines have been modified.

**Write-back miss**

```
int x = 1; (&x = 0x12345604)

1. Processor performs store at 0x12345604 which is not in cache.
2. If the cache location has a dirty line, it is written to memory.
3. Cache loads an entire line containing 0x12345604.
4. Bytes [0x12345604-0x12345608) are modified.
5. Cache line is marked as dirty.
```

**Cache coherence problem**: Every processor has local caches. With write-back cache, processors can observe different values for the same memory location.

The problem exists because there is both global storage (main memory) and local storage (caches) implementing the abstraction of a single shared address space.

This is an issue in single-CPU system too. For example, network card may transfer stale data if the processor's writes have not been flushed back to memory.

**Coference definition**: Memory operations issued by any one processor occur in the order issued by the processor + The value returned by a read is the value written by the last write to the location.

**Simple solution - shared cache**: One single cache shared by all processors. Cache becomes less local thus slower.

**Snooping cache**: All coherence-related activity is broadcast to all processors. Cache controllers monitor memory operations to maintain its own memory coherence.

**Simple coherence implementation**: Processors write directly to memory. When P1 writes to memory, cache in P2 is invalidated. P2 will have a cache miss on the next read.

**State diagram for simple coherence**: Either a cache line is valid or invalid (Not in cache). 

If the cache line is invalid, when the processor asks to read (`prRd`), the cache line puts a message on bus (`BusRd`: Retrieving data from memory) and moves to a valid state. If the cache line is valid, when the processor asks to read (`prRd`), the cache line does nothing.

If the cache line is valid, when the processor writes (`PrWr`), the cache line puts a message on bus (`BusWr`: Data has been written to memory) and remains valid.

If the cache line is valid, when the processor hears another processor write to memory (`BusWr`), the cache line moves to an invalid state.

![](images/Pasted%20image%2020220301162646.png)

**MSI cache coherence**: Cache line can only be written to if no other cache has the data.

There are two valid states (Shared, Modified) or invalid state. Data in the shared state means it is valid but other caches may have a copy. Data in the modified state means no other cache has it.

If cache line is invalid, when the processor reads the data (`PrRd`), cache announces reading memory (`BusRd`) and cache line moves to the shared state. If cache line is shared, when the processor or other caches read the data, the cache line does nothing.

If cache line is invalid, when the processor writes the data (`PrWr`), cache announces exclusive reading of memory (`BusRdX`) and cache line moves to modified state.

If cache line is shared or modified, and another processor wants to write (`BusRdX`), the cache line becomes invalid. The data is now stale.

If cache line is modified and another processor wants to read, the cache line is flushed to memory and becomes shared. The other processor will be able to retrieve the latest data.

![](images/Pasted%20image%2020220301165306.png)

**GPUs and cache coherence**: GPUs tend not to implement cache coherence as it introduces some overhead. NVIDIA GPUs have incoherent L1 caches and a single unified L2 cache. L1 caches are write-through to L2 by default.

However, the latest Intel Integrated GPUs do implement.

**False sharing**: Two processors write to different addresses but the addresses map to the same cache line. Cache lines "ping-pong" between different processors, generating significant amounts of communication.

**False sharing demo**

Each thread runs this code.

```c
void* worker(void* arg) {
	volatile int* counter = (int*) arg;
	for (int i=0; i<MANY_ITERATIONS; i++)
		(*counter)++;
	return NULL;
}
```

Each thread updates a position of the counter array many times. The second code runs faster than the first one because cache lines do not bounce between different processors.

```c
/* Allocate per-thread variable */
int counter[MAX_THREADS];

/* Allocate per-thread variable with padding */
struct padded_t {
	int counter;
	char padding[CACHE_LINE_SIZE - sizeof(int)];
}
padded_t counter[MAX_THREADS];
```