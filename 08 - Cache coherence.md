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

**MESI cache coherence**: If another cache has the data, the cache line is shared. If not, the cache line is exclusive.

![](images/Pasted%20image%2020220319182056.png)

It is common in programs to read and write data together. If a program wants to write immediately after reading, the cache does not waste resources to notify other caches.

**GPUs and cache coherence**: GPUs tend not to implement cache coherence as it introduces some overhead. NVIDIA GPUs have incoherent L1 caches and a single unified L2 cache. L1 caches are write-through to L2 by default.

However, the latest Intel Integrated GPUs do implement.

**False sharing**: Two processors write to different addresses but the addresses map to the same cache line. Cache lines "ping-pong" between different processors as they transit between different MSI states, generating significant amounts of communication.

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
/* 8 threads on 4-core takes 7.2 sec */
int counter[MAX_THREADS];

/* Allocate per-thread variable with padding */
/* 8 threads on 4-core takes 3.06 sec */
struct padded_t {
	int counter;
	char padding[CACHE_LINE_SIZE - sizeof(int)];
}
padded_t counter[MAX_THREADS];
```

**Snooping cache limitation**

Snooping cache coherence protocols relied on broadcasting coherence information to all processors over the chip interconnect. Every time a cache miss occurs, the triggering cache communicates with all other caches.

![](images/Pasted%20image%2020220317235250.png)

This may not be scalable when machines get bigger (supercomputers). Every processor may have memory next to it to reduce latency when there is locality. This feature does not not any good if cache coherence requires communicating with all other processors.

![](images/Pasted%20image%2020220317235505.png)

**Directory-based cache coherence**

Directory-based cache coherence avoids expensive broadcast by storng information about the status of a line in one place. For every cache line of memory, there is a directory entry which contains the processors that have the particular line in their caches.

![](images/Pasted%20image%2020220318000919.png)

**Home node**: Node with memory for the particular line. Node 0 is the home node for yellow line.

**Read miss**: If processor 0 wants blue line, it sends a request to processor 1 for reading the data. Processor 1 updates the directory and sends the data to processor 0.

If the dirty bit is off, the processor 1 provides data from its memory.

![](images/Pasted%20image%2020220318001226.png)

If the dirty bit is on, the processor 1 tells the requesting node where to find the updated data (Let's say processor 2). Processor 1 sends a request to processor 2. Processor 2 changes state in cache to SHARED and provides the data to processor 1. Processor 2 provides the data and directory revision to processor 1 (No more dirty bit).
 
**Write miss**: If processor 0 wants to write blue line, it sends a request to processor 1 for writing the data. 

![](images/Pasted%20image%2020220318003139.png)

Processor 1 provides the data and the sharing processors to processor 0.

![](images/Pasted%20image%2020220318003238.png)

Processor 1 sends an invalidation request to the sharing processors so that the blue lines are cleared. Processor 1 waits for acknowledgements from the sharing processors before writing.

![](images/Pasted%20image%2020220318003747.png)

In practice, it is rare to have many processors sharing the same cache line.

**Directory-based cache limitation**

There is a lot of storage overhead for the cache line directories. Storage is proportional to `num_processors x num_lines_in_memory`.

If each cache line is 64 byes (512 bits) and there are 64 nodes, 64 bits will be needed for a directory entry. Thus 12% overhead.

**Limited pointer scheme**: In practice, data is only expected to be in a few caches at once. Hence, can store as a list of the nodes instead. We do not need to track a list of all processors for every cache line.

**Directory-based cache in Intel Core i7**

L3 cache stores a directory of all lines. This contains lists of L2 caches containing each line. Coherence messages are only sent to the L2's that contain the line, instead of broadcasting to all L2's.

![](images/Pasted%20image%2020220319181016.png)