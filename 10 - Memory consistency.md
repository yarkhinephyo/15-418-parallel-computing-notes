**Memory coherence**: Making memory behave as if caches do not exist. Put all operations involving `X` on a timeline such that the observations of all processors are consistent with the timeline.

**Memory consistency**: Defines the allowed behaviour of loads and stores to different addresses in a parallel system. The allowed behaviour should be specified whether or not caches are present.

**Memory operation ordering**: Program defines a sequenec of loads and stores.

```
W -> R   # Write to X must commit before read from Y
R -> R
R -> W
W -> W
```

If X-write must commit before Y-read, the X-write's results must be visible before the read occurs.

**Sequentially consistent**: Memory system maintains all four memory operation orderings. All processors issue loads and stores in program order.

For example, with simultaneous threads on a two processor system, "hello" or "world" can be printed by not both.

```
# Thread 1 (Processor 1)
A = 1;
if (B == 0)
	print "Hello";
	
# Thread 2 (Processor 2)
B = 1;
if (A == 0)
	print "World";
```

**Relaxed memory consistent**: Allows certain orderings to be violated. Hiding memory latency is possible by overlapping memory access operations when they are independent.

**Relaxing W->R ordering**: Allows the processor to hide latency of writes when later read is independent. When store is issued, processor "buffers" the store operation in the write buffer and immediately executes subsequent loads.

The write buffer holds writes that have been issued by the processor and is different from cache's write-back buffer which holds dirty cache lines.

![](images/Pasted%20image%2020220319192822.png)

**Total store ordering (TSO)** Processor can start reading `B` before its earlier writes to `A` is seen by all processors. Until the write is seen by all processors, the other processors will not read the new value of `A`.

**Processor consistency (PC)**: Any processor can read the new value of `A` before the write is seen by all processors.

**TSO vs PC examples**: Assume A and B are initialized to 0.

Results of execution match that of sequential consistency since there is no W->R.

```
# Program 1

# Thread 1 (Processor 1)
A = 1;
flag = 1;
	
# Thread 2 (Processor 2)
while (flag == 0);
print A;
```

Results of execution match that of sequential consistency since there is no W->R.

```
# Program 2

# Thread 1 (Processor 1)
A = 1;
B = 1;
	
# Thread 2 (Processor 2)
print B;
print A;
```

Read can move up before the write. In TSO, once thread 2 knows that A = 1, thread 3 also knows that A = 1. So results match that of sequential consistency.

In PC, once thread 2 knows that A = 1, thread 3 may not know the same. So thread 3 may print A as 0. Results do not match that of sequential consistency.

```
# Program 3

# Thread 1 (Processor 1)
A = 1;
	
# Thread 2 (Processor 2)
while (A == 0);
B = 1;

# Thread 3 (Processor 3)
while (B == 0);
A = 1;
```

Prints can occur before variable assignments for both threads. In sequential consistency, at least something will be printed. With TSO and PC, there is a scenario where nothing will be printed. Both results do not match that of sequential consistency.

```
# Program 4

# Thread 1 (Processor 1)
A = 1;
print B;
	
# Thread 2 (Processor 2)
B = 1;
print A;
```

**Why reorder other operations**

(W->W) Processors may reorder write operations to optimize cache hits. (R->W, R->R) Processor may reorder independent instructions in an instruction stream. If there is only a single instruction stream, these are valid optimizations.

**Release consistency (RC)**: Processors support special synchronization operations. For example, `fence` keyword ensures memory accesses before the instruction is completed before moving forward.

**Intel x86-64**: Total store ordering.
**ARM processors**: Very relaxed consistency model.

**Acquire**: Prevents reordering of `X` with any load/store AFTER `X` in program order.

**Release**: Prevents reordering of `X` with any load/store BEFORE `X` in program order.

**C++ 11 atomic \<T\>** Provides memory ordering semantics for operations before and after atomic operations.

```c++
// Thread 1

atomic<int> flag;
int foo;
foo = 1;
flag.store(1, std::memory_order_release);

// Thread 2

// Other code
while (flag.load(std::memory_order_acquire) == 0);
// Use foo here
```