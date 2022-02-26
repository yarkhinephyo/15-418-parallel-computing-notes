**Work balancing**: Ideally all processors should be computing all the time. Amdahl's Law implies a small amount of load imbalance can bound maximum speed up.

**Static assignment**: Assignment of work is pre-determined. Applicable when the cost of work is predictable. Even if jobs are not equal, equal assignment can still be done to multiple processors.

**Semi-static assignment**: Periodically profiles the tasks and re-adjust assignments.

**Dynamic assignment**: Program determines assignment at runtime to ensure a well distributed load.

**Prime checker example**: Creating a work queue to distribute work evenly.

```c++
/* Assume the allocations are done by 1 thread */
int N = 1024;
int* x = new int[N];
bool* is_prime = new bool[N];

/* Initialize x here */

LOCK counter_lock;
int counter = 0; // Global variable

/* For each thread */
while (1) {
	int i;
	lock(counter_lock);    //
	i = counter++;         // Ensures atomicity
	unlock(counter_lock);  //
	if (i >= N)
		break;
	is_prime[i] = test_is_prime(x[i]);
}
```

The implementation above has good workload balance but high synchronization cost (locks). The white spaces represent the cost. If the cost of job function is low, too much time is wasted with synchronization overhead.

![](images/Pasted%20image%2020220224232801.png)

**Prime checker example with high granularity**

Use work queue but make each "task" larger by checking multiple numbers in the array before acquiring the lock again.

This reduces synchronization overhead but increases potential for uneven work distribution.

```c++
/* Assume the allocations are done by 1 thread */
const int GRANULARITY = 10;
int N = 1024;
int* x = new int[N];
bool* is_prime = new bool[N];

/* Initialize x here */

LOCK counter_lock;
int counter = 0; // Global variable

/* For each thread */
while (1) {
	int i;
	lock(counter_lock);     //
	i = counter;           // Ensures atomicity
	counter += GRANULARITY; //
	unlock(counter_lock);   //
	if (i >= N)
		break;
	int end = min(i + GRANULARITY, N);
	for (int j=i; j<end; j++)
		is_prime[j] = test_is_prime(x[j]);
}
```

![](images/Pasted%20image%2020220224233524.png)

**Smarter task scheduling**: Schedule long tasks first to prevent potential imbalances near the end of execution.

Knowing ahead about the duration of tasks is useful.

![](images/Pasted%20image%2020220224234121.png)

**Distributed work queues** Each worker has its own work queue. Avoid need for all works to synchronize on a single work queue. Worker threads pull and push data from their own queues. They steal work from another work queue when the local queue is empty.

![](images/Pasted%20image%2020220224234644.png)

**Common parallel programming patterns**: Data parallelism, Multi-threading, Fork-join pattern

**Cilk Plus**: C/C++ extension to support data/task parallelism. Find the pun.

```c++
cilk_spawn foo();
bar();
cilk_sync;
```

![](images/Pasted%20image%2020220225201558.png)

```c++
cilk_spawn foo();
cilk_spawn bar();
cilk_sync;
```

![](images/Pasted%20image%2020220225201612.png)

Notice the `cilk_spawn` abstraction does not specify how the spawned calls are "scheduled" to execute.

**Quicksort example with Cilk Plus**

```c++
void quick_sort(int* begin, int* end) {
	/* If problem is small, just do sequentially */
	if (begin >= end - PARALLEL_CUROFF)
		std::sort(begin, end);
	
	else {
		int* middle = partition(begin, end);
		cilk_spawn quick_sort(begin, middle);
		quick_sort(middle+1, last);
	}
}
```

![](images/Pasted%20image%2020220225202953.png)

**Problems with spawning new threads every time**: Creating a thread is expensive. Context switching adds overhead. Larger working set than necessary reduces cache locality.

**Cilk Plus implementation**: A pool of worker threads with separate queues that "steal" from one another for tasks.

In "child stealing", the caller thread records child for later execution (Add to queue). In "continuation stealing", the caller thread records continuation for later execution. Think of them as BFS and DFS respectively.

If the workload is more balanced, less stealing is required. Stealing is more expensive than spawns due to requiring locks and barriers.

![](images/Pasted%20image%2020220225205047.png)

**Quicksort example scheduling**

Thread 0 puts continuations onto the queue. The other threads steal from top of the queue.

This creates two advantages. Thread 0 has cache locality after returning from "0-25" section. The other threads can ensure that longer tasks are worked on first.

![](images/Pasted%20image%2020220225210838.png)

**For loop implementation in Cilk Plus**

Let's consider a `for` loop generating parallel tasks. If child is ran first, the continuation bounces back and forth between threads so the parallelism is revealed sequentially. If continuation is ran first, there is only one thread generating parallel work. Both have problems.

To solve this, `cilk_for` takes a divide-and-conquer approach to generate tasks more quickly for processors.

```c++
// Normal iterative for
for (int i=0; i<N; i++) {
	cilk_spawn foo(i);
}


// Divide-and-conquer
// Implementation of cilk_for loop
void recursive_for(int start, int end) {
	while (start <= end - GRANULARITY) {
		int mid = (end - start) / 2;
		cilk_spawn recursive_for(start, mid);
		start = mid;
	}
	
	for (int i=start; i<end; i++)
		foo(i);
}
recursive_for(0, N);
```

![](images/Pasted%20image%2020220225211629.png)

**Sync implementation in Cilk Plus**

Sync does not do anything if no work is stolen by other threads.

![](images/Pasted%20image%2020220225215416.png)

When thread 1 steals from thread 0, a descriptor for block A is created, which tracks the number of spawns for the block and how many have been completed. This creates overhead but if the work pieces are large, it should be small.

![](images/Pasted%20image%2020220225215321.png)

![](images/Pasted%20image%2020220225215015.png)