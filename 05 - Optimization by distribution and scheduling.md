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
	i = counter;          // Ensures atomicity
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