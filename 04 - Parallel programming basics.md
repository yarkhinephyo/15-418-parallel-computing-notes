**Speedup**: Time for 1 processor / Time for P processors

**Steps to parallelize**: Break up a problem into many pieces of work -> Assign work to workers

![](images/Pasted%20image%2020220210235349.png)

**(1) Decomposition**: Create at least enough tasks to keep all execution units on a machine busy. This depends largely on the programmer. Compilers are not good at parallelizing code yet.

**Amdahl's law**: If `S` fraction of the program is inherently sequential, maximum speedup with parallel execution is `1/S`.

**(2) Assignment**: Many languages and runtimes take responsiblity for the assignment. 

For example, in the SPMD code, the programmer only decomposes the work into loops and the system manages the assignment of iterations to program instances. Another example is that ISPC can also assign tasks at the thread level.

**(3) Orchestration**: Communication between workers, adding synchronizations, scheduling tasks.

**(4) Mapping to hardware**: Mapping workers (threads) to execution units. The programmer does not have to concern with it.

OS maps thread to hardware execution context on CPU. Compiler maps program instances to vector instruction lanes. Hardware maps CUDA thread blocks to GPU cores.

**Mapping decisions**: Related threads on the same processor maximizes locality, data sharing, minimize costs of synchronization. Unrelated threads on the same processor (bandwidth-limited vs compute-limited) may utilize the machine more efficiently

**2D-grid based solver example**

Perform Gauss-Seidel (Avg of 5 cells) sweeps over grid till convergence.

![](images/Pasted%20image%2020220211161459.png)

```
/* Pseudocode */

const int n;
float* A; /* Assume grid of N+2 x N+2 elements */

void solve(float* A) {
	float diff, prev;
	bool done = false;
	
	while (!done) {
		diff = 0.f;
		for (int i=1; i<n; i++) {
			for (int j=1; j<n; j++) {
				prev = A[i,j];
				A[i,j] = 0.2f * (A[i,j] + A[i,j-1] + A[i,j+1] +
										A[i-1,j] + A[i+1,j]);
				diff += fabs(A[i,j] - prev);
			}
		}
		if (diff/(n*n) < TOLERANCE)
			done = true;
	}
}
```

**Decomposition for 2D-grid based solver**

Parallelism is difficult. Each row element depends on the element to left. Each row depends on the previous row.

![](images/Pasted%20image%2020220211162621.png)

There is independent work along the diagonals. Possible strategy is to break down each diagonal into tasks and update in parallel. However, there is not much parallelism at the beginning and end of the computation.

![](images/Pasted%20image%2020220211162852.png)

Algorithm can be changed if domain knowledge is available. New approach is to update red cells and black cells in parallel. New algorithm iterates to approximately the same solution.

![](images/Pasted%20image%2020220211163113.png)

**Assignment for 2D-grid based solver**

Blocked assignment may require less communication with other processors. For example, P2 has to find out two rows from P1 and P3. For interleaved assignment, P2 has to find out 6 rows from other processors.

![](images/Pasted%20image%2020220211163523.png)

**Data-parallel model**: Perform same operation on each element on the array. Programming in numPy is a good example.

**Stream programming model**: Elements in a stream are processed independently by "kernels" which are side-effect-free functions.

```
const int N = 1024;
stream<float> input(N);
stream<float> output(N);
stream<float> tmp(N);

foo(input, tmp);
bar(tmp, output);
```

**Gather operation**: Gather from buffer `mem_base` into `R1` according to indices specified by `R0`.

```
gather(R1, R0, mem_base)
``` 

![](images/Pasted%20image%2020220212105934.png)

**Shared address space pseudocode for 2D-grid based solver**

Pseudocode only showing the red cells. Blocked assignment is done for each thread. Barrier blocks until the expected number of threads arrive.

```c++
int n;
bool done = false;
float diff = 0.0;
LOCK mylock;
BARRIER myBarrier;

float* A = allocate(n+2, n+2);

/* Function solve() is run in all threads */

void solve(float* A) {
	int threadId = getThreadId();
	int myMin = 1 + (threadId * n / NUM_PROCESSORS);
	int myMax = myMin + (n / NUM_PROCESSORS);
	
	while (!done) {
		diff = 0.f;
		barrier(myBarrier, NUM_PROCESSORS);
		for (i = myMin to myMax) {
			for (j = red cells in the row) {
				float prev = A[i,j];
				A[i,j] = 0.2f * (A[i,j] + A[i,j-1] + A[i,j+1] +
										A[i-1,j] + A[i+1,j]);
				lock(myLock);
				diff += abs(A[i,j] - prev);
				unlock(mylock);
			}
		}
		barrier(myBarrier, NUM_PROCESSORS);
		if (diff/(n*n) < TOLERANCE)
			done = true;
		barrier(myBarrier, NUM_PROCESSORS); /* All threads same answer */
	}
}
```

However, every update to an (i, j) element requires acquiring the lock to modify `diff`. This is an expensive operation and only one thread can execute at a time. To improve performance, accumulate into a per-thread `diff` variable then complete the reduction globally at the end of the iteration.

```c++
/* The rest of code */

while (!done) {
	float myDiff = 0.f;
	diff = 0.f;
	barrier(myBarrier, NUM_PROCESSORS);
	for (i = myMin to myMax) {
		for (j = red cells in the row) {
			float prev = A[i,j];
			A[i,j] = 0.2f * (A[i,j] + A[i,j-1] + A[i,j+1] +
									A[i-1,j] + A[i+1,j]);
			myDiff += abs(A[i,j] - prev);
		}
	}
	lock(myLock); /* Lock is only acquired one time per thread */
	diff += myDiff;
	unlock(mylock);
	
	/* The rest of code */
}
```

**Barrier**: All computations by all threads before the barrier must complete before any computation in any thread after the barrier begins. Three barriers in the example code. 

First barrier - Ensure that each thread do not reset the `diff` while others are incrementing it.

Second barrier - Ensures all the threads have contributed to `diff` before running conditional check. 

Third barrier - Ensures the checking is done by all threads before going to the next iteration.

**Shared address space pseudocode with one barrier only**

Store a `diff` array with size of 3. One accumulator for the past iteration, one accumulator for the current iteration, one for preparing the next iteration.

```c++
int n;
bool done = false;
LOCK mylock;
BARRIER myBarrier;
float diff[3]; // Global array to store diffs

float* A = allocate(n+2, n+2);

/* Function solve() is run in all threads */

void solve(float* A) {
	float myDiff;  // Thread local variable
	int index = 0; // Thread local variable
	
	diff[0] = 0.0f;
	barrier(myBarrier, NUM_PROCESSORS);
	
	while (!done) {
		myDiff = 0.f;
		//
		// Perform accumulation into myDiff
		//
		lock(myLock);
		diff[index] += myDiff;
		unlock(mylock);
		
		diff[(index+1) % 3] = 0.0f; // Reset the next accumulator
		barrier(myBarrier, NUM_PROCESSORS);
		if (diff[index]/(n*n) < TOLERANCE)
			break;
		index = (index + 1) % 3;
	}
}
```