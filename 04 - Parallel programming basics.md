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

![](../Pasted%20image%2020220211162621.png)

There is independent work along the diagonals. Possible strategy is to break down each diagonal into tasks and update in parallel. However, there is not much parallelism at the beginning and end of the computation.

![](../Pasted%20image%2020220211162852.png)

Algorithm can be changed if domain knowledge is available. New approach is to update red cells and black cells in parallel. New algorithm iterates to approximately the same solution.

![](../Pasted%20image%2020220211163113.png)

**Assignment for 2D-grid based solver**

Blocked assignment may require less communication with other processors. For example, P2 has to find out two rows from P1 and P3. For interleaved assignment, P2 has to find out 6 rows from other processors.

![](../Pasted%20image%2020220211163523.png)