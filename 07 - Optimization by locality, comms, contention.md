**Message passing model abstraction**: Each thread operates within its own private address space. Threads communicate by sending and receiving data.

![](images/Pasted%20image%2020220227133031.png)

If each thread operates in its own address space, data replication is now required to correctly execute the program.

![](images/Pasted%20image%2020220227133456.png)

```c
/* Note that this code involves DEADLOCK */
int N;
int tid = get_thread_id();
int rows_per_thread = N / get_num_threads();

/* Local copy of array A*/
float* localA = allocate(rows_per_thread+2, N+2);

/* Assume MSG_ID_ROW, MSG_ID_DONE, MSG_ID_DIFF are constants */
void solve() {
	bool done = false;
	while (!done) {
		float my_diff = 0.0f;
		
		/* If not first row, send the first row to processor above */
		if (tid != 0)
			send(&localA[1,0], sizeof(float)*(N+2), tid-1, MSG_ID_ROW);
		if (tid != get_num_threads()-1)
			send(&localA[rows_per_thread,0],
				sizeof(float)*(N+2), tid+1,
				MSG_ID_ROW);

		/* If not first row, receive a row from processor above */
		if (tid != 0)
			recv(&localA[0,0], sizeof(float)*(N+2), tid-1, MSG_ID_ROW);
		if (tid != get_num_threads()-1)
			recv(&localA[rows_per_thread+1,0],
				sizeof(float)*(N+2), tid+1,
				MSG_ID_ROW);
		
		for (int i=; i<rows_per_thread+1; i++) {
			for (int j=1; j<n+1; j++) {
				float prev = localA[i,j];
				localA[i,j] = 0.2 * (localA[i-1,j] + localA[i,j] +
									localA[i+1,j] + localA[i,j-1] +
									localA[i,j+1]);
				my_diff += fabs(localA[i,j] - prev);
			}
		}
		
		/* Other threads send my_diff to thread 0 */
		/* Other threads receive done state from thread 0 */
		if (tid != 0) {
			send(&my_diff, sizeof(float), 0, MSG_ID_DIFF);
			recv(&done, sizeof(bool), 0, MSG_ID_DONE);
		} else {
			float remote_diff;
			for (int i=1; i<get_num_threads()-1; i++) {
				recv(&remote_diff, sizeof(float), i, MSG_ID_DIFF);
				my_diff += remote_diff;
			}
			if (my_diff/(N*N) < TOLERENCE)
				done = true;
			for (int i=1; i<get_num_threads()-1; i++)
				send(&done, sizeof(bool), i, MSG_ID_DONE);
		}
	}
}
```

**Synchronous send**: Call returns when sender receives acknowledgement from receiver.
**Sychronous recv**: Call returns when the message is copied and the acknowledgement is sent back.

```c
/* Solution to DEADLOCK */
if (tid % 2 == 0) {
	sendDown(); recvDown();
	sendUp(); recvUp();
} else {
	recvUp(); sendUp();
	recvDown(); sendDown();
}
```

**Asynchronous send**: Returns immediately. The buffer provided to be sent cannot be modified.
**Asychronous recv**: Returns immediately. Use `checksend()` and `checkrecv()` to determine actual status.

**Instruction pipeline**: Break execution of each instruction into smaller steps. Different hardware run in parallel to complete each of the step. Latency does not change but throughput increases.

![](images/Pasted%20image%2020220227142720.png)

**Throughput of pipelined communication**: 1 / Speed-of-slowest-component.

![](images/Pasted%20image%2020220227143646.png)

**Cache review**

Consider a small cache with 16-byte cache line and capacity of 32 bytes. Below shows the cache state as each 4-byte element is being accessed. (One dot is 4 bytes)

![](images/Pasted%20image%2020220227150001.png)

**Arithmetric intensity**: Elements computed over elements communicated.

![](images/Pasted%20image%2020220227151111.png)

**Grid solver example considering cache**

Assume cache line is 4 grid elements and cache capacity is 24 grid elements.

Blue color shows elements in cache at the end of the first row. The program loads 3 cache lines for every 4 elements.

![](images/Pasted%20image%2020220227150201.png)

Temporal locality can be improving by changing grid traversal order.

![](images/Pasted%20image%2020220227153821.png)

**Improving temporal locality by fusing loops**

Each arithmetic operation requires two loads, one store. Arithmetic intensity of 1/3.

```c
void add(int n, float* A, float* B, float* C) {
	for (int i=0; i<n; i++)
		C[i] = A[i] + B[i];
}
void mul(int n, float* A, float* B, float* C) {
	for (int i=0; i<n; i++)
		C[i] = A[i] * B[i];
}

/* E = D + ((A+B) * C) */
add(n, A, B, tmp1);
mul(n, tmp1, C, tmp2);
add(n, tmp2, D, E);
```

There are three arithmetic operations, four loads and one store. Arithmetic intensity of 3/5. If the program was bandwidth-bound, this will see a huge improvement.

```c
void fused(int n, float* A, float* B, float* C, float* D, float* E) {
	for (int i=0; i<n; i++)
		E[i] = D[i] + (A[i] + B[i]) * C[i];
}

/* E = D + ((A+B) * C) */
fused(n, A, B, C, D, E);
```

**Contention**: Many requests to a resource are made within a small window of time.

**CUDA memory contention**

For NVIDIA GTX 480 shared memory, the storage is physically partitioned into 32 SRAM banks. Address X is stored in bank (X % 32). If all the CUDA threads only share a few banks, the threads will not truly run in parallel.

```c
__shared__ float A[512];
int index = threadIdx.x;

float x2 = A[index];       // Single cycle
float x3 = A[3 * index];   // Single cycle
float x4 = A[16 * index]; // 16 cycles
```

![](images/Pasted%20image%2020220227162812.png)

**Grid-of-particles example**: Place 1M point particles in a 16-cell uniform grid. Build 2D array of lists.

![](images/Pasted%20image%2020220227172703.png)

One solution is to parallelize over particles. For each particle, compute the cell and atomatically updates list. However, there will be massive contention for the 2D array.

```
list cell_lists[16];
lock cell_list_lock;

for each particle p
	c = compute cell containing p
	lock(cell_list_lock)
	append p to cell_list[c]
	unlock(cell_list_lock)
```

Another solution is to use finer granularity locks. If there are locks for each list item, there is 16x less contention.

```
list cell_lists[16];
lock cell_list_lock[16];

for each particle p
	c = compute cell containing p
	lock(cell_list_lock[c])
	append p to cell_list[c]
	unlock(cell_list_lock[c])
```

Another solution is to generate grids for each thread-block. If there are N thread-blocks, contention reduces by a factor of N and cost of synchronization is lower because it is performed on block-local variables. However, it requires extra memory footprint.

Final solution is data-parallel approach. Compute the grid cell for each particle index. -> Sort the grid cells with a parallel sorting algorithm. -> Find start and end index of each grid cell in parallel.

![](images/Pasted%20image%2020220227172519.png)

![](images/Pasted%20image%2020220227172330.png)


```c
/* Run for each element of particle_index array */
cell = grid_index[index]
if (index == 0)
	cell_starts[cell] = index;
else if (cell != grid_index[index-1]) {
	cell_starts[cell] = index;
	cell_ends[grid_index[index-1]] = index;
}
if (index == numParticles - 1)
	cell_ends[cell] = index + 1;
```