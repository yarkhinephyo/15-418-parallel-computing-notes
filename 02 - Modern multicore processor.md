**Example program**: Compute sin(x) with Taylor expansion for an array of N floating point numbers.

```c
/* sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ... */
void sinx(int N, int terms, float* x, float* result) {
	int i;
	for (i=0; i<N; i++) {
		float value = x[i];
		float numer = x[i] * x[i] * x[i];
		int denom = 6 // 3!;
		int sign = -1;
		
		int j;
		for (j=1; i<=terms; j++) {
			value += sign * numer / denom;
			numer *= x[i] * x[i];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		result[i] = value;
	}
}
```

ARM instructions for inside of the loop.

![](images/Pasted%20image%2020220206220040.png)

**Simple processor**

![](images/Pasted%20image%2020220206220506.png)

**Super scalar processor**: Decode and execute more than one instruction per clock cycle. Parallelism is discovered by the hardware and not by the programmer.

Existing binary executable programs have varying degrees of intrinsic parallelism. In some cases instructions are not dependent on each other and can be executed simultaneously. In other cases they are inter-dependent: one instruction impacts either resources or results of the other.

![](images/Pasted%20image%2020220206220612.png)

**Processor pre multi-core era**: Increase in chip transistors were used to help a single instruction stream run fast. However, the bottlenecks in data dependencies (such as the example) prevented high performance gains.

![](images/Pasted%20image%2020220206221234.png)

**(Idea #1) Processor multi-core era**: Use the increase in transistors to add more cores to the processor instead. If each core is 25% slower than a "fancy" core, it is still 2 x 0.75 = 1.5 times faster.

**Example program with pthreads**: Use one more thread to split the work by half.

```c
typedef struct {
	int N;
	int terms;
	float* x;
	float* result;
} my_args;

/* sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ... */
void parallel_sinx(int N, int terms, float* x, float* result) {
	pthread_t thread_id;
	my_args args;
	args.N = N / 2;
	args.terms = terms;
	args.x = x;
	args.result = result;
	
	pthread_create(&thread_id, NULL, my_thread_start, &args);
	sinx(N - args.N, terms, x + args.N, result + args.N); // Do work
	pthread_join(thread_id, NULL);
}

void my_thread_start(void* thread_arg) {
	my_args* thread_args = (my_args*) thread_arg;
	sinx(args->N, args->terms, args->x, args->result);    // Do work
}
```

**(Idea #2) SIMD processing**: Add only arithmetic logic units (ALU) to increase compute capability. They are cheaper than the instruction decoder. Same instruction is broadcasted to all ALUs so that execution is parallel.

![](images/Pasted%20image%2020220206222837.png)

**Example program with AVX**: Process eight array elements simultaneously with vector instructions on 256-bit vector registers.

```c
#include <immintrin.h>

void sinx(int N, int terms, float* x, float* result) {
	float three_fact = 6;  // 3!
	int;
	for (i=0; i<N; i+=8) { // 256-bit vector -> 8 integers
		/* Moves singe-precision floats from aligned memory to vector */
		__m256 origx = _mm256_load_ps(&x[i]);
		__m256 value = origx;
		__m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
		__m256 denom = _mm256_broadcast_ss(&three_fact);
		int sign = -1;
		
		int j;
		for (j=1; i<=terms; j++) {
			__m256 tmp = _mm256_div_ps(
				_mm256_mul_ps(mm256_broadcast_ss(sign), numer), denom);
			value = _mm256_add_ps(value, tmp);
			
			numer = _mm256_mul_ps(
				numer, _mm256_mul_ps(origx, origx));
			denom = _mm256_mul_ps(
				denom, _mm256_broadcast_ss((2*j+2) * (2*j+3));
			sign *= -1;
		}
		_mm256_store_ps(&sin[i], value);
	}
}
```

Advanced vector extensions (AVX) instructions.

![](images/Pasted%20image%2020220206225625.png)

16 SIMD cores with 8 ALU each can run 128 elements in parallel.

![](images/Pasted%20image%2020220206225733.png)

**Conditional execution**: All ALUs share the same instruction stream. If there is conditional branching for different items in the array, every instruction is executed for all items but the relevant ALUs are masked on/off.

Worst case performance will be 1/8 peak performance.

![](images/Pasted%20image%2020220206230302.png)

**Explicit SIMD**: SIMD parallelization is performed at compile time. The programmer requests with intrinsics or a parallel language semantics. In the binary instructions, commands such as `vstoreps`, `vmulps` can be found.
**Implicit SIMD**: Compiler generates a scalar binary. Hardware is responsible for simultaneously executing the same instruction from on different data with SIMD ALUs.

**Memory latency**: Time taken for a memory request from a processor to be serviced by the memory system. (100 cycles)
**Memory bandwidth**: The rate at which the memory system provides data to a processor. (20 GB/s)

**Cache**: Reduces latencies.

**(Idea #3) Hardware multi-threading**: Hides latencies. Processor has useful work as it awaits for some completion of memory requests.

Execution context in the picture is considered as a hardware thread. The processor below can maintain information about four threads at once.

When thread 1 issues load instruction for 8 elements, there will be a stall while waiting for more than 100 clock cycles. The processor can then switch to thread 2 instead.

The efficiency is 100% because the resources are always utilized.

![](images/Pasted%20image%2020220207145527.png)

![](images/Pasted%20image%2020220207145901.png)

**Logical cores**: Number of physical cores times the number of threads that can run on each core.

**Throughput computing trade-off**: Potentially increase time to complete work by any one thread so that overall system throughput can be increased with multiple threads.

**Multi-threading relies on memory bandwidth**: May go to memory often but can hide the latency. With more threads, the working set is larger so there is less cache space per thread.

**Example multi-core chip**

```
16 cores -> 16 simultaneous instruction streams
64 logical cores -> 64 concurrent instruction streams
8 SIMD ALU/core -> 512 independent pieces of work

# If not, the processor will not be 100% efficient
```

![](images/Pasted%20image%2020220207161733.png)

**NVIDIA GTX 1080 core (SM for Streaming Multiprocessor)**: Warps are threads issuing 32-wide vector instructions. One core can support 64 warps "concurrently". Different instructions from up to 4 warps can be executed "simultaneously". In other words, every clock cycle, 4 of the 64 warps are executed. Over 64 x 32 = 2048 elements can be processed concurrently by a core.

![](images/Pasted%20image%2020220207162514.png)

**NVIDIA GPX 1080**: 20 SM cores. 20960 pieces of data to be processed concurrently. Unlike CPU, GPU assumes that there is a large memory bandwidth. Caches are much smaller so there is more memory accesses. Having many threads hide latencies.

![](images/Pasted%20image%2020220207163035.png)

**Though experiment on CPU vs GPU**: Element-wise multiplication of two very large vectors.

```
Load A[i] -> Load B[i] -> Compute A[i] * B[i] -> Store to C[i]
```

Three memory operations (12 bytes) for every MUL. Considering that NVIDIA GTX 1080 GPU can do 2560 MULs per clock (@1.6 GHz), ~45 TB/sec of memory bandwidth is required to keep functional units busy. With only 320 GB/sec memory bandwidth, there is less than 1% efficiency.

On a laptop CPU, the efficiency may be higher (~3%).

GPU is faster 4.2 times but not because of parallelism, only because of better memory bandwidth!!

**Bandwidth as a critical resource**: Overcoming bandwidth limits are a common challenge for throughput-optimized systems.

Organize computation to fetch data from memory less often. For example, reuse data previously loaded by the thread or share data with across threads.

**Modern multicore processor**: 4x cores, 2-way multi-threading per core, 2x instructions per clock per core (1x instruction being 8-wide SIMD).

![](images/Pasted%20image%2020220207170914.png)

**How to assign two threads onto CPU?**: If they share data, put them on same physical core to use the resources more efficiently and avoid stalls. If stall never occurs, no need to put multiple threads on the same physical core.