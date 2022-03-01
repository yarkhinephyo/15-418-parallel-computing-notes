**CPU to GPU**: Similar concepts as CPU. More cores, wider SIMDs, more hardware threads.

**Graphics rendering pipeline**: Programmers provides mini-programs (shaders) that define pipeline logic at certain stages. Pipeline executes shader function for all elements of input stream.

![](images/Pasted%20image%2020220226130103.png)

**Hack!**: Early GPU-based scientific computation.

Set graphics pipeline output image to be output array size (512x512). Custom fragment shader function is mapped over the element collection.

**Brook stream programming language**: Early compiler that translate a generic stream program into OpenGL commands and a set of OpenGL shader programs.

**Running code on CPU vs GPU**

For CPU, OS loads binary into memory. -> OS selects CPU execution context. -> OS interrupts processor and prepares execution context. -> Processor executes instructions within the execution context.

![](images/Pasted%20image%2020220226131449.png)

For 2007 NVIDIA Tesla GPU, the application allocates buffers in GPU memory -> Application provides GPU a single kernel program binary. -> Application tells GPU to run the kernel in SPMD fashion.

**CUDA**: C-like language to express SPMD programs that run on GPUs. OpenCL is the open standards version of CUDA.

**CUDA thread**: Similar logical abstraction as `pthread` but the implementation is very different.

**CUDA programs**: SPMD programs. One program instance is one "CUDA thread". CUDA threads are organized into "thread blocks". Thread IDs can become 2-dimensional or 3-dimensional.

![](images/Pasted%20image%2020220226132342.png)

**Example of matrix addition**: 12 threads per block, 6 blocks.

CPU application code.

```c
const int Nx = 12;
const int Ny = 6;

dim3 threadsPerBlock(4, 3, 1);
dim3 numBlocks(Nx / threadsPerBlock.x, Ny / threadsPerBlock.y, 1);

/* Nx x Ny float arrays */
float *A, *B, *C;

/* This call will execute 12 x 6 CUDA threads */
matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

CUDA kernel definition.

```c
__global__ void matrixAdd(float A[Ny][Nx],
						  float B[Ny][Nx],
						  float C[Ny][Nx])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}
```

Kernel definition requires manually guarding against out of bounds array access.

```c
const int Nx = 12;  // Not a multiple of threadsPerBlock.x
const int Ny = 6;   // Not a multiple of threadsPerBlock.x

/* Kernel definition */
__global__ void matrixAdd(float A[Ny][Nx],
						  float B[Ny][Nx],
						  float C[Ny][Nx])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < Nx && j < Ny)
		C[i][j] = A[i][j] + B[i][j];
}
```

**CUDA Address spaces**: Distinct address spaces between CPU and GPU. Memory manipulation can be done with `cudaMalloc/cudaFree` and `cudaMemcpy`. Pointers allocated with `cudaMalloc` cannot be accessed on CPU.

![](images/Pasted%20image%2020220226140002.png)

```c
/* Buffer in host memory */
float *hostA = new float[N];

/* Initialize host address space buffer */
for (int i=0; i<N; i++)
	hostA[i] = (float)i;

int bytes = sizeof(float) * N;
float *deviceA;
cudaMalloc(&deviceA, bytes);

/* Invalid to access deviceA[i] */
```

**CUDA device memory model**

Three types of memory. "GPU memory" where every thread can read/write. "Per-thread-block memory" where all the threads inside can read/write. "Per-thread memory" for local variables.

**1D convolution in CUDA (Version 1)**

![](images/Pasted%20image%2020220226164704.png)

CUDA kernel definition.

```c
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
	index = blockIdx.x * blockDim.x + threadId.x;
	float result = 0.0f;              // Thread local
	for (int i=0; i<3; i++)
		result += input[index + i];
	output[index] = result / 3.f;     // Global variable
}
```

CPU application code.

```c
int N = 2014 * 1024;
cudaMalloc(&devInput, sizeof(float) * (N+2));
cudaMalloc(&devOutput, sizeof(float) * N);

/* Initialize devInput here */

convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput);
```

**1D convolution in CUDA (Version 2)**: Load the array elements to the per-thread-block memory region first. Loading from this region is much faster than the GPU global memory. By modifying the program, global memory is accessed 3x less.

```c
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
	__shared__ float support[THREADS_PER_BLK + 2]; // Per thread blk
	index = blockIdx.x * blockDim.x + threadId.x;  // Thread local
	
	support[threadIdx.x] = input[index];
	if (threadIdx.x < 2) 
		support[THREADS_PER_BLK + threadIdx.x] = 
			input[index + THREADS_PER_BLK];
	
	__syncthreads();
	
	float result = 0.0f;              // Thread local
	for (int i=0; i<3; i++)
		result += support[threadIdx.x + i];
	output[index] = result / 3.f;     // Global variable
}
```

Required resources include 128 threads per block, "B" bytes of local data per thread. 130 floats (520 bytes) of memory in thread-block.

**CUDA compilation**: CUDA threads are logical. The same number of hardware threads are not ran in the GPU.

**NVIDIA GTX 1080 (2016)**: Every core has 96 KB of shared memory. 64 hardware threads. 32-wide SIMD instructions per thread.

Interleaved "multi-threading" where 4 warp contexts can be selected out of 64 (Hyperthreading). Simultaneous execution of 4 warps at a time. For any warp, up to two runnable instructions can be done (Instruction-level parallelism).

![](images/Pasted%20image%2020220226172606.png)

**1D convolution thread-block assignment**

On NVIDIA GPUs, groups of 32 CUDA threads share one instruction stream. These groups are called "warps". A `convolve` thread-block is executed by 4 warps (32 x 4 = 128 CUDA threads).

Host send CUDA device a command. -> CUDA device schedules the thread-blocks. -> Only 4 thread-blocks can be fit at one time due to per-thread-block memory limitation.

![](images/Pasted%20image%2020220226174636.png)

![](images/Pasted%20image%2020220226174910.png)
