**SPMD**: Single program multiple data. High level programming languages to program parallelism. The compiled low level machine code may contain SIMD instructions.

**ISPC**: Intel SPMD Program Compiler

**Example program with ISPC**

C++ code calls ISPC code.

```c
#include "sinx_ispc.h"

int N = 1024;
int terms = 5;
float* x = new float[N];
float* result = new float[N];

/* Initialize x here */

/* Execute ISPC code */
sinx(N, terms, x, result);
```

Syntax `programCount` is the number of simultaneously executing instances in the gang. Syntax `programIndex` is the current instance in the gang. Syntax `uniform` is a type modifier where each instance have "same" value for this variable (For optimization).

```
export void sinx(
	uniform int N,
	uniform int terms,
	uniform float* x,
	uniform float* result
) {
	/* Assumes N % programCount = 0 */
	for (uniform int i=0; i<N; i+= programCount) {
		int idx = i + programIndex;
		float value = x[idx];
		float number = x[idx] * x[idx] * x[idx];
		uniform int denom = 6; // 3!
		uniform int sign = - 1;
		
		for (uniform int j=1; j<=terms; j++) {
			value += sign * numer / denom;
			numer *= x[idx] * x[idx];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		result[idx] = value;
	}
}
```

Even though the programmer can program as if there are `programCount` instances of the code running concurrently, the ISPC compiler generates `.o` binary with SIMD instructions.

**Interleaved assignment of program instances**

![](images/Pasted%20image%2020220210225831.png)

**Blocked assignment of program instances**

```
export void sinx(
	uniform int N,
	uniform int terms,
	uniform float* x,
	uniform float* result
) {
	/* Assumes N % programCount = 0 */
	uniform int count = N / programCount;
	int start = programIndex * count;
	for (uniform int i=0; i<count; i++) {
		int idx = i + start;
		float value = x[idx];
		float number = x[idx] * x[idx] * x[idx];
		uniform int denom = 6; // 3!
		uniform int sign = - 1;
		
		for (uniform int j=1; j<=terms; j++) {
			value += sign * numer / denom;
			numer *= x[idx] * x[idx];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		result[idx] = value;
	}
}
```

![](images/Pasted%20image%2020220210230329.png)

**Interleaved assignment is better**: Parallel instructions are done on contiguous memory.

![](images/Pasted%20image%2020220210230657.png)

**A higher level abstraction with foreach**: Each iteration is executed exactly once for a gang of ISPC instances. ISPC determines which instance executes which loop iteration (Interleaved).

```
export void sinx(
	uniform int N,
	uniform int terms,
	uniform float* x,
	uniform float* result
) {
	foreach (i = 0 ... N) {
		float value = x[i];
		float number = x[i] * x[i] * x[i];
		uniform int denom = 6; // 3!
		uniform int sign = - 1;
		
		for (uniform int j=1; j<=terms; j++) {
			value += sign * numer / denom;
			numer *= x[i] * x[i];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		result[i] = value;
	}
}
```

**ISPC reduction**: Each program instance sums into a per-instance variable named `partial`, then `reduce_add` adds all the `partial` floats into one uniform variable.

```
export uniform float sumall2(
	uniform int N,
	uniform float* x
) {
	uniform float sum;
	float partial = 0.0f;
	foreach (i = 0 ... N) {
		partial += x[i];
	}
	/* From ISPC math library */
	sum = reduce_add(partial);
	return sum;
}
```

The equivalent C code is shown below.

```c
float sumall2(int N, float* x) {
	float tmp[8];
	__mm256 partial = _mm256_broadcast_ss(0.0f);
	
	for (int i=0; i<N; i+=8)
		partial += _mm256_add_ps(partial, _mm256_load_ps(&x[i]));
	
	_mm256_store_ps(tmp, partial);
	
	float sum = 0.0f;
	for (int i=0; i<8; i++)
		sum += tmp[i];
	
	return sum;
}
```