**Wrong lock implementation**: Since load-test-store is not atomic, two processors can acquire the lock at the same time.

```
lock:    ld   R0, mem[addr]
         cmp  R0, #0
	     bnz  lock           // If nonzero, jump to "lock"
	     st   mem[addr], #1

unlock:  st   mem[addr], #0
```

**Test-and-set lock implementation**

```
ts   R0, mem[addr]  // Load mem[addr] into R0
                    // If R0 is 0, set 1 to mem[addr]
					// Returns R0
```

If test-and-set instruction fails, do it again. This is equivalent to acquiring lock.

```
lock:    ts   R0, mem[addr]
	     bnz  lock
unlock:  st   mem[addr], #0
```

**Test-and-set with coherence traffic**

Processor 1 runs `ts` instruction and invalidates cache lines in other caches. The other processors will loop the `ts` instruction until they acquire the lock.

As the `ts` instruction is a write and the cache has been invalidated, the next processor would broadcast `BusRdX` to bring address into cache. Processor 1 then would have to invalidate the cache line. The `ts` instruction fails because the lock has already been acquired by processor 1.

The invalidation process repeats when another processor broadcasts `BusRdX` to start running the `ts` instruction.

![](images/Pasted%20image%2020220401232845.png)

Even though processor 1 took the lock, it does not have the cache line anymore. It would have to broadcast `BusRdX` again to update the cache line for releasing the lock.

![](images/Pasted%20image%2020220401233253.png)

**Test-and-test-and-set lock implementation**

Additional test before the test-and-set instruction. This is only a read instruction, so the other processors will only be loading into a SHARED state. There is no interconnect traffic when the lock is taken. There will only be traffic again after the lock is released.

Slightly higher latency than test-and-set in uncontended case but generates less interconnect traffic. However, all the processors try to acquire lock at the same time when the lock is released (a lot of traffic).

```
void Lock(int* lock) {
	while (1) {
		while (*lock != 0);
		
		if (test_and_set(*lock) == 0)
			return;
	}
}

void Unlock(int* lock) {
	*lock = 0;
}
```

![](images/Pasted%20image%2020220401234556.png)

**Ticket lock**

When acquiring the lock, each thread is given an integer as a ticket. Each only needs to write one time when taking a ticket. No need to write when actually acquiring a lock.

```c++
struct lock {
	int next_ticket;
	int now_serving;
};

void Lock(lock* l) {
	int my_ticket = atomic_increment(&l->next_ticket);
	while (my_ticket != l->now_serving);
}

void Unlock(lock* l) {
	l->now_serving++;
}
```

**CUDA atomic compare-and-swap**

Atomic function in CUDA that acts like test-and-set.

```
int atomicCAS(int* addr, int compare, int val) {
	int old = *addr;
	*addr = (old == compare) ? val : old;
	return old;
}
```

How do you implement atomic_min? Atomicity is important because if we load two integers then do the comparison, another thread may have modified one of the integers in the meantime.

The implementation below makes sure that the operations are carried out correctly, even when others have modified the memory.

```
int atomic_min(int* addr, int x) {
	int old = *addr;
	int new = min(old, x);
	/* If the address value is not changed, update it */
	while (atomicCAS(addr, old, new) != old) {
		old = *addr;
		new = min(old, x);
	}
}
```

Implementing a lock with compare-and-swap.

```c
// Simple
typedef int lock;
void lock(Lock* l) {
	while (atomicCAS(1, 0, 1) == 1);
}
void unlock(Lock* l) {
	*l = 0;
}

// More efficient
void lock(Lock* l) {
	while (1) {
		while (*l == 1);
		if (atomicCAS(1, 0, 1) == 0)
			return;
	}
}
```

**C++ 11 atomic\<T\>**: Provides atomic read, write, read-modify-write of entire C++ objects. Provides memory ordering semantics for operations before and after the atomic operations (Default: Sequential consistency).

**Example of sorted linked list**: For subsequent demonstrations of simultaneous operations.

```c
struct Node {
	int value;
	Node* next;
};
struct List {
	Node* head;
};

void insert(List* list, int value) {
	Node* n = new Node;
	n->value = value;
	
	// Assume insert before head is handled
	
	Node* prev = list->head;
	Node* cur = list->head->next;
	
	while (cur) {
		if (cur->value > value)
			break;
		
		prev = cur;
		cur = cur->next;
	}
	
	n->next = cur;
	prev->next = n;
}

void delete(List* list, int value) {
	// Assume delete first node is handled
	
	Node* prev = list->head;
	Node* cur = list->head->next;
	
	while (cur) {
		if (cur->value == value) {
			prev->next = cur->next;
			delete cur;
			return;
		}
		
		prev = cur;
		cur = cur->next;
	}
}
```