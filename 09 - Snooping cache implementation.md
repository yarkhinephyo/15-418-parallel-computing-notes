**Basic system design**: Single-level, write-back cache per processor. Cache can stall processor as it is carrying out coherence operations.

![](images/Pasted%20image%2020220319182239.png)

**System interconnect ATOMIC scenario**

An atomic shared bus. Only one cache can send data at a time. Client is granted access -> Client places command on bus -> Another client responds by placing command on bus -> Next client is granted access.

**Snoop results**: Responses of all caches must appear on bus. If any line is dirty, memory should not respond yet. If line is shared, cache cannot be loaded into "exclusive" state.

**Write-back buffer**: Replacing a dirty cache line requires reading the incoming line and writing the outgoing line. Ideally, the processor would like to read to continue as fast as possible. The buffer stores the dirty line which is flushed at a later time.

Bus-side controller check the cache tags and write-back buffer for snooping related tasks while the processor does its own thing.

![](images/Pasted%20image%2020220319184321.png)

**When is a write committed**: A write "commits" when a read-exclusive appears on bus and is acknowledged by all other caches. This is different from "completion" which means a store instruction is done.

**ATOMIC bus issue**: Bus is idle when responses of other caches are pending. This is not very efficient.