# JCuda wrapper

Simple Cuda kernel wrapper for Java.

### Simple example
Let's say you have this simple cuda kernel in a file `sum.cu`
```cuda
extern "C" __global__ void sum(int N, int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
```

The kernel method can be mapped in java with this interface
```java
public interface SumModule extends AutoCloseable {
    int BLOCK_DIM = 1024;

    @Kernel(blockDim = {BLOCK_DIM, 1, 1})
    void sum(@GridDim(dim = DimName.x, exposed = false) int gridDimX, int N, Pointer a, Pointer b, Pointer c);

    /**
     * Will unload the cuModule
     */
    @Override
    void close();
}
```
@kernel annotation will look up for a cuda function with the same name, or with name configured in the annotation.
It also accepts the blockDim used for this kernel (missing dimension default to 1).

@GridDim will extract the gridDim (number of grid blocks) from the annotated parameter, using the given converter (default to identity, but only works for integer parameter). 
It will also be exposed as a kernel parameter if not configured with exposed=false.

@GridDims can be uses to create the all the gridDims from a single parameter (ie: texture/cube data). 

For now, the parameter type that can be sent to the kernel are JNA Pointer or primitive byte,short,int,long,float,double or enum (as an int containing the ordinal).
It's possible to annotate the parameter with [Writer](src%2Fmain%2Fjava%2Forg%2Fyah%2Ftools%2Fjcuda%2Fsupport%2Fmodule%2Fannotations%2FWriter.java) annotation to configure a custom [KernelArgumentWriter](src%2Fmain%2Fjava%2Forg%2Fyah%2Ftools%2Fjcuda%2Fsupport%2Fmodule%2FKernelArgumentWriter.java) to marshall java type to native memory. 

The previous kernel method can then be called using this code:
```java
public final class TestModule {
    
    public static void main(String[] args) throws IOException {
        // initialize Cuda driver API
        DevicePointer device = DriverSupport.getDevice(0);
        CudaContextPointer ctx = device.createContext(0);
        ctx.setCurrent();
    
        // load cuda source
        String source = Files.readString(Path.of("sum.cu"));
        // create the cuda program (and compile for current device GPU compute version)
        CudaProgramPointer program = CudaProgramBuilder.create(source)
                .withProgramName("sumKernel")
                .withComputeVersion(device)
                .build();
    
        // create the module build, it will introspect the given interface and create a proxy to launch any method annotated with @Kernel
        CudaModuleBuilder cudaModuleBuilder = new CudaModuleBuilder();
    
        int N = 512;
        // create java heap data
        int[] a = IntStream.range(0, N).toArray(), b = IntStream.range(N, 2 * N).toArray();
    
        // prepare a host memory used to transfer from/to java heap to/from cuda device
        try (Memory hostMemory = new Memory(N * (long) Integer.BYTES);
             // proxy will unload the module on close
             SumModule module = cudaModuleBuilder.createModule(program, SumModule.class)) {
            // allocate and copy java a and b array to device
            Pointer aPtr = copyToDevice(a, hostMemory);
            Pointer bPtr = copyToDevice(b, hostMemory);
    
            // allocate result array on device
            Pointer cPtr = allocateInts(N);
    
            // launch the kernel
            module.sum(Dim.blocks(N, SumModule.BLOCK_DIM), N, aPtr, bPtr, cPtr);
            // wait for completion, check for error
            CudaContextPointer.synchronize();
    
            // read back results
            check(driverAPI().cuMemcpyDtoH(hostMemory, cPtr, N * (long) Integer.BYTES));
            for (int i = 0; i < N; i++) {
                int expected = a[i] + b[i];
                int actual = hostMemory.getInt(i * Integer.BYTES);
                assertEquals(expected, actual);
            }
            System.out.println("Ok");
        }
    
    }
    
    private static Pointer copyToDevice(int[] a, Memory hostMemory) {
        hostMemory.write(0, a, 0, a.length);
        Pointer ptr = allocateInts(a.length);
        check(driverAPI().cuMemcpyHtoD(ptr, hostMemory, a.length * (long) Integer.BYTES));
        return ptr;
    }
    
    private static Pointer allocateInts(int n) {
        PointerByReference ptr = new PointerByReference();
        check(driverAPI().cuMemAlloc(ptr, n * (long) Integer.BYTES));
        return ptr.getValue();
    }
}
```