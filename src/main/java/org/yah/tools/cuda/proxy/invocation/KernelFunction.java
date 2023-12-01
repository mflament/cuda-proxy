package org.yah.tools.cuda.proxy.invocation;

import com.sun.jna.Pointer;
import org.yah.tools.cuda.proxy.dim3;
import org.yah.tools.cuda.proxy.services.KernelArgumentWriter;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Objects;

public final class KernelFunction {

    private final Method method;
    private final Pointer functionPtr;
    private final BlockDimFactory blockDimFactory;
    private final GridDimFactory gridDimFactory;
    private final SharedMemorySupplier sharedMemorySupplier;

    private final List<KernelArgumentWriter<Object>> argumentWriters;
    private final long nativeArgumentsSize;
    private final int kernelParameterCount;

    public KernelFunction(Method method, Pointer functionPtr,
                          BlockDimFactory blockDimFactory,
                          GridDimFactory gridDimFactory,
                          List<KernelArgumentWriter<Object>> argumentWriters,
                          SharedMemorySupplier sharedMemorySupplier,
                          long nativeArgumentsSize, int kernelParameterCount) {
        this.method = method;
        this.functionPtr = Objects.requireNonNull(functionPtr, "functionPtr is null");
        this.blockDimFactory = Objects.requireNonNull(blockDimFactory, "blockDim is null");
        this.gridDimFactory = Objects.requireNonNull(gridDimFactory, "gridDimFactory is null");
        this.argumentWriters = Objects.requireNonNull(argumentWriters, "argumentWriters is null");
        this.sharedMemorySupplier = sharedMemorySupplier;
        this.nativeArgumentsSize = nativeArgumentsSize;
        this.kernelParameterCount = kernelParameterCount;

    }

    public Method method() {
        return method;
    }

    public Pointer functionPtr() {
        return functionPtr;
    }

    public dim3 blockDim(Object[] args) {
        return blockDimFactory.createBlockDim(args);
    }

    public dim3 gridDim(Object[] args, dim3 blockDim) {
        return gridDimFactory.createGridDim(args, blockDim);
    }

    public int sharedMemory(Object[] args, dim3 blockDim, dim3 gridDim) {
        return sharedMemorySupplier.get(args, blockDim, gridDim);
    }

    public KernelArgumentWriter<Object> argumentWriter(int i) {
        return argumentWriters.get(i);
    }

    public long nativeArgumentsSize() {
        return nativeArgumentsSize;
    }

    public int kernelParameterCount() {
        return kernelParameterCount;
    }

}
