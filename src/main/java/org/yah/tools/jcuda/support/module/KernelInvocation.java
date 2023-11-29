package org.yah.tools.jcuda.support.module;

import com.sun.jna.ptr.PointerByReference;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Objects;

public final class KernelInvocation {

    private final Method method;
    private final PointerByReference functionPtr;
    private final Dim blockDim;
    private final GridDimFactory gridDimFactory;
    private final List<KernelArgumentWriter<Object>> argumentWriters;
    private final int sharedMemory;

    public KernelInvocation(Method method, PointerByReference functionPtr, Dim blockDim, GridDimFactory gridDimFactory,
                            List<KernelArgumentWriter<Object>> argumentWriters, int sharedMemory) {
        this.method = method;
        this.functionPtr = Objects.requireNonNull(functionPtr, "functionPtr is null");
        this.blockDim = Objects.requireNonNull(blockDim, "blockDim is null");
        this.gridDimFactory = Objects.requireNonNull(gridDimFactory, "gridDimFactory is null");
        this.argumentWriters = Objects.requireNonNull(argumentWriters, "argumentWriters is null");
        this.sharedMemory = sharedMemory;
    }

    public Method method() {
        return method;
    }

    public PointerByReference functionPtr() {
        return functionPtr;
    }

    public Dim blockDim() {
        return blockDim;
    }

    public GridDimFactory gridDimFactory() {
        return gridDimFactory;
    }

    public int sharedMemory() {
        return sharedMemory;
    }

    public List<KernelArgumentWriter<Object>> argumentWriters() {
        return argumentWriters;
    }

    public int getKernelParameterCount() {
        int count = 0;
        for (KernelArgumentWriter<Object> writer : argumentWriters) {
            if (writer != null)
                count++;
        }
        return count;
    }
}
