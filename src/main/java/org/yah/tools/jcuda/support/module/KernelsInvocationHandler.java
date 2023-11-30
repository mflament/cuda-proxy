package org.yah.tools.jcuda.support.module;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import javax.annotation.Nullable;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.yah.tools.jcuda.support.DriverSupport.check;
import static org.yah.tools.jcuda.support.DriverSupport.driverAPI;

public class KernelsInvocationHandler implements InvocationHandler {

    private final Pointer module;
    private final Map<Method, KernelInvocation> kernelInvocations;

    private final Memory stack;
    private final Memory parameterPointers;

    public KernelsInvocationHandler(Pointer module, List<KernelInvocation> kernelInvocations) {
        this(module, kernelInvocations, 1024);
    }

    public KernelsInvocationHandler(Pointer module, List<KernelInvocation> kernelInvocations, int stackSize) {
        this.module = module;
        this.kernelInvocations = kernelInvocations.stream().collect(Collectors.toMap(KernelInvocation::method, Function.identity()));
        stack = new Memory(stackSize);
        int pointersCount = kernelInvocations.stream().mapToInt(KernelInvocation::getKernelParameterCount).max().orElse(0);
        parameterPointers = new Memory(pointersCount * (long) Native.POINTER_SIZE);
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) {
        if (method.getName().equals("close") && method.getParameterCount() == 0) {
            check(driverAPI().cuModuleUnload(module));
            return null;
        }

        KernelInvocation kernelInvocation = kernelInvocations.get(method);
        if (kernelInvocation == null)
            throw new IllegalStateException("Method " + method.getName() + " is not a kernel method");

        Pointer funcPtr = kernelInvocation.functionPtr().getValue();
        Dim blockDim = kernelInvocation.blockDim();
        Dim gridDim = kernelInvocation.gridDimFactory().createGridDim(args);

        Pointer ptr = stack;
        int pointerIndex = 0;
        Iterator<KernelArgumentWriter<Object>> iterator = kernelInvocation.argumentWriters().iterator();
        for (int i = 0; i < args.length && iterator.hasNext(); i++) {
            KernelArgumentWriter<Object> writer = iterator.next();
            if (writer == null)
                continue;
            long size = writer.write(args[i], ptr);
            parameterPointers.setPointer(pointerIndex++ * (long) Native.POINTER_SIZE, ptr);
            ptr = ptr.share(size);
        }
        int sharedMemBytes = kernelInvocation.sharedMemory();
        check(driverAPI().cuLaunchKernel(funcPtr,
                gridDim.x(), gridDim.y(), gridDim.z(),
                blockDim.x(), blockDim.y(), blockDim.z(),
                sharedMemBytes, Pointer.NULL, parameterPointers, Pointer.NULL));

        // noinspection SuspiciousInvocationHandlerImplementation : if equals is marked with @Kernel ... an error is deserverd as punishment :-)
        return null;
    }
}
