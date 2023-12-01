package org.yah.tools.jcuda.support.module.invocation;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.yah.tools.jcuda.support.module.dim3;
import org.yah.tools.jcuda.support.module.services.KernelArgumentWriter;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.yah.tools.jcuda.support.DriverSupport.check;
import static org.yah.tools.jcuda.support.DriverSupport.driverAPI;

/**
 * Intercept kernel method invocations and launch the corresponding kernel function.
 * Note: not thread safe.
 */
public class KernelsInvocationHandler implements InvocationHandler {

    private final Pointer module;
    private final Map<InvocationKey, KernelFunction> kernelFunctions;

    /**
     * Host memory used to copy java argument to native memory.
     */
    private final Memory stack;
    /**
     * Host memory containing pointers to each argument (can be from stack, or device pointer
     */
    private final Memory parameterPointers;

    public KernelsInvocationHandler(Pointer module, List<KernelFunction> kernelFunctions) {
        this.module = module;
        this.kernelFunctions = kernelFunctions.stream().collect(Collectors.toMap(InvocationKey::new, Function.identity()));

        long maxStackSize = kernelFunctions.stream().mapToLong(KernelFunction::nativeArgumentsSize).max().orElse(0);
        stack = maxStackSize == 0 ? null : new Memory(maxStackSize);

        int maxParametersCount = kernelFunctions.stream().mapToInt(KernelFunction::kernelParameterCount).max().orElse(0);
        parameterPointers = new Memory(maxParametersCount * (long) Native.POINTER_SIZE);
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws InvocationTargetException, IllegalAccessException {
        if (method.getName().equals("close") && method.getParameterCount() == 0) {
            check(driverAPI().cuModuleUnload(module));
            return null;
        }

        KernelFunction kernelFunction = kernelFunctions.get(new InvocationKey(method));
        if (kernelFunction == null)
            return method.invoke(proxy, args);

        dim3 blockDim = kernelFunction.blockDim(args);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        int sharedMemory = kernelFunction.sharedMemory(args, blockDim, gridDim);
        final Pointer argumentsMemory = stack.share(0, kernelFunction.nativeArgumentsSize());
        long stackOffset = 0;
        int kernelArgIndex = 0;

        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            KernelArgumentWriter<Object> writer = kernelFunction.argumentWriter(i);
            if (writer == null)
                continue;
            long size = writer.size();
            Pointer argumentPtr = argumentsMemory.share(stackOffset, size);
            writer.write(arg, argumentPtr);
            stackOffset += size;
            parameterPointers.setPointer(kernelArgIndex * (long) Native.POINTER_SIZE, argumentPtr);
            kernelArgIndex++;
        }

        Pointer funcPtr = kernelFunction.functionPtr();
        check(driverAPI().cuLaunchKernel(funcPtr,
                gridDim.x(), gridDim.y(), gridDim.z(),
                blockDim.x(), blockDim.y(), blockDim.z(),
                sharedMemory, Pointer.NULL, parameterPointers, Pointer.NULL));

        return null;
    }

    private static final class InvocationKey {
        private final String methodName;
        private final Class<?>[] parameterTypes;
        private final int hashCode;

        public InvocationKey(KernelFunction kernelFunction) {
            this(kernelFunction.method());
        }

        public InvocationKey(Method method) {
            this.methodName = method.getName();
            this.parameterTypes = method.getParameterTypes();
            hashCode = 31 * Objects.hash(methodName) + Arrays.hashCode(parameterTypes);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            InvocationKey that = (InvocationKey) o;
            return Objects.equals(methodName, that.methodName) && Arrays.equals(parameterTypes, that.parameterTypes);
        }

        @Override
        public int hashCode() {
            return hashCode;
        }
    }
}
