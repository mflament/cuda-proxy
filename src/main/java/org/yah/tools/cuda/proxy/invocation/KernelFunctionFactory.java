package org.yah.tools.cuda.proxy.invocation;

import com.sun.jna.Pointer;
import org.yah.tools.cuda.proxy.KernelBindingException;
import org.yah.tools.cuda.proxy.annotations.BlockDim;
import org.yah.tools.cuda.proxy.annotations.GridDim;
import org.yah.tools.cuda.proxy.annotations.GridThreads;
import org.yah.tools.cuda.proxy.annotations.Kernel;
import org.yah.tools.cuda.proxy.annotations.SharedMemory;
import org.yah.tools.cuda.proxy.annotations.Writer;
import org.yah.tools.cuda.proxy.dim3;
import org.yah.tools.cuda.proxy.services.KernelArgumentWriter;
import org.yah.tools.cuda.proxy.services.ServiceFactory;
import org.yah.tools.cuda.proxy.services.TypeWriterRegistry;

import javax.annotation.Nullable;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class KernelFunctionFactory {

    private final ServiceFactory serviceFactory;
    private final TypeWriterRegistry writerRegistry;
    private final Function<String, Pointer> functionPointerResolver;

    public KernelFunctionFactory(ServiceFactory serviceFactory, TypeWriterRegistry writerRegistry, Function<String, Pointer> functionPointerResolver) {
        this.serviceFactory = serviceFactory;
        this.writerRegistry = writerRegistry;
        this.functionPointerResolver = functionPointerResolver;
    }

    @Nullable
    public KernelFunction create(Method method) {
        Kernel annotation = method.getAnnotation(Kernel.class);
        if (annotation == null)
            return null;

        if (method.getReturnType() != Void.TYPE)
            throw new KernelBindingException("Kernel method " + method + " must return void");

        String functionName = annotation.name().isBlank() ? method.getName() : annotation.name();
        Pointer function = functionPointerResolver.apply(functionName);

        dim3 kernelBlockDim = new dim3(annotation.blockDim());
        BlockDimFactory blockDimFactory = args -> kernelBlockDim;

        GridDimFactory gridDimFactory = null;
        if (annotation.gridDim().length > 0) {
            dim3 kernelGridDim = new dim3(annotation.gridDim());
            gridDimFactory = (args, blockDim) -> kernelGridDim;
        }

        if (annotation.gridThreads().length > 0) {
            if (gridDimFactory != null)
                throw newKernelBindingException(method, "Either @GridDim or @GridThreads can be declared, not both");
            dim3 kernelThreads = new dim3(annotation.gridThreads());
            gridDimFactory = (args, blockDim) -> dim3.fromThreads(kernelThreads, blockDim);
        }

        int kernelSharedMemory = annotation.sharedMemory();
        SharedMemorySupplier sharedMemorySupplier = (args, blockDim, gridDim) -> kernelSharedMemory;

        Parameter[] parameters = method.getParameters();
        List<KernelArgumentWriter<Object>> argumentWriters = new ArrayList<>(parameters.length);
        long nativeArgumentsSize = 0;
        int kernelParameterCount = 0;
        int blockDimParameter = -1, gridDimParameter = -1, sharedMemoryParameter = -1;
        for (int i = 0; i < parameters.length; i++) {
            Parameter parameter = parameters[i];

            boolean exposed = false;
            BlockDim blockDim = parameter.getAnnotation(BlockDim.class);
            if (blockDim != null) {
                if (blockDimParameter >= 0)
                    throw newKernelBindingException(method, "Duplicate @BlockDim, already set on " + parameters[blockDimParameter].getName());
                blockDimParameter = i;
                blockDimFactory = createBlockDimFactory(parameter, i);
            }

            GridDim gridDim = parameter.getAnnotation(GridDim.class);
            if (gridDim != null) {
                if (gridDimParameter >= 0)
                    throw newKernelBindingException(method, "Duplicate @GridDim, already set on " + parameters[gridDimParameter].getName());
                gridDimParameter = i;
                gridDimFactory = createGridDimFactory(parameter, i);
            }

            GridThreads gridThreads = parameter.getAnnotation(GridThreads.class);
            if (gridThreads != null) {
                if (gridDimParameter >= 0)
                    throw newKernelBindingException(method, "Duplicate @GridDim, already set on " + parameters[gridDimParameter].getName());
                gridDimParameter = i;
                gridDimFactory = createGridDimFromThreadsFactory(parameter, i);
                exposed |= gridThreads.exposed();
            }

            SharedMemory sharedMemory = parameter.getAnnotation(SharedMemory.class);
            if (sharedMemory != null) {
                if (sharedMemoryParameter >= 0)
                    throw newKernelBindingException(method, "Duplicate @sharedMemory, already set on " + parameters[sharedMemoryParameter].getName());
                sharedMemoryParameter = i;
                sharedMemorySupplier = createsharedMemorySupplier(parameter, i);
                exposed |= sharedMemory.exposed();
            }

            if (blockDim == null && gridDim == null && gridThreads == null && sharedMemory == null) // no annotation, must be a kernel parameter
                exposed = true;

            if (exposed) {
                KernelArgumentWriter<Object> argumentWriter = createArgumentWriter(parameter);
                nativeArgumentsSize += argumentWriter.size();
                kernelParameterCount++;
                argumentWriters.add(argumentWriter);
            } else {
                argumentWriters.add(null);
            }
        }

        if (gridDimFactory == null)
            gridDimFactory = (args, blockDim) -> new dim3(0, 0, 0);
        return new KernelFunction(method, function, blockDimFactory, gridDimFactory, argumentWriters, sharedMemorySupplier,
                nativeArgumentsSize, kernelParameterCount);
    }

    @SuppressWarnings("unchecked")
    private KernelArgumentWriter<Object> createArgumentWriter(Parameter parameter) {
        Writer annotation = parameter.getAnnotation(Writer.class);
        KernelArgumentWriter<?> writer;
        Class<?> parameterType = parameter.getType();
        if (annotation != null) {
            writer = serviceFactory.getInstance(annotation.value(), parameter);
        } else {
            writer = writerRegistry.find(parameterType).orElse(null);
        }
        if (writer == null)
            throw newKernelBindingException(parameter, "No writer resolved for parameterType parameterType " + parameterType.getName());

        Class<?> writerParameterType = getWriterParameterType(parameter, writer);
        if (!writerParameterType.isAssignableFrom(boxed(parameterType)))
            throw newKernelBindingException(parameter, "Incompatible KernelArgumentWriter<" + writerParameterType.getName() + ">, parameterType '" + parameterType.getName() + "'  parameterType is " + parameterType);

        return (KernelArgumentWriter<Object>) writer;

    }

    private BlockDimFactory createBlockDimFactory(Parameter parameter, int i) {
        if (dim3.class.isAssignableFrom(parameter.getType()))
            return args -> (dim3) args[i];
        if (int.class.isAssignableFrom(parameter.getType()))
            return args -> new dim3((int) args[i]);
        throw newKernelBindingException(parameter, "Invalid @BlockDim parameter type " + parameter.getType().getName() + ", must be 'dim3'");
    }

    private GridDimFactory createGridDimFactory(Parameter parameter, int i) {
        if (dim3.class.isAssignableFrom(parameter.getType()))
            return (args, blockDim) -> (dim3) args[i];
        if (int.class.isAssignableFrom(parameter.getType()))
            return (args, blockDim) -> new dim3((int) args[i]);
        throw newKernelBindingException(parameter, "Invalid @GridDim parameter type " + parameter.getType().getName() + ", must be 'dim3'");
    }

    private GridDimFactory createGridDimFromThreadsFactory(Parameter parameter, int i) {
        if (dim3.class.isAssignableFrom(parameter.getType()))
            return (args, blockDim) -> dim3.fromThreads((dim3) args[i], blockDim);
        if (int.class.isAssignableFrom(parameter.getType()))
            return (args, blockDim) -> new dim3(dim3.blocks((int) args[i], blockDim.x()));
        throw newKernelBindingException(parameter, "Invalid @GridThreads parameter type " + parameter.getType().getName() + ", must be 'dim3'");
    }

    private SharedMemorySupplier createsharedMemorySupplier(Parameter parameter, int i) {
        Class<?> type = parameter.getType();
        if (type != Integer.TYPE)
            throw newKernelBindingException(parameter, "Invalid @ShaderMemory parameter type " + parameter.getType().getName() + ", must be 'int'");
        return (args, blockDim, gridDim) -> (int) args[i];
    }

    private KernelBindingException newKernelBindingException(Method method, String message) {
        return new KernelBindingException(String.format("Error in kernel method %s : %s", method.getName(), message));
    }

    private KernelBindingException newKernelBindingException(Parameter parameter, String message) {
        Method method = (Method) parameter.getDeclaringExecutable();
        return new KernelBindingException(String.format("Error in kernel parameter '%s' of method '%s#%s' : %s", parameter.getName(), method.getDeclaringClass().getName(), method.getName(), message));
    }

    private Class<?> getWriterParameterType(Parameter parameter, KernelArgumentWriter<?> writer) {
        for (Type genericInterface : writer.getClass().getGenericInterfaces()) {
            if (genericInterface instanceof ParameterizedType) {
                ParameterizedType pt = (ParameterizedType) genericInterface;
                Class<?> rawType = (Class<?>) pt.getRawType();
                if (rawType == KernelArgumentWriter.class) {
                    Type actualTypeArgument = pt.getActualTypeArguments()[0];
                    if (actualTypeArgument instanceof Class)
                        return (Class<?>) actualTypeArgument;
                    throw new IllegalStateException("Invalid " + rawType.getSimpleName() + " type argument " + actualTypeArgument);
                }
            }
        }
        throw newKernelBindingException(parameter, "KernelArgumentWriter interface not found on writer " + writer);
    }

    private static Class<?> boxed(Class<?> type) {
        if (type == byte.class) return Byte.class;
        if (type == short.class) return Short.class;
        if (type == int.class) return Integer.class;
        if (type == long.class) return Long.class;
        if (type == float.class) return Float.class;
        if (type == double.class) return Double.class;
        return type;
    }
}

