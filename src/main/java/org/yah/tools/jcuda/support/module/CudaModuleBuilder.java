package org.yah.tools.jcuda.support.module;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.jcuda.support.CudaException;
import org.yah.tools.jcuda.support.module.annotations.GridDim;
import org.yah.tools.jcuda.support.module.annotations.GridDims;
import org.yah.tools.jcuda.support.module.annotations.Kernel;
import org.yah.tools.jcuda.support.module.annotations.Writer;
import org.yah.tools.jcuda.support.program.CudaProgramPointer;

import javax.annotation.Nullable;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import static org.yah.tools.jcuda.jna.RuntimeAPI.CUDA_ERROR_INVALID_IMAGE;
import static org.yah.tools.jcuda.support.DriverSupport.check;
import static org.yah.tools.jcuda.support.DriverSupport.driverAPI;
import static org.yah.tools.jcuda.support.NTSHelper.readNTS;
import static org.yah.tools.jcuda.support.NTSHelper.writeNTS;

public class CudaModuleBuilder {

    private static final Logger LOGGER = LoggerFactory.getLogger(CudaModuleBuilder.class);

    private final ServiceFactory serviceFactory;
    private final LinkedList<TypeArgumentWriter<?>> typeWriters = new LinkedList<>();

    public CudaModuleBuilder() {
        this(new DefaultServiceFactory());
    }

    public CudaModuleBuilder(ServiceFactory serviceFactory) {
        this.serviceFactory = serviceFactory;
        typeWriters.addAll(createDefaultConverters());
    }

    private Collection<TypeArgumentWriter<?>> createDefaultConverters() {
        //TODO : primitive, enum
        return List.of(
                new TypeArgumentWriter<>(Byte.TYPE, DefaultWriters::writeByte),
                new TypeArgumentWriter<>(Short.TYPE, DefaultWriters::writeShort),
                new TypeArgumentWriter<>(Integer.TYPE, DefaultWriters::writeInt),
                new TypeArgumentWriter<>(Long.TYPE, DefaultWriters::writeLong),
                new TypeArgumentWriter<>(Float.TYPE, DefaultWriters::writeFloat),
                new TypeArgumentWriter<>(Double.TYPE, DefaultWriters::writeDouble),
                new TypeArgumentWriter<>(Enum.class, DefaultWriters::writeEnum),
                new TypeArgumentWriter<>(Pointer.class, DefaultWriters::writePointer)
        );
    }

    private static final class DefaultWriters {
        static long writeByte(byte b, Pointer dst) {
            dst.setByte(0, b);
            return Byte.BYTES;
        }

        static long writeShort(short s, Pointer dst) {
            dst.setShort(0, s);
            return Short.BYTES;
        }

        static long writeInt(int i, Pointer dst) {
            dst.setInt(0, i);
            return Integer.BYTES;
        }

        static long writeLong(long l, Pointer dst) {
            dst.setLong(0, l);
            return Long.BYTES;
        }

        static long writeFloat(float f, Pointer dst) {
            dst.setFloat(0, f);
            return Float.BYTES;
        }

        static long writeDouble(double b, Pointer dst) {
            dst.setDouble(0, b);
            return Double.BYTES;
        }

        static long writePointer(Pointer p, Pointer dst) {
            dst.setPointer(0, p);
            return Native.POINTER_SIZE;
        }

        static long writeEnum(Enum<?> e, Pointer dst) {
            return writeInt(e.ordinal(), dst);
        }

    }

    public <T> CudaModuleBuilder addTypeConverter(Class<T> parameterType, KernelArgumentWriter<T> converter) {
        typeWriters.addFirst(new TypeArgumentWriter<>(parameterType, converter));
        return this;
    }

    @SuppressWarnings("unchecked")
    public <T> T createModule(CudaProgramPointer program, Class<T> nativeInterface) {
        List<Method> kernelMethods = collectKernelMethods(nativeInterface);
        try (Memory ptx = program.getPTX()) {
            PointerByReference module = new PointerByReference();
            try {
                check(driverAPI().cuModuleLoadData(module, ptx));
            } catch (CudaException e) {
                if (e.getStatus() == CUDA_ERROR_INVALID_IMAGE)
                    LOGGER.info("PTX:\n{}", readNTS(ptx, ptx.size()));
                throw e;
            }

            List<KernelInvocation> kernelInvocations = new ArrayList<>();
            for (Method method : kernelMethods) {
                Kernel annotation = checkKernel(method);
                String name = annotation.name().isBlank() ? method.getName() : annotation.name();
                Dim blockDim = new Dim(annotation.blockDim());
                GridDimFactory gridDimFactory = createGridDimFactory(method, blockDim);
                Parameter[] parameters = method.getParameters();
                List<KernelArgumentWriter<Object>> argumentWriters = new ArrayList<>(parameters.length);
                for (Parameter parameter : parameters)
                    argumentWriters.add(createArgumentWrite(parameter));
                PointerByReference function = getFunction(module, name);

                KernelInvocation kernelInvocation = new KernelInvocation(method, function, blockDim, gridDimFactory,
                        argumentWriters, annotation.sharedMemory());
                kernelInvocations.add(kernelInvocation);
            }
            InvocationHandler invocationHandler = new KernelsInvocationHandler(kernelInvocations);
            return (T) Proxy.newProxyInstance(nativeInterface.getClassLoader(), new Class<?>[]{nativeInterface}, invocationHandler);
        }
    }

    @SuppressWarnings("unchecked")
    @Nullable
    private KernelArgumentWriter<Object> createArgumentWrite(Parameter parameter) {
        GridDim gridDim = parameter.getAnnotation(GridDim.class);
        if (gridDim != null && !gridDim.exposed()) // parameter only used as gridDim, not a kernel parameter
            return null;
        GridDims gridDims = parameter.getAnnotation(GridDims.class);
        if (gridDims != null && !gridDims.exposed()) // parameter only used as gridDims, not a kernel parameter
            return null;

        Writer annotation = parameter.getAnnotation(Writer.class);
        KernelArgumentWriter<?> writer;
        if (annotation != null) { // parameter declare its own writer
            writer = serviceFactory.getInstance(annotation.value());
            Class<?> parameterType = getFirstParameterType(writer, KernelArgumentWriter.class);
            if (!parameterType.isAssignableFrom(parameter.getType()))
                throw new KernelBindingException("Incompatible KernelArgumentWriter<" + parameterType.getName() + ">, parameter '" + parameter.getName() + "'  type is " + parameter.getType());
        } else {  // resolved converter from typeConverters
            Class<?> parameterType = parameter.getType();
            TypeArgumentWriter<?> typeWriter = typeWriters.stream()
                    .filter(tc -> tc.accept(parameterType))
                    .findFirst()
                    .orElseThrow(() -> new KernelBindingException("No KernelArgumentWriter resolved for parameter " + parameter.getName() + " with type " + parameter.getType().getTypeName()));
            writer = typeWriter.writer();
        }

        return (KernelArgumentWriter<Object>) writer;
    }

    private GridDimFactory createGridDimFactory(Method method, Dim blockDim) {
        GridDimSupplier[] suppliers = new GridDimSupplier[3];
        Parameter[] parameters = method.getParameters();
        int gridDimsParameterIndex = -1;
        for (int i = 0; i < parameters.length; i++) {
            Parameter parameter = parameters[i];
            if (parameter.getAnnotation(GridDims.class) != null) {
                if (gridDimsParameterIndex >= 0)
                    throw new KernelBindingException("Duplicate GridDims : " + parameter.getName() + ", " + parameters[gridDimsParameterIndex]);
                gridDimsParameterIndex = i;
            }

            GridDim gridDim = parameter.getAnnotation(GridDim.class);
            if (gridDim == null)
                continue;
            if (gridDimsParameterIndex >= 0)
                throw new KernelBindingException("Mixed GridDim (" + parameter.getName() + ") and GridDims (" + parameters[gridDimsParameterIndex].getName() + ")");
            int dimIndex = gridDim.dim().ordinal();
            if (suppliers[dimIndex] != null)
                throw new KernelBindingException("Duplicate GridDim(" + gridDim.dim() + ") : " + parameter.getName() + ", " + parameters[suppliers[dimIndex].argIndex].getName());
            suppliers[dimIndex] = createGridDimSupplier(gridDim, parameter, i);
        }

        if (gridDimsParameterIndex >= 0) {
            Parameter parameter = parameters[gridDimsParameterIndex];
            return createGridDimFactory(parameter.getAnnotation(GridDims.class), parameter, gridDimsParameterIndex, blockDim);
        }

        return args -> {
            int x = suppliers[0] != null ? suppliers[0].get(args, blockDim) : 1;
            int y = suppliers[1] != null ? suppliers[1].get(args, blockDim) : 1;
            int z = suppliers[2] != null ? suppliers[2].get(args, blockDim) : 1;
            return new Dim(x, y, z);
        };
    }


    private static Kernel checkKernel(Method kernelMethod) {
        Kernel annotation = kernelMethod.getAnnotation(Kernel.class);
        if (annotation == null)
            throw new IllegalStateException("no kernel annotation on method " + kernelMethod);
        if (kernelMethod.getReturnType() != Void.TYPE)
            throw new KernelBindingException("Kernel method " + kernelMethod + " must return void");
        return annotation;
    }

    private static final class GridDimSupplier {
        private final GridDimConverter<Object> converter;
        private final int argIndex;

        public GridDimSupplier(GridDimConverter<Object> converter, int argIndex) {
            this.converter = converter;
            this.argIndex = argIndex;
        }

        public int get(Object[] args, Dim blockDim) {
            return converter.toDim(args[argIndex], blockDim);
        }
    }

    @SuppressWarnings("unchecked")
    private GridDimFactory createGridDimFactory(GridDims gridDims, Parameter parameter, int parameterIndex, Dim blockDim) {
        GridDimsConverter<?> converter = serviceFactory.getInstance(gridDims.converter());
        Class<?> converterParameterType = getFirstParameterType(converter, GridDimsConverter.class);
        if (!converterParameterType.isAssignableFrom(parameter.getType()))
            throw new KernelBindingException("Incompatible GridDimsConverter<" + converterParameterType.getName() + ">, parameter '" + parameter.getName() + "'  type is " + parameter.getType());
        return args -> ((GridDimsConverter<Object>) converter).toGridDim(args[parameterIndex], blockDim);
    }

    @SuppressWarnings("unchecked")
    private GridDimSupplier createGridDimSupplier(GridDim gridDim, Parameter parameter, int parameterIndex) {
        GridDimConverter<?> converter = serviceFactory.getInstance(gridDim.converter());
        Class<?> parameterType = parameter.getType();
        if (parameterType.isPrimitive())
            parameterType = boxed(parameterType);
        Class<?> supplierParameterType = getFirstParameterType(converter, GridDimConverter.class);
        if (!supplierParameterType.isAssignableFrom(parameterType))
            throw new KernelBindingException("Incompatible GridDimParameterConverter<" + supplierParameterType.getName() + ">, parameter '" + parameter.getName() + "'  type is " + parameter.getType());
        return new GridDimSupplier((GridDimConverter<Object>) converter, parameterIndex);
    }

    private static Class<?> boxed(Class<?> parameterType) {
        if (parameterType == Byte.TYPE) return Byte.class;
        if (parameterType == Short.TYPE) return Short.class;
        if (parameterType == Integer.TYPE) return Integer.class;
        if (parameterType == Long.TYPE) return Long.class;
        if (parameterType == Float.TYPE) return Float.class;
        if (parameterType == Double.TYPE) return Double.class;
        return parameterType;
    }

    private static Class<?> getFirstParameterType(Object instance, Class<?> interfaceType) {
        for (Type genericInterface : instance.getClass().getGenericInterfaces()) {
            if (genericInterface instanceof ParameterizedType) {
                ParameterizedType pt = (ParameterizedType) genericInterface;
                Class<?> rawType = (Class<?>) pt.getRawType();
                if (rawType == interfaceType) {
                    Type actualTypeArgument = pt.getActualTypeArguments()[0];
                    if (actualTypeArgument instanceof Class)
                        return (Class<?>) actualTypeArgument;
                    throw new KernelBindingException("Invalid " + rawType.getSimpleName() + " type argument " + actualTypeArgument);
                }
            }
        }
        throw new KernelBindingException("Converter interface " + interfaceType.getName() + " not found in " + instance.getClass().getName());
    }

    private PointerByReference getFunction(PointerByReference hmod, String name) {
        PointerByReference hfunc = new PointerByReference();
        try (Memory namePtr = new Memory(name.length() + 1)) {
            writeNTS(namePtr, name);
            check(driverAPI().cuModuleGetFunction(hfunc, hmod.getValue(), namePtr));
        }
        return hfunc;
    }


    private static List<Method> collectKernelMethods(Class<?> container) {
        List<Method> methods = new ArrayList<>();
        collectKernelMethods(container, methods);
        return methods;
    }

    private static void collectKernelMethods(Class<?> container, List<Method> bindings) {
        Class<?>[] interfaces = container.getInterfaces();
        for (Class<?> parent : interfaces) {
            collectKernelMethods(parent, bindings);
        }

        Method[] kernelMethods = container.getDeclaredMethods();
        for (Method method : kernelMethods) {
            Kernel annotation = method.getAnnotation(Kernel.class);
            if (annotation != null) {
                bindings.add(method);
            }
        }
    }

    private static final class TypeArgumentWriter<T> {
        private final Class<T> type;
        private final KernelArgumentWriter<T> writer;

        public TypeArgumentWriter(Class<T> type, KernelArgumentWriter<T> writer) {
            this.type = type;
            this.writer = writer;
        }

        boolean accept(Class<?> parameterType) {
            return type.isAssignableFrom(parameterType);
        }

        boolean accept(Object arg) {
            return type.isInstance(arg);
        }

        public KernelArgumentWriter<T> writer() {
            return writer;
        }
    }

}
