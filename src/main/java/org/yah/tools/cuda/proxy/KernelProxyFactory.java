package org.yah.tools.cuda.proxy;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.support.CudaException;
import org.yah.tools.cuda.proxy.invocation.KernelFunction;
import org.yah.tools.cuda.proxy.invocation.KernelFunctionFactory;
import org.yah.tools.cuda.proxy.invocation.KernelsInvocationHandler;
import org.yah.tools.cuda.proxy.services.DefaultServiceFactory;
import org.yah.tools.cuda.proxy.services.ListTypeWriterRegistry;
import org.yah.tools.cuda.proxy.services.ServiceFactory;
import org.yah.tools.cuda.proxy.services.TypeWriterRegistry;
import org.yah.tools.cuda.support.program.CudaProgramPointer;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.yah.tools.cuda.api.Runtime.CUDA_ERROR_INVALID_IMAGE;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NTSHelper.readNTS;
import static org.yah.tools.cuda.support.NTSHelper.writeNTS;

public class KernelProxyFactory {

    private static final Logger LOGGER = LoggerFactory.getLogger(KernelProxyFactory.class);

    private final ServiceFactory serviceFactory;
    private final TypeWriterRegistry typeWriterRegistry;

    public KernelProxyFactory() {
        this(new DefaultServiceFactory(), ListTypeWriterRegistry.create(true));
    }

    public KernelProxyFactory(ServiceFactory serviceFactory, TypeWriterRegistry typeWriterRegistry) {
        this.serviceFactory = Objects.requireNonNull(serviceFactory, "serviceFactory is null");
        this.typeWriterRegistry = Objects.requireNonNull(typeWriterRegistry, "typeWriterRegistry is null");
    }

    @SuppressWarnings("unchecked")
    public <T> T createKernelProxy(CudaProgramPointer program, Class<T> nativeInterface) {
        List<Method> kernelMethods = collectKernelMethods(nativeInterface);
        try (Memory ptx = program.getPTX()) {
            PointerByReference ptrRef = new PointerByReference();
            try {
                check(driverAPI().cuModuleLoadData(ptrRef, ptx));
            } catch (CudaException e) {
                if (e.getStatus() == CUDA_ERROR_INVALID_IMAGE)
                    LOGGER.info("PTX:\n{}", readNTS(ptx, ptx.size()));
                throw e;
            }

            Pointer module = ptrRef.getValue();
            List<KernelFunction> kernelFunctions = new ArrayList<>();
            KernelFunctionFactory functionFactory = new KernelFunctionFactory(serviceFactory, typeWriterRegistry, name -> getKernelFunction(module, name));
            for (Method method : kernelMethods) {
                KernelFunction kernelFunction = functionFactory.create(method);
                if (kernelFunction != null)
                    kernelFunctions.add(kernelFunction);
            }
            InvocationHandler invocationHandler = new KernelsInvocationHandler(module, kernelFunctions);
            return (T) Proxy.newProxyInstance(nativeInterface.getClassLoader(), new Class<?>[]{nativeInterface}, invocationHandler);
        }
    }

    private Pointer getKernelFunction(Pointer module, String name) {
        PointerByReference hfunc = new PointerByReference();
        try (Memory namePtr = new Memory(name.length() + 1)) {
            writeNTS(namePtr, name);
            check(driverAPI().cuModuleGetFunction(hfunc, module, namePtr));
        }
        return hfunc.getValue();
    }

    private static List<Method> collectKernelMethods(Class<?> container) {
        List<Method> methods = new ArrayList<>();
        collectKernelMethods(container, methods);
        return methods;
    }

    private static void collectKernelMethods(Class<?> container, List<Method> methods) {
        Class<?>[] interfaces = container.getInterfaces();
        for (Class<?> parent : interfaces) {
            collectKernelMethods(parent, methods);
        }
        Method[] kernelMethods = container.getDeclaredMethods();
        methods.addAll(Arrays.asList(kernelMethods));
    }

}
