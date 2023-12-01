package org.yah.tools.cuda.proxy.invocation;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.yah.tools.cuda.proxy.KernelBindingException;
import org.yah.tools.cuda.proxy.annotations.BlockDim;
import org.yah.tools.cuda.proxy.annotations.GridDim;
import org.yah.tools.cuda.proxy.annotations.GridThreads;
import org.yah.tools.cuda.proxy.annotations.Kernel;
import org.yah.tools.cuda.proxy.annotations.SharedMemory;
import org.yah.tools.cuda.proxy.annotations.Writer;
import org.yah.tools.cuda.proxy.dim3;
import org.yah.tools.cuda.proxy.services.DefaultWriters;
import org.yah.tools.cuda.proxy.services.KernelArgumentWriter;
import org.yah.tools.cuda.proxy.services.ServiceFactory;
import org.yah.tools.cuda.proxy.services.TypeWriterRegistry;
import org.yah.tools.cuda.proxy.services.Writable;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class KernelFunctionFactoryTest {

    private ServiceFactory serviceFactory;
    private TypeWriterRegistry writerRegistry;
    private Function<String, Pointer> functionPointerResolver;
    private KernelFunctionFactory factory;

    @SuppressWarnings("unchecked")
    @BeforeEach
    void setup() {
        serviceFactory = mock(ServiceFactory.class);
        writerRegistry = mock(TypeWriterRegistry.class);
        functionPointerResolver = mock(Function.class);
        factory = new KernelFunctionFactory(serviceFactory, writerRegistry, functionPointerResolver);
    }

    @Test
    void notAKernelIsIgnored() {
        assertNull(factory.create(getMethod("notAKernel")));
    }

    @Test
    void shouldReturnVoid() {
        assertThrows(KernelBindingException.class, () -> factory.create(getMethod("shouldReturnVoid")));
    }

    @Test
    void withGriDim() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("test01")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("withGriDim"));
        assertNotNull(kernelFunction);
        assertSame(functionPointer, kernelFunction.functionPtr());
        Object[] args = new Object[0];
        dim3 blockDim = kernelFunction.blockDim(args);
        assertEquals(new dim3(16, 16, 4), blockDim);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(new dim3(512, 256, 128), gridDim);
        int sharedMem = kernelFunction.sharedMemory(args, blockDim, gridDim);
        assertEquals(42, sharedMem);
    }

    @Test
    void withGridThreads() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("test02")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("withGridThreads"));
        assertNotNull(kernelFunction);
        assertSame(functionPointer, kernelFunction.functionPtr());
        Object[] args = new Object[0];
        dim3 blockDim = kernelFunction.blockDim(args);
        assertEquals(new dim3(16, 16, 4), blockDim);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(new dim3(512 / 16, 256 / 16, 128 / 4), gridDim);
        int sharedMem = kernelFunction.sharedMemory(args, blockDim, gridDim);
        assertEquals(0, sharedMem);
    }

    @Test
    void withGriDimAndGridThreads() {
        KernelBindingException exception = assertThrows(KernelBindingException.class, () -> factory.create(getMethod("withGriDimAndGridThreads")));
        assertTrue(exception.getMessage().contains("Either @GridDim or @GridThreads can be declared, not both"));
    }

    @Test
    void argIsNotAGridDim() {
        assertThrows(KernelBindingException.class, () -> factory.create(getMethod("argIsNotAGridDim")));
    }

    @Test
    void argIsGridDimX() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsGridDimX")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("argIsGridDimX"));
        assertNotNull(kernelFunction);
        Object[] args = {16, 512};
        dim3 blockDim = kernelFunction.blockDim(args);
        assertEquals(new dim3(16, 1, 1), blockDim);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(new dim3(512, 1, 1), gridDim);
        assertEquals(0, kernelFunction.nativeArgumentsSize());
        assertEquals(0, kernelFunction.kernelParameterCount());
        assertNull(kernelFunction.argumentWriter(0));
    }

    @Test
    void argIsGridDim() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsGridDim")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("argIsGridDim"));
        assertNotNull(kernelFunction);
        Object[] args = {new dim3(16, 16, 4), new dim3(512, 256, 128)};
        dim3 blockDim = kernelFunction.blockDim(args);
        assertEquals(new dim3(16, 16, 4), blockDim);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(new dim3(512, 256, 128), gridDim);
        assertEquals(0, kernelFunction.nativeArgumentsSize());
        assertEquals(0, kernelFunction.kernelParameterCount());
        assertNull(kernelFunction.argumentWriter(0));
    }

    @Test
    void argIsGridThreads() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsGridThreads")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("argIsGridThreads"));
        assertNotNull(kernelFunction);
        Object[] args = {new dim3(16, 16, 4), new dim3(512, 256, 128)};
        dim3 blockDim = kernelFunction.blockDim(args);
        assertEquals(new dim3(16, 16, 4), blockDim);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(new dim3(512 / 16, 256 / 16, 128 / 4), gridDim);
        assertEquals(0, kernelFunction.nativeArgumentsSize());
        assertEquals(0, kernelFunction.kernelParameterCount());
        assertNull(kernelFunction.argumentWriter(0));
    }

    @Test
    void argIsExposedGridThreads() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsExposedGridThreads")).thenReturn(functionPointer);
        when(writerRegistry.find(dim3.class)).thenReturn(Optional.of(DefaultWriters.Dim3Writer.INSTANCE));
        KernelFunction kernelFunction = factory.create(getMethod("argIsExposedGridThreads"));
        assertNotNull(kernelFunction);
        assertEquals(Integer.BYTES * 3, kernelFunction.nativeArgumentsSize());
        assertEquals(1, kernelFunction.kernelParameterCount());
        KernelArgumentWriter<?> argumentWriter = kernelFunction.argumentWriter(1);
        assertSame(DefaultWriters.Dim3Writer.INSTANCE, argumentWriter);
    }

    @Test
    void argIsSharedMem() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsSharedMem")).thenReturn(functionPointer);
        KernelFunction kernelFunction = factory.create(getMethod("argIsSharedMem"));
        assertNotNull(kernelFunction);
        Object[] args = {128};
        dim3 blockDim = kernelFunction.blockDim(args);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(128, kernelFunction.sharedMemory(args, blockDim, gridDim));
        assertEquals(0, kernelFunction.nativeArgumentsSize());
        assertEquals(0, kernelFunction.kernelParameterCount());
        assertNull(kernelFunction.argumentWriter(0));
    }

    @Test
    void argIsExposedSharedMem() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsExposedSharedMem")).thenReturn(functionPointer);
        when(writerRegistry.find(int.class)).thenReturn(Optional.of(DefaultWriters.IntegerWriter.INSTANCE));
        KernelFunction kernelFunction = factory.create(getMethod("argIsExposedSharedMem"));
        assertNotNull(kernelFunction);
        Object[] args = {128};
        dim3 blockDim = kernelFunction.blockDim(args);
        dim3 gridDim = kernelFunction.gridDim(args, blockDim);
        assertEquals(128, kernelFunction.sharedMemory(args, blockDim, gridDim));
        assertEquals(Integer.BYTES, kernelFunction.nativeArgumentsSize());
        assertEquals(1, kernelFunction.kernelParameterCount());
        KernelArgumentWriter<?> argumentWriter = kernelFunction.argumentWriter(0);
        assertSame(DefaultWriters.IntegerWriter.INSTANCE, argumentWriter);
    }

    @Test
    void argIsWritable() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argIsWritable")).thenReturn(functionPointer);
        when(writerRegistry.find(Writable.class)).thenReturn(Optional.of(DefaultWriters.WritableWriter.INSTANCE));
        KernelFunction kernelFunction = factory.create(getMethod("argIsWritable"));
        assertNotNull(kernelFunction);
        assertEquals(Native.POINTER_SIZE, kernelFunction.nativeArgumentsSize());
        assertEquals(1, kernelFunction.kernelParameterCount());
        KernelArgumentWriter<?> argumentWriter = kernelFunction.argumentWriter(0);
        assertSame(DefaultWriters.WritableWriter.INSTANCE, argumentWriter);
    }

    @Test
    void argHasCustomWriter() {
        Pointer functionPointer = new Pointer(0);
        when(functionPointerResolver.apply("argHasCustomWriter")).thenReturn(functionPointer);
        TestWriter testWriter = new TestWriter();
        when(serviceFactory.getInstance(same(TestWriter.class), any())).thenReturn(testWriter);
        KernelFunction kernelFunction = factory.create(getMethod("argHasCustomWriter"));
        assertNotNull(kernelFunction);
        assertEquals(3 * Long.BYTES, kernelFunction.nativeArgumentsSize());
        assertEquals(1, kernelFunction.kernelParameterCount());
        KernelArgumentWriter<?> argumentWriter = kernelFunction.argumentWriter(0);
        assertSame(testWriter, argumentWriter);
    }

    private static Method getMethod(String name) {
        return Arrays.stream(TestModule.class.getMethods()).filter(m -> m.getName().equals(name)).findFirst().orElseThrow(() -> new NoSuchElementException("Method " + name));
    }

    @SuppressWarnings("unused")
    interface TestModule {
        void notAKernel();

        @Kernel
        int shouldReturnVoid();

        @Kernel(name = "test01", blockDim = {16, 16, 4}, gridDim = {512, 256, 128}, sharedMemory = 42)
        void withGriDim();

        @Kernel(name = "test02", blockDim = {16, 16, 4}, gridThreads = {512, 256, 128})
        void withGridThreads();

        @Kernel(gridDim = {16}, gridThreads = {256})
        void withGriDimAndGridThreads();

        @Kernel
        void argIsNotAGridDim(@GridDim float d);

        @Kernel
        void argIsGridDimX(@BlockDim int blockDimX, @GridDim int gridDimX);

        @Kernel
        void argIsGridDim(@BlockDim dim3 blockDim, @GridDim dim3 gridDim);

        @Kernel
        void argIsGridThreads(@BlockDim dim3 blockDim, @GridThreads dim3 gridThreads);

        @Kernel
        void argIsExposedGridThreads(@BlockDim dim3 blockDim, @GridThreads(exposed = true) dim3 gridThreads);

        @Kernel
        void argIsSharedMem(@SharedMemory int sm);

        @Kernel
        void argIsExposedSharedMem(@SharedMemory(exposed = true) int sm);

        @Kernel
        void argIsWritable(Writable writable);

        @Kernel
        void argHasCustomWriter(@Writer(TestWriter.class) long[] customParam);

    }

    public static final class TestWriter implements KernelArgumentWriter<long[]> {
        @Override
        public long size() {
            return 3 * Long.BYTES;
        }

        @Override
        public void write(long[] arg, Pointer dst) {
            dst.write(0, arg, 0, 3);
        }
    }
}