package org.yah.tools.cuda.proxy;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.junit.jupiter.api.Test;
import org.yah.tools.cuda.TestsHelper;
import org.yah.tools.cuda.proxy.annotations.GridDim;
import org.yah.tools.cuda.proxy.annotations.GridThreads;
import org.yah.tools.cuda.proxy.annotations.Kernel;
import org.yah.tools.cuda.proxy.services.Writable;
import org.yah.tools.cuda.support.CudaContextPointer;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.device.DevicePointer;
import org.yah.tools.cuda.support.program.CudaProgramBuilder;
import org.yah.tools.cuda.support.program.CudaProgramPointer;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.yah.tools.cuda.support.DriverSupport.*;

class KernelProxyFactoryTest {

    private final Random random = new Random(12345);

    @Test
    void createModule() {
        String src = TestsHelper.loadSource("module_build_test1.cu");

        DevicePointer device = DriverSupport.getDevice(0);
        CudaContextPointer ctx = device.createContext(0);
        ctx.setCurrent();

        CudaProgramPointer program = CudaProgramBuilder.create(src)
                .withProgramName("test_program")
                .withComputeVersion(device)
                .build();

        KernelProxyFactory kernelProxyFactory = new KernelProxyFactory();
        int N = 512;
        int[] a = randomInts(N), b = randomInts(N);
        try (Memory hostMemory = new Memory(N * (long) Integer.BYTES);
             TestModule module = kernelProxyFactory.createKernelProxy(program, TestModule.class)) {
            Pointer aPtr = copyToDevice(a, hostMemory), bPtr = copyToDevice(b, hostMemory), cPtr = allocateInts(N);
            module.sum(dim3.fromThreads(new dim3(N), new dim3(TestModule.BLOCK_DIM)), N, aPtr, bPtr, cPtr);

            check(driverAPI().cuMemcpyDtoH(hostMemory, cPtr, N * (long) Integer.BYTES));
            for (int i = 0; i < N; i++) {
                int expected = a[i] + b[i];
                int actual = hostMemory.getInt(i * Integer.BYTES);
                assertEquals(expected, actual);
            }
            synchronizeContext();

            DeviceIntArray aIntArray = new DeviceIntArray(aPtr);
            DeviceIntArray bIntArray = new DeviceIntArray(bPtr);
            DeviceIntArray cIntArray = new DeviceIntArray(cPtr);
            module.sum2(N, aIntArray, bIntArray, cIntArray);
            synchronizeContext();

            check(driverAPI().cuMemcpyDtoH(hostMemory, cPtr, N * (long) Integer.BYTES));
            for (int i = 0; i < N; i++) {
                int expected = a[i] + b[i];
                int actual = hostMemory.getInt(i * Integer.BYTES);
                assertEquals(expected, actual);
            }
        }
    }

    private Pointer copyToDevice(int[] a, Memory hostMemory) {
        hostMemory.write(0, a, 0, a.length);
        Pointer ptr = allocateInts(a.length);
        check(driverAPI().cuMemcpyHtoD(ptr, hostMemory, a.length * (long) Integer.BYTES));
        return ptr;
    }

    private Pointer allocateInts(int n) {
        PointerByReference ptr = new PointerByReference();
        check(driverAPI().cuMemAlloc(ptr, n * (long) Integer.BYTES));
        return ptr.getValue();
    }

    private int[] randomInts(int count) {
        int[] ints = new int[count];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = random.nextInt(1000);
        }
        return ints;
    }

    public interface TestModule extends AutoCloseable {
        int BLOCK_DIM = 1024;

        @Kernel(blockDim = {BLOCK_DIM, 1, 1})
        void sum(@GridDim dim3 gridDim, int N, Pointer a, Pointer b, Pointer c);

        @Kernel(name = "sum", blockDim = {BLOCK_DIM, 1, 1})
        void sum2(@GridThreads(exposed = true) int N, DeviceIntArray a, DeviceIntArray b, DeviceIntArray c);

        @Override
        void close();
    }

    public static final class DeviceIntArray implements Writable {
        private final Pointer pointer;

        public DeviceIntArray(Pointer pointer) {
            this.pointer = pointer;
        }

        @Override
        public Pointer pointer() {
            return pointer;
        }
    }
}