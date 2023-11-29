package org.yah.tools.jcuda.support.module;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.junit.jupiter.api.Test;
import org.yah.tools.jcuda.support.CudaContextPointer;
import org.yah.tools.jcuda.support.DriverSupport;
import org.yah.tools.jcuda.support.TestsHelper;
import org.yah.tools.jcuda.support.module.annotations.GridDim;
import org.yah.tools.jcuda.support.module.annotations.Kernel;
import org.yah.tools.jcuda.support.program.CudaProgramBuilder;
import org.yah.tools.jcuda.support.program.CudaProgramPointer;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.yah.tools.jcuda.support.DriverSupport.check;
import static org.yah.tools.jcuda.support.DriverSupport.driverAPI;

class CudaModuleBuilderTest {

    private final Random random = new Random(12345);

    @Test
    void createModule() {
        CudaModuleBuilder cudaModuleBuilder = new CudaModuleBuilder();
        String src = TestsHelper.loadSource("module_build_test1.cu");
        CudaContextPointer context = DriverSupport.createContext(0, 0);
        CudaProgramPointer program = CudaProgramBuilder.create(src).withProgramName("test_program").build();
        TestModule module = cudaModuleBuilder.createModule(program, TestModule.class);

        context.setCurrent();
        int N = 512;
        int[] a = randomInts(N), b = randomInts(N);
        try (Memory hostMemory = new Memory(N * (long) Integer.BYTES)) {
            Pointer aPtr = copyToDevice(a, hostMemory), bPtr = copyToDevice(b, hostMemory), cPtr = allocateInts(N);
            module.sum(Dim.roundup(N, TestModule.BLOCK_DIM), N, aPtr, bPtr, cPtr);

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

    public interface TestModule {
        int BLOCK_DIM = 1024;

        @Kernel(blockDim = {BLOCK_DIM, 1, 1})
        void sum(@GridDim(dim = DimName.x, exposed = false) int gridDimX, int N, Pointer a, Pointer b, Pointer c);

        @Kernel(name = "sum", blockDim = {BLOCK_DIM, 1, 1})
        void sum2(@GridDim(dim = DimName.x, converter = DeviceIntArrayGridDimConverter.class) DeviceIntArray a, Pointer b, Pointer c);
    }

    public static final class DeviceIntArrayGridDimConverter implements GridDimConverter<DeviceIntArray>, GridDimsConverter<DeviceIntArray> {

        @Override
        public int toDim(DeviceIntArray arg, Dim blockDim) {
            return Dim.roundup(arg.length, blockDim.x());
        }

        @Override
        public Dim toGridDim(DeviceIntArray arg, Dim blockDim) {
            return Dim.createGrid(new Dim(arg.length, 1, 1), blockDim);
        }
    }

    public static final class DeviceIntArray extends Pointer {
        private final int length;

        public DeviceIntArray(long peer, int length) {
            super(peer);
            this.length = length;
        }

        public int length() {
            return length;
        }

        public long size() {
            return length * (long) Integer.BYTES;
        }
    }
}