package org.yah.tools.cuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.Test;
import org.yah.tools.cuda.support.NTSHelper;

import static org.junit.jupiter.api.Assertions.*;
import static org.yah.tools.cuda.TestsHelper.loadSource;

class CudaProgramBuilderTest {

    @Test
    void createAndCompileSucceed() {
        String src = loadSource("nvrtcTest.cu");
        CudaProgramPointer program = CudaProgramBuilder.create(src).build();
        assertNotNull(program);
        assertNotEquals(0, Pointer.nativeValue(program));
        try (Memory ptx = program.getPTX()){
            assertNotNull(ptx);
            assertEquals(program.getPTXSize(), ptx.size());
            System.out.println(NTSHelper.readNTS(ptx, ptx.size()));
        }
        program.destroy();
    }

    @Test
    void createAndCompileWithError() {
        String src = loadSource("nvrtcTest_error.cu");
        try {
            CudaProgramBuilder.create(src).withProgramName("test_program").build();
            fail("Should have thrown BuildProgramException");
        } catch (BuildProgramException e) {
            assertTrue(e.getLog().contains("1 error detected in the compilation of \"test_program\"."));
        }
    }

}