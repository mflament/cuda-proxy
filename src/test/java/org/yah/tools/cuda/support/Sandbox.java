package org.yah.tools.cuda.support;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.support.program.CudaProgramBuilder;
import org.yah.tools.cuda.support.program.CudaProgramPointer;

import static org.yah.tools.cuda.TestsHelper.loadSource;
import static org.yah.tools.cuda.support.DriverSupport.*;
import static org.yah.tools.cuda.support.NTSHelper.*;

public class Sandbox {

    public static void main(String[] args) {
        String src = loadSource("nvrtcTest.cu");
        CudaProgramPointer program = CudaProgramBuilder.create(src).build();
        Memory ptx = program.getPTX();

        PointerByReference ptrRef = new PointerByReference();
        check(driverAPI().cuLibraryLoadData(ptrRef, ptx, null, null, 0, null , null, 0));
        Pointer library = ptrRef.getValue();
        check(driverAPI().cuLibraryGetKernel(ptrRef, library, "matSum"));
        Pointer kernel = ptrRef.getValue();
        Memory nameBuffer = new Memory(256) ;
        check(driverAPI().cuKernelGetName(nameBuffer, kernel));
        String name = readNTS(nameBuffer, nameBuffer.size());
        System.out.println(name);

//        System.out.println(NTSHelper.readNTS(ptx, ptx.size()));
        ptx.close();
    }
}
