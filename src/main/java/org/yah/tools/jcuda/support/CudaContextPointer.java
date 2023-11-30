package org.yah.tools.jcuda.support;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import static org.yah.tools.jcuda.support.DriverSupport.*;

public class CudaContextPointer extends Pointer {
    public CudaContextPointer(PointerByReference ref) {
        super(Pointer.nativeValue(ref.getValue()));
    }

    public void setCurrent() {
        check(driverAPI().cuCtxSetCurrent(this));
    }

    public void destroy() {
        check(driverAPI().cuCtxDestroy(this));
    }

    public static CudaContextPointer getCurrent() {
        PointerByReference ptrRef = new PointerByReference();
        check(driverAPI().cuCtxGetCurrent(ptrRef));
        return new CudaContextPointer(ptrRef);
    }

    public static void synchronize() {
        check(driverAPI().cuCtxSynchronize());
    }
}
