package org.yah.tools.cuda.support;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import static org.yah.tools.cuda.support.DriverSupport.*;

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

}
