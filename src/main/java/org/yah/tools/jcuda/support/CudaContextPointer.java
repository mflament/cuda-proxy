package org.yah.tools.jcuda.support;

import com.sun.jna.ptr.PointerByReference;

public class CudaContextPointer extends PointerByReference  {
    public CudaContextPointer() {
    }

    public void setCurrent() {
        DriverSupport.check(DriverSupport.driverAPI().cuCtxSetCurrent(getValue()));
    }

    public void destroy() {
        DriverSupport.check(DriverSupport.driverAPI().cuCtxDestroy(getValue()));
    }
}
