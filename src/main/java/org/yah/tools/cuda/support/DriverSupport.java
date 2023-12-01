package org.yah.tools.cuda.support;

import com.sun.jna.ptr.PointerByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.api.Driver;
import org.yah.tools.cuda.support.device.DevicePointer;

import javax.annotation.Nullable;

import static org.yah.tools.cuda.support.NTSHelper.readNTS;

public class DriverSupport {

    private static final Logger LOGGER = LoggerFactory.getLogger(DriverSupport.class);

    @Nullable
    private static Driver driver;

    public static synchronized Driver driverAPI() {
        if (driver == null) {
            driver = CudaSupport.createDriverAPI();
            int error = driverAPI().cuInit(0);
            if (error != 0)
                throw new CudaException("nvcuda", error, "?");
        }
        return driver;
    }

    public static synchronized DevicePointer getDevice(int ordinal) {
        PointerByReference ptrRef = new PointerByReference();
        check(driverAPI().cuDeviceGet(ptrRef, ordinal));
        DevicePointer devicePointer = new DevicePointer(ptrRef);
        if (LOGGER.isInfoEnabled()) {
            LOGGER.info("Using device {} : {}", ordinal, devicePointer.getDeviceName());
        }
        return devicePointer;
    }

    public static void check(int status) {
        if (status != 0) {
            PointerByReference strRef = new PointerByReference();
            driverAPI().cuGetErrorName(status, strRef);
            String error = readNTS(strRef.getValue(), 256);
            throw new CudaException("nvcuda", status, error);
        }
    }

}
