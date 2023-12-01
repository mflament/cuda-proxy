package org.yah.tools.jcuda.support.module.services;

import com.sun.jna.Pointer;

/**
 * Used to get a device pointer from a java object argument.
 * The implementation must allocate and copy the device memory.
 */
public interface Writable {
    /**
     * @return the device pointer address to pass as parameter
     */
    Pointer pointer();
}
