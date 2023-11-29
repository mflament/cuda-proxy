package org.yah.tools.jcuda.support.module;

import com.sun.jna.Pointer;

public interface KernelArgumentWriter<T> {

    /**
     *
     * @param arg the parameter to write
     * @param dst the destination pointer
     * @return the size of this parameter
     */
    long write(T arg, Pointer dst);

}
