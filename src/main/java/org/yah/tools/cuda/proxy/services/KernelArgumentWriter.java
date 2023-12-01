package org.yah.tools.cuda.proxy.services;

import com.sun.jna.Pointer;

/**
 * Write an argument to native memory.
 * @param <T> the argument type handled by this writer.
 */
public interface KernelArgumentWriter<T> {

    /**
     * @return the size (in bytes) of the native memory representing this argument. Size of kernel argument are always
     * constant (dynamic memory is a {@link Pointer}).
     */
    long size();

    /**
     * @param arg the parameter to write
     * @param dst the destination pointer
     */
    void write(T arg, Pointer dst);

}
