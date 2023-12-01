package org.yah.tools.jcuda.support.module;

public interface Module extends AutoCloseable {
    /**
     * Close the cuModule
     */
    @Override
    void close();
}
