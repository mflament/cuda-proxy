package org.yah.tools.cuda.proxy;

public interface Module extends AutoCloseable {
    /**
     * Close the cuModule
     */
    @Override
    void close();
}
