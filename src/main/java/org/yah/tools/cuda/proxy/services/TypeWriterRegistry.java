package org.yah.tools.cuda.proxy.services;

import org.yah.tools.cuda.proxy.annotations.Writer;

import java.util.Optional;

/**
 * Provide {@link Writer} per java type.
 * <p>
 * Used to expose {@link DefaultWriters} and user custom mappers.
 */
public interface TypeWriterRegistry {

    <T> void register(Class<T> type, KernelArgumentWriter<T> writer);

    <T> Optional<KernelArgumentWriter<T>> find(Class<T> type);

}
