package org.yah.tools.jcuda.support.module.services;

import javax.annotation.Nullable;
import java.util.LinkedList;
import java.util.Optional;

/**
 * A {@link TypeWriterRegistry} using a {@link java.util.LinkedList} of {@link WriterEntry} to resolve a writer from a type.
 * Writer registration defines the resolution order. Last registered type override previous registered writer.
 */
public class ListTypeWriterRegistry implements TypeWriterRegistry {

    public static TypeWriterRegistry create(boolean addDefaults) {
        ListTypeWriterRegistry registry = new ListTypeWriterRegistry();
        if (addDefaults)
            DefaultWriters.register(registry);
        return registry;
    }

    private final LinkedList<WriterEntry<?>> entries = new LinkedList<>();

    public ListTypeWriterRegistry() {
    }

    @Override
    public <T> void register(Class<T> type, KernelArgumentWriter<T> writer) {
        entries.addFirst(new WriterEntry<>(type, writer));
    }

    @Nullable
    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<KernelArgumentWriter<T>> find(Class<T> type) {
        return entries.stream().filter(e -> e.accept(type))
                .findFirst().map(e -> (KernelArgumentWriter<T>) e.writer());
    }

    public static class WriterEntry<T> {
        private final Class<T> type;
        private final KernelArgumentWriter<T> writer;

        public WriterEntry(Class<T> type, KernelArgumentWriter<T> writer) {
            this.type = type;
            this.writer = writer;
        }

        boolean accept(Class<?> parameterType) {
            return type.isAssignableFrom(parameterType);
        }

        public KernelArgumentWriter<T> writer() {
            return writer;
        }
    }

}
