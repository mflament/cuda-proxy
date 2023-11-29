package org.yah.tools.jcuda.support.module;

import javax.annotation.Nonnull;

public interface ServiceFactory {
    @Nonnull
    <T> T getInstance(Class<T> type);
}
