package org.yah.tools.jcuda.support.module;

import javax.annotation.Nonnull;

public class DefaultServiceFactory implements ServiceFactory {
    @Nonnull
    @Override
    public <T> T getInstance(Class<T> type) {
        try {
            return type.getConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
