package org.yah.tools.jcuda.support.module.services;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;

public class DefaultServiceFactory implements ServiceFactory {

    @Nonnull
    @Override
    public <T> T getInstance(Class<T> type, Parameter parameter) {
        try {
            Constructor<T> constructor = getConstructorFromParameter(type);
            if (constructor != null) {
                return constructor.newInstance(parameter);
            } else {
                return type.getConstructor().newInstance();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Nullable
    private static <T> Constructor<T> getConstructorFromParameter(Class<T> type) {
        try {
            return type.getConstructor(Parameter.class);
        } catch (NoSuchMethodException e) {
            return null;
        }
    }
}
