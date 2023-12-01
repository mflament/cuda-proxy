package org.yah.tools.jcuda.support.module.services;

import javax.annotation.Nonnull;
import java.lang.reflect.Parameter;

public interface ServiceFactory {

    /**
     * Get an instance of service S for a given parameter.
     *
     * @param parameter the parameter this service should handle
     * @param <T>       Any service that will work on parameter value during kernel invocation
     * @return The service to handle the parameter
     */
    @Nonnull
    <T> T getInstance(Class<T> type, Parameter parameter);

}
