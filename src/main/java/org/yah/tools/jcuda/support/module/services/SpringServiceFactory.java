package org.yah.tools.jcuda.support.module.services;

import org.springframework.beans.factory.BeanDefinitionStoreException;
import org.springframework.context.ApplicationContext;

import javax.annotation.Nonnull;
import java.lang.reflect.Parameter;
import java.util.Objects;

/**
 * Implementation of {@link ServiceFactory} using spring context. Spring dependency must be provided by the caller.
 */
public class SpringServiceFactory implements ServiceFactory {

    private final ApplicationContext applicationContext;

    public SpringServiceFactory(ApplicationContext applicationContext) {
        this.applicationContext = Objects.requireNonNull(applicationContext, "applicationContext is null");
    }

    @Nonnull
    @Override
    public <T> T getInstance(Class<T> type, Parameter parameter) {
        try {
            return applicationContext.getBean(type, parameter);
        } catch (BeanDefinitionStoreException e) {
            return applicationContext.getBean(type);
        }
    }

}
