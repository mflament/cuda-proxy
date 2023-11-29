package org.yah.tools.jcuda.support.module;

public interface GridDimConverter<T> {
    // return an unsigned int
    int toDim(T arg, Dim blockDim);
}
