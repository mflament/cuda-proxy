package org.yah.tools.jcuda.support.module.invocation;

import org.yah.tools.jcuda.support.module.dim3;

public interface SharedMemorySupplier {

    int get(Object[] args, dim3 blockDim, dim3 gridDim);

}
