package org.yah.tools.cuda.proxy.invocation;

import org.yah.tools.cuda.proxy.dim3;

public interface SharedMemorySupplier {

    int get(Object[] args, dim3 blockDim, dim3 gridDim);

}
