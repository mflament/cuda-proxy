package org.yah.tools.cuda.proxy.invocation;

import org.yah.tools.cuda.proxy.dim3;

public interface BlockDimFactory {
    dim3 createBlockDim(Object[] args);
}
