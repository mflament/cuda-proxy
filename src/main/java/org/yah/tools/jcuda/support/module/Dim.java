package org.yah.tools.jcuda.support.module;

public class Dim {
    private final int x;
    private final int y;
    private final int z;

    public Dim(int[] dims) {
        this(dims.length > 0 ? dims[0] : 1, dims.length > 1 ? dims[1] : 1, dims.length > 2 ? dims[2] : 1);
    }

    public Dim(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public int x() {
        return x;
    }

    public int y() {
        return y;
    }

    public int z() {
        return z;
    }


    public static Dim createGrid(Dim gridThreads, Dim blockDim) {
        return new Dim(roundup(gridThreads.x, blockDim.x), roundup(gridThreads.y, blockDim.y), roundup(gridThreads.z, blockDim.z));
    }

    public static int roundup(int a, int b) {
        return (int) Math.ceil(a / (double) b);
    }
}
