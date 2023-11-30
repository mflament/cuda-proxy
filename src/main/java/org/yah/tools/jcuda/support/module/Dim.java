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

    @Override
    public String toString() {
        return "{" +
                "x=" + x +
                ", y=" + y +
                ", z=" + z +
                '}';
    }

    public static Dim createGrid(Dim gridThreads, Dim blockDim) {
        return new Dim(blocks(gridThreads.x, blockDim.x), blocks(gridThreads.y, blockDim.y), blocks(gridThreads.z, blockDim.z));
    }

    public static int blocks(int threads, int blockDim) {
        return (int) Math.ceil(threads / (double) blockDim);
    }
}
