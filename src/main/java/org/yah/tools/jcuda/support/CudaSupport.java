package org.yah.tools.jcuda.support;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import org.yah.tools.jcuda.jna.DriverAPI;
import org.yah.tools.jcuda.jna.NVRTC;
import org.yah.tools.jcuda.jna.RuntimeAPI;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Helper for Cuda native libraries
 */
public final class CudaSupport {
    private static final String DRIVER_API_LIBRARY_PROPERTY = "jcuda.driverApi";
    private static final String RUNTIME_API_LIBRARY_PROPERTY = "jcuda.runtimeApi";
    private static final String NVRTC_LIBRARY_PROPERTY = "jcuda.nvrtc";

    private CudaSupport() {
    }

    public static DriverAPI createDriverAPI() {
        return loadLibrary(DRIVER_API_LIBRARY_PROPERTY, () -> "nvcuda", DriverAPI.class);
    }

    public static RuntimeAPI createRuntimeApi() {
        return loadLibrary(RUNTIME_API_LIBRARY_PROPERTY, CudaSupport::getRuntimeLibraryName, RuntimeAPI.class);
    }

    public static NVRTC createNVRTC() {
        return loadLibrary(NVRTC_LIBRARY_PROPERTY, CudaSupport::getNVRTCLibraryName, NVRTC.class);
    }

    private static <T extends Library> T loadLibrary(String propertyKey, Supplier<String> defaultNameSupplier, Class<T> libraryClass) {
        String name = System.getProperty(propertyKey);
        if (name == null)
            name = defaultNameSupplier.get();
        return Native.load(name, libraryClass);
    }

    private static String getRuntimeLibraryName() {
        String namePrefix;
        if (Platform.isWindows())
            namePrefix = "cudart64";
        else if (Platform.isLinux())
            namePrefix = "libcudart";
        else
            throw new UnsupportedOperationException("Unsupported platform " + System.getProperty("os.name"));
        Path path = findLibraryFile(namePrefix);
        return path.toAbsolutePath().toString();
    }

    private static String getNVRTCLibraryName() {
        String namePrefix;
        if (Platform.isWindows())
            namePrefix = "nvrtc64";
        else if (Platform.isLinux())
            namePrefix = "libnvrtc";
        else
            throw new UnsupportedOperationException("Unsupported platform " + System.getProperty("os.name"));
        Path path = findLibraryFile(namePrefix);
        return path.toAbsolutePath().toString();
    }

    private static Path findLibraryFile(String name) {
        Path cudaPath = getCudaPath();
        Path libraryPath;
        if (!Platform.is64Bit())
            throw new IllegalStateException("Only 64bits platform are supported");

        if (Platform.isWindows())
            libraryPath = cudaPath.resolve("bin");
        else if (Platform.isLinux())
            libraryPath = cudaPath.resolve("lib64");
        else
            throw new IllegalStateException("Unsupported platform " + System.getProperty("os.name"));

        try (Stream<Path> stream = Files.list(libraryPath)) {
            return stream.filter(createLibraryPredicate(name)).findFirst().orElseThrow(() -> new IllegalStateException("Cuda library " + name + " not found in " + libraryPath));
        } catch (IOException e) {
            throw new IllegalStateException("Error listing library files in " + libraryPath, e);
        }
    }

    private static Predicate<Path> createLibraryPredicate(String name) {
        String ext = Platform.isWindows() ? ".dll" : ".so";
        return path -> {
            String fileName = path.getFileName().toString();
            return fileName.startsWith(name) && fileName.endsWith(ext);
        };
    }

    private static Path getCudaPath() {
        String cudaPathVar = System.getenv("CUDA_PATH");
        if (cudaPathVar == null)
            throw new IllegalStateException("CUDA_PATH environment variable not found");
        Path cudaPath = Path.of(cudaPathVar);
        if (!Files.isDirectory(cudaPath))
            throw new IllegalStateException("CUDA_PATH " + cudaPathVar + " directory not found");
        return cudaPath;
    }
}
