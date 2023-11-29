package org.yah.tools.jcuda.support;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

public final class TestsHelper {
    private TestsHelper() {
    }

    public static String loadSource(String name) {
        try (InputStream is = NVRTCSupport.class.getClassLoader().getResourceAsStream(name)) {
            if (is == null)
                throw new IOException("Resource " + name + " not found");
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
