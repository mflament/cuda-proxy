package org.yah.tools.cuda.support.program;

public class BuildProgramException extends RuntimeException {
    private final String log;

    public BuildProgramException(String log) {
        super("Error building program:\n"+log);
        this.log = log;
    }

    public String getLog() {
        return log;
    }
}
