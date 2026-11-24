package data_access;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.zip.CRC32;
import java.util.zip.CheckedOutputStream;

import entities.Model;
import entities.ModelConfig;
import entities.Tokenizer;
import torch.Tensor;
import use_case.save_load_model.ModelWeightsDataAccessInterface;

/**
 * Saves models in the version 1 {@code .tgsw} binary format described in
 * {@code docs/weight-format.md}.
 *
 * <p>This is the only class that knows about the byte layout, string encoding,
 * or checksum. Files are written to a temporary file in the same directory and
 * then moved into place, so a failed save never leaves a partial file at the
 * target path.</p>
 */
public class TgswModelWeightsDataAccessObject implements ModelWeightsDataAccessInterface {

    /** The four ASCII bytes every file starts with: {@code TGSW}. */
    static final byte[] MAGIC = {'T', 'G', 'S', 'W'};
    /** The format version this class writes. */
    static final int VERSION = 1;
    /** Reserved header flags; always 0 in version 1. */
    static final int FLAGS = 0;

    /** Config type tag for a 32-bit signed integer value. */
    static final byte TAG_I32 = 0;
    /** Config type tag for a 32-bit float value. */
    static final byte TAG_F32 = 1;
    /** Config type tag for a UTF-8 string value. */
    static final byte TAG_STR = 2;

    static final String TOKENIZER_CHAR = "char";
    static final String TOKENIZER_BPE = "bpe";

    @Override
    public void save(String path, Model model) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(model, "model");

        // Read everything from the model in one place, so a change to how the
        // model exposes its data only touches these lines.
        ModelConfig config = Objects.requireNonNull(model.config(), "model.config()");
        Tokenizer tokenizer = Objects.requireNonNull(model.tokenizer(), "model.tokenizer()");
        Map<String, Tensor> parameters =
                Objects.requireNonNull(model.parameters(), "model.parameters()");

        List<ConfigEntry> configEntries = configEntries(config);
        validate(config, tokenizer, parameters);

        Path target = Paths.get(path).toAbsolutePath();
        Path temp = Files.createTempFile(target.getParent(), target.getFileName() + ".", ".tmp");
        boolean moved = false;
        try {
            writeFile(temp, configEntries, tokenizer, parameters);
            moveIntoPlace(temp, target);
            moved = true;
        } finally {
            if (!moved) {
                Files.deleteIfExists(temp);
            }
        }
    }

    // ------------------------------------------------------------------
    // Config
    // ------------------------------------------------------------------

    /**
     * Builds the config entries to write, in a fixed order. Adding a new key
     * (for example a model type) only requires a change here.
     */
    private static List<ConfigEntry> configEntries(ModelConfig config) {
        List<ConfigEntry> entries = new ArrayList<>();
        entries.add(ConfigEntry.i32("vocab_size", config.getVocabSize()));
        entries.add(ConfigEntry.i32("context_length", config.getContextLength()));
        entries.add(ConfigEntry.i32("d_model", config.getDModel()));
        entries.add(ConfigEntry.i32("n_layers", config.getNLayers()));
        entries.add(ConfigEntry.i32("n_heads", config.getNHeads()));
        entries.add(ConfigEntry.i32("d_ff", config.getDFf()));

        OptionalDouble learningRate =
                Objects.requireNonNull(config.getLearningRate(), "config.getLearningRate()");
        if (learningRate.isPresent()) {
            entries.add(ConfigEntry.f32("learning_rate", (float) learningRate.getAsDouble()));
        }
        OptionalInt trainSteps =
                Objects.requireNonNull(config.getTrainSteps(), "config.getTrainSteps()");
        if (trainSteps.isPresent()) {
            entries.add(ConfigEntry.i32("train_steps", trainSteps.getAsInt()));
        }
        OptionalDouble finalLoss =
                Objects.requireNonNull(config.getFinalLoss(), "config.getFinalLoss()");
        if (finalLoss.isPresent()) {
            entries.add(ConfigEntry.f32("final_loss", (float) finalLoss.getAsDouble()));
        }
        Optional<String> corpusName =
                Objects.requireNonNull(config.getCorpusName(), "config.getCorpusName()");
        corpusName.ifPresent(name -> entries.add(ConfigEntry.str("corpus_name", name)));
        return entries;
    }

    /** One config key-value pair with its type tag. */
    private static final class ConfigEntry {
        final String key;
        final byte tag;
        final int intValue;
        final float floatValue;
        final String stringValue;

        private ConfigEntry(String key, byte tag, int intValue, float floatValue, String stringValue) {
            this.key = key;
            this.tag = tag;
            this.intValue = intValue;
            this.floatValue = floatValue;
            this.stringValue = stringValue;
        }

        static ConfigEntry i32(String key, int value) {
            return new ConfigEntry(key, TAG_I32, value, 0f, null);
        }

        static ConfigEntry f32(String key, float value) {
            return new ConfigEntry(key, TAG_F32, 0, value, null);
        }

        static ConfigEntry str(String key, String value) {
            return new ConfigEntry(key, TAG_STR, 0, 0f, value);
        }
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /**
     * Rejects models that would produce a file the loader must refuse. Only
     * checks rules that hold for every model architecture; shape rules that
     * depend on the architecture belong to the loader.
     */
    private static void validate(ModelConfig config, Tokenizer tokenizer,
                                 Map<String, Tensor> parameters) {
        String type = tokenizer.type();
        if (!TOKENIZER_CHAR.equals(type) && !TOKENIZER_BPE.equals(type)) {
            throw new IllegalArgumentException("Unknown tokenizer type '" + type + "'");
        }

        List<String> tokens = Objects.requireNonNull(tokenizer.tokens(), "tokenizer.tokens()");
        int vocabSize = config.getVocabSize();
        if (tokens.size() != vocabSize) {
            throw new IllegalArgumentException("Tokenizer has " + tokens.size()
                    + " tokens but vocab_size is " + vocabSize);
        }
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens.get(i) == null) {
                throw new IllegalArgumentException("Token " + i + " is null");
            }
        }

        List<int[]> merges = Objects.requireNonNull(tokenizer.merges(), "tokenizer.merges()");
        if (TOKENIZER_CHAR.equals(type) && !merges.isEmpty()) {
            throw new IllegalArgumentException("A char tokenizer must not have merges");
        }
        for (int i = 0; i < merges.size(); i++) {
            int[] merge = merges.get(i);
            if (merge == null || merge.length != 2) {
                throw new IllegalArgumentException("Merge " + i + " is not a pair of token IDs");
            }
            for (int id : merge) {
                if (id < 0 || id >= vocabSize) {
                    throw new IllegalArgumentException("Merge " + i + " refers to token ID " + id
                            + ", outside [0, " + vocabSize + ")");
                }
            }
        }

        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("Model has no parameters to save");
        }
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            String name = entry.getKey();
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("Tensor name must be non-empty");
            }
            Tensor tensor = entry.getValue();
            if (tensor == null) {
                throw new IllegalArgumentException("Tensor '" + name + "' is null");
            }
            int[] shape = tensor.shape();
            if (shape.length < 1) {
                throw new IllegalArgumentException("Tensor '" + name + "' has rank 0");
            }
            for (int dim : shape) {
                if (dim < 1) {
                    throw new IllegalArgumentException("Tensor '" + name + "' has invalid shape "
                            + Arrays.toString(shape));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Writing
    // ------------------------------------------------------------------

    private static void writeFile(Path file, List<ConfigEntry> configEntries, Tokenizer tokenizer,
                                  Map<String, Tensor> parameters) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(file.toFile())) {
            CheckedOutputStream checked = new CheckedOutputStream(fileOut, new CRC32());
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(checked));

            writeHeader(out);
            writeConfig(out, configEntries);
            writeTokenizer(out, tokenizer);
            writeTensors(out, parameters);

            // The checksum covers every byte before the footer, not the footer itself.
            out.flush();
            int crc = (int) checked.getChecksum().getValue();
            out.writeInt(crc);
            out.flush();
            fileOut.getFD().sync();
        }
    }

    private static void writeHeader(DataOutputStream out) throws IOException {
        out.write(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(FLAGS);
    }

    private static void writeConfig(DataOutputStream out, List<ConfigEntry> entries)
            throws IOException {
        out.writeInt(entries.size());
        for (ConfigEntry entry : entries) {
            writeString(out, entry.key);
            out.writeByte(entry.tag);
            switch (entry.tag) {
                case TAG_I32:
                    out.writeInt(entry.intValue);
                    break;
                case TAG_F32:
                    out.writeFloat(entry.floatValue);
                    break;
                case TAG_STR:
                    writeString(out, entry.stringValue);
                    break;
                default:
                    throw new IllegalStateException("Unknown config tag " + entry.tag);
            }
        }
    }

    private static void writeTokenizer(DataOutputStream out, Tokenizer tokenizer)
            throws IOException {
        String type = tokenizer.type();
        List<String> tokens = tokenizer.tokens();
        writeString(out, type);
        out.writeInt(tokens.size());
        for (String token : tokens) {
            writeString(out, token);
        }
        if (TOKENIZER_BPE.equals(type)) {
            List<int[]> merges = tokenizer.merges();
            out.writeInt(merges.size());
            for (int[] merge : merges) {
                out.writeInt(merge[0]);
                out.writeInt(merge[1]);
            }
        }
    }

    private static void writeTensors(DataOutputStream out, Map<String, Tensor> parameters)
            throws IOException {
        out.writeInt(parameters.size());
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            String name = entry.getKey();
            Tensor tensor = entry.getValue();
            int[] shape = tensor.shape();
            float[] data = tensor.toArray();

            long expected = 1;
            for (int dim : shape) {
                expected *= dim;
            }
            if (data.length != expected) {
                throw new IllegalStateException("Tensor '" + name + "' has " + data.length
                        + " elements but its shape needs " + expected);
            }

            writeString(out, name);
            out.writeInt(shape.length);
            for (int dim : shape) {
                out.writeInt(dim);
            }
            for (float value : data) {
                out.writeFloat(value);
            }
        }
    }

    /**
     * Writes a length-prefixed UTF-8 string. Deliberately avoids
     * {@link DataOutputStream#writeUTF}, which uses modified UTF-8 and caps
     * strings at 65,535 bytes.
     */
    private static void writeString(DataOutputStream out, String s) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    // ------------------------------------------------------------------
    // File replacement
    // ------------------------------------------------------------------

    private static void moveIntoPlace(Path temp, Path target) throws IOException {
        try {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
