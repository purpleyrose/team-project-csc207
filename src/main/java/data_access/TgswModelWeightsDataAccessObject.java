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
 * or checksum. The model is read exactly once into a snapshot, which is
 * validated and then written, so the file always matches what was checked even
 * if the model changes during the save. Files are written to a temporary file
 * in the same directory and then moved into place, so a failed save never
 * leaves a partial file at the target path.</p>
 *
 * <p>The optional {@code learning_rate} and {@code final_loss} values are
 * stored as 32-bit floats, as the format requires, so they lose precision
 * (very small values may become 0 and very large ones infinity). They are for
 * display only.</p>
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

        Snapshot snapshot = Snapshot.of(model);

        Path target = Paths.get(path).toAbsolutePath();
        if (Files.isDirectory(target)) {
            throw new IOException("Cannot save model: '" + target + "' is a directory");
        }
        Path parent = target.getParent();
        if (parent == null) {
            throw new IOException("Cannot save model: '" + target + "' has no parent directory");
        }

        Path temp = Files.createTempFile(parent, target.getFileName() + ".", ".tmp");
        boolean moved = false;
        try {
            writeFile(temp, snapshot);
            moveIntoPlace(temp, target);
            moved = true;
        } finally {
            if (!moved) {
                Files.deleteIfExists(temp);
            }
        }
    }

    // ------------------------------------------------------------------
    // Snapshot: read the model once, copy, and validate
    // ------------------------------------------------------------------

    /**
     * Everything to be written, read from the model exactly once and already
     * validated. Only rules that hold for every model architecture are checked;
     * shape rules that depend on the architecture belong to the loader.
     */
    private static final class Snapshot {
        final List<ConfigEntry> config;
        final String tokenizerType;
        final List<String> tokens;
        final List<int[]> merges;
        final List<NamedTensor> tensors;

        private Snapshot(List<ConfigEntry> config, String tokenizerType, List<String> tokens,
                         List<int[]> merges, List<NamedTensor> tensors) {
            this.config = config;
            this.tokenizerType = tokenizerType;
            this.tokens = tokens;
            this.merges = merges;
            this.tensors = tensors;
        }

        /**
         * Reads and validates a model. This is the only place that calls the
         * model, so a change to how the model exposes its data only touches here.
         */
        static Snapshot of(Model model) {
            ModelConfig config = Objects.requireNonNull(model.config(), "model.config()");
            Tokenizer tokenizer = Objects.requireNonNull(model.tokenizer(), "model.tokenizer()");
            Map<String, Tensor> parameters =
                    Objects.requireNonNull(model.parameters(), "model.parameters()");

            List<ConfigEntry> configEntries = configEntries(config);
            int vocabSize = config.getVocabSize();

            String type = tokenizer.type();
            if (!TOKENIZER_CHAR.equals(type) && !TOKENIZER_BPE.equals(type)) {
                throw new IllegalArgumentException("Unknown tokenizer type '" + type + "'");
            }

            List<String> tokens = copyTokens(tokenizer, vocabSize);
            List<int[]> merges = copyMerges(tokenizer, type, vocabSize);
            List<NamedTensor> tensors = copyTensors(parameters);
            return new Snapshot(configEntries, type, tokens, merges, tensors);
        }

        private static List<String> copyTokens(Tokenizer tokenizer, int vocabSize) {
            List<String> tokens = new ArrayList<>(
                    Objects.requireNonNull(tokenizer.tokens(), "tokenizer.tokens()"));
            if (tokens.size() != vocabSize) {
                throw new IllegalArgumentException("Tokenizer has " + tokens.size()
                        + " tokens but vocab_size is " + vocabSize);
            }
            for (int i = 0; i < tokens.size(); i++) {
                String token = tokens.get(i);
                if (token == null) {
                    throw new IllegalArgumentException("Token " + i + " is null");
                }
                requireWellFormed(token, "Token " + i);
            }
            return tokens;
        }

        private static List<int[]> copyMerges(Tokenizer tokenizer, String type, int vocabSize) {
            List<int[]> source = Objects.requireNonNull(tokenizer.merges(), "tokenizer.merges()");
            if (TOKENIZER_CHAR.equals(type) && !source.isEmpty()) {
                throw new IllegalArgumentException("A char tokenizer must not have merges");
            }
            List<int[]> merges = new ArrayList<>(source.size());
            for (int i = 0; i < source.size(); i++) {
                int[] merge = source.get(i);
                if (merge == null || merge.length != 2) {
                    throw new IllegalArgumentException("Merge " + i + " is not a pair of token IDs");
                }
                int[] copy = merge.clone();
                for (int id : copy) {
                    if (id < 0 || id >= vocabSize) {
                        throw new IllegalArgumentException("Merge " + i + " refers to token ID " + id
                                + ", outside [0, " + vocabSize + ")");
                    }
                }
                merges.add(copy);
            }
            return merges;
        }

        private static List<NamedTensor> copyTensors(Map<String, Tensor> parameters) {
            // Copy the entries first so the map is iterated exactly once.
            List<Map.Entry<String, Tensor>> entries = new ArrayList<>(parameters.entrySet());
            if (entries.isEmpty()) {
                throw new IllegalArgumentException("Model has no parameters to save");
            }
            List<NamedTensor> tensors = new ArrayList<>(entries.size());
            for (Map.Entry<String, Tensor> entry : entries) {
                String name = entry.getKey();
                if (name == null || name.isEmpty()) {
                    throw new IllegalArgumentException("Tensor name must be non-empty");
                }
                requireWellFormed(name, "Tensor name '" + name + "'");
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
                tensors.add(new NamedTensor(name, shape, tensor));
            }
            return tensors;
        }
    }

    /** A parameter with its name and the shape captured when it was validated. */
    private static final class NamedTensor {
        final String name;
        final int[] shape;
        final Tensor tensor;

        NamedTensor(String name, int[] shape, Tensor tensor) {
            this.name = name;
            this.shape = shape;
            this.tensor = tensor;
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
        entries.add(requiredInt("vocab_size", config.getVocabSize()));
        entries.add(requiredInt("context_length", config.getContextLength()));
        entries.add(requiredInt("d_model", config.getDModel()));
        entries.add(requiredInt("n_layers", config.getNLayers()));
        entries.add(requiredInt("n_heads", config.getNHeads()));
        entries.add(requiredInt("d_ff", config.getDFf()));

        OptionalDouble learningRate =
                Objects.requireNonNull(config.getLearningRate(), "config.getLearningRate()");
        if (learningRate.isPresent()) {
            entries.add(ConfigEntry.f32("learning_rate", (float) learningRate.getAsDouble()));
        }
        OptionalInt trainSteps =
                Objects.requireNonNull(config.getTrainSteps(), "config.getTrainSteps()");
        if (trainSteps.isPresent()) {
            entries.add(requiredInt("train_steps", trainSteps.getAsInt()));
        }
        OptionalDouble finalLoss =
                Objects.requireNonNull(config.getFinalLoss(), "config.getFinalLoss()");
        if (finalLoss.isPresent()) {
            entries.add(ConfigEntry.f32("final_loss", (float) finalLoss.getAsDouble()));
        }
        Optional<String> corpusName =
                Objects.requireNonNull(config.getCorpusName(), "config.getCorpusName()");
        if (corpusName.isPresent()) {
            requireWellFormed(corpusName.get(), "Config value 'corpus_name'");
            entries.add(ConfigEntry.str("corpus_name", corpusName.get()));
        }
        return entries;
    }

    /**
     * Creates an integer entry, rejecting negative values. Zero is allowed
     * because simpler architectures may not use every size (for example, a
     * bigram model has no attention heads).
     */
    private static ConfigEntry requiredInt(String key, int value) {
        if (value < 0) {
            throw new IllegalArgumentException("Config value '" + key
                    + "' must not be negative, got " + value);
        }
        return ConfigEntry.i32(key, value);
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
    // Strings
    // ------------------------------------------------------------------

    /**
     * Rejects strings containing an unpaired UTF-16 surrogate. Java's UTF-8
     * encoder silently replaces those with {@code '?'}, which would corrupt the
     * saved vocabulary (for example, a tokenizer that splits an emoji with
     * {@code charAt} would save both halves as the same token).
     */
    private static void requireWellFormed(String s, String what) {
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (i + 1 < s.length() && Character.isLowSurrogate(s.charAt(i + 1))) {
                    i++;
                    continue;
                }
                throw unpairedSurrogate(what, i);
            }
            if (Character.isLowSurrogate(c)) {
                throw unpairedSurrogate(what, i);
            }
        }
    }

    private static IllegalArgumentException unpairedSurrogate(String what, int index) {
        return new IllegalArgumentException(what
                + " is not valid Unicode (unpaired surrogate at index " + index + ")");
    }

    // ------------------------------------------------------------------
    // Writing
    // ------------------------------------------------------------------

    private static void writeFile(Path file, Snapshot snapshot) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(file.toFile())) {
            CheckedOutputStream checked = new CheckedOutputStream(fileOut, new CRC32());
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(checked));

            writeHeader(out);
            writeConfig(out, snapshot.config);
            writeTokenizer(out, snapshot);
            writeTensors(out, snapshot.tensors);

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

    private static void writeTokenizer(DataOutputStream out, Snapshot snapshot) throws IOException {
        writeString(out, snapshot.tokenizerType);
        out.writeInt(snapshot.tokens.size());
        for (String token : snapshot.tokens) {
            writeString(out, token);
        }
        if (TOKENIZER_BPE.equals(snapshot.tokenizerType)) {
            out.writeInt(snapshot.merges.size());
            for (int[] merge : snapshot.merges) {
                out.writeInt(merge[0]);
                out.writeInt(merge[1]);
            }
        }
    }

    private static void writeTensors(DataOutputStream out, List<NamedTensor> tensors)
            throws IOException {
        out.writeInt(tensors.size());
        for (NamedTensor t : tensors) {
            float[] data = t.tensor.toArray();

            long expected = 1;
            for (int dim : t.shape) {
                expected *= dim;
            }
            if (data.length != expected) {
                throw new IllegalStateException("Tensor '" + t.name + "' has " + data.length
                        + " elements but its shape needs " + expected);
            }

            writeString(out, t.name);
            out.writeInt(t.shape.length);
            for (int dim : t.shape) {
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
