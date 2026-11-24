package data_access;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.CRC32;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import entities.Model;
import entities.ModelConfig;
import entities.Tokenizer;
import torch.NaiveTensorFactory;
import torch.Tensor;
import torch.TensorFactory;

/**
 * Tests for {@link TgswModelWeightsDataAccessObject}. Saved files are decoded
 * with a small reader written from {@code docs/weight-format.md}, independent
 * of the class under test.
 */
class TgswModelWeightsDataAccessObjectTest {

    private static final TensorFactory TENSORS = new NaiveTensorFactory();

    @TempDir
    Path dir;

    private final TgswModelWeightsDataAccessObject dao = new TgswModelWeightsDataAccessObject();

    // ------------------------------------------------------------------
    // Header and footer
    // ------------------------------------------------------------------

    @Test
    void fileStartsWithExactHeaderBytes() throws IOException {
        byte[] bytes = saveAndRead(sampleModel());

        byte[] expected = {
                0x54, 0x47, 0x53, 0x57,   // "TGSW"
                0x00, 0x00, 0x00, 0x01,   // version 1
                0x00, 0x00, 0x00, 0x00    // flags 0
        };
        assertArrayEquals(expected, Arrays.copyOf(bytes, 12));
    }

    @Test
    void footerIsCrc32OfEverythingBeforeIt() throws IOException {
        byte[] bytes = saveAndRead(sampleModel());

        CRC32 crc = new CRC32();
        crc.update(bytes, 0, bytes.length - 4);
        int footer = new DataInputStream(new ByteArrayInputStream(bytes, bytes.length - 4, 4)).readInt();
        assertEquals((int) crc.getValue(), footer);
    }

    @Test
    void decoderConsumesFileExactlyUpToFooter() throws IOException {
        // Decoding checks that the footer is the last 4 bytes and nothing follows it.
        Decoded decoded = Decoded.parse(saveAndRead(sampleModel()));
        assertEquals(3, decoded.tensors.size());
    }

    // ------------------------------------------------------------------
    // Full model contents
    // ------------------------------------------------------------------

    @Test
    void savesCompleteModel() throws IOException {
        Decoded decoded = Decoded.parse(saveAndRead(sampleModel()));

        // Config: required keys in a fixed order, then optional keys that are present.
        assertEquals(List.of("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "d_ff",
                "learning_rate", "train_steps", "final_loss", "corpus_name"),
                new ArrayList<>(decoded.config.keySet()));
        assertConfig(decoded, "vocab_size", 0, 4);
        assertConfig(decoded, "context_length", 0, 8);
        assertConfig(decoded, "d_model", 0, 2);
        assertConfig(decoded, "n_layers", 0, 1);
        assertConfig(decoded, "n_heads", 0, 1);
        assertConfig(decoded, "d_ff", 0, 8);
        assertConfig(decoded, "learning_rate", 1, 0.001f);
        assertConfig(decoded, "train_steps", 0, 500);
        assertConfig(decoded, "final_loss", 1, 1.25f);
        assertConfig(decoded, "corpus_name", 2, "Shakespeare sonnets");

        // Tokenizer.
        assertEquals("char", decoded.tokenizerType);
        assertEquals(List.of("a", "b", " ", "\n"), decoded.tokens);
        assertNull(decoded.merges);

        // Tensors: names in parameters() order, shapes, and every float bit-for-bit.
        assertEquals(List.of("embed.token", "lm_head.weight", "ln_final.gamma"),
                new ArrayList<>(decoded.tensors.keySet()));
        Map<String, Tensor> params = sampleModel().parameters();
        for (Map.Entry<String, Tensor> entry : params.entrySet()) {
            DecodedTensor t = decoded.tensors.get(entry.getKey());
            assertArrayEquals(entry.getValue().shape(), t.shape, entry.getKey());
            assertArrayEquals(entry.getValue().toArray(), t.data, entry.getKey());
        }
    }

    @Test
    void tensorsAreWrittenInParametersOrderNotSorted() throws IOException {
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("z.last", vector(1f));
        params.put("a.first", vector(2f));
        params.put("m.middle", vector(3f));
        StubModel model = new StubModel(sampleConfig(), charTokenizer(), params);

        Decoded decoded = Decoded.parse(saveAndRead(model));
        assertEquals(List.of("z.last", "a.first", "m.middle"), new ArrayList<>(decoded.tensors.keySet()));
    }

    @Test
    void specialFloatValuesArePreservedExactly() throws IOException {
        float[] values = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.0f,
                Float.MIN_VALUE, Float.MAX_VALUE};
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("w", TENSORS.create(values, new int[]{2, 3}));
        StubModel model = new StubModel(sampleConfig(), charTokenizer(), params);

        DecodedTensor t = Decoded.parse(saveAndRead(model)).tensors.get("w");
        for (int i = 0; i < values.length; i++) {
            assertEquals(Float.floatToRawIntBits(values[i]), Float.floatToRawIntBits(t.data[i]), "index " + i);
        }
    }

    @Test
    void rank3TensorIsRowMajor() throws IOException {
        float[] data = new float[2 * 3 * 4];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        Tensor tensor = TENSORS.create(data, new int[]{2, 3, 4});
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("cube", tensor);
        StubModel model = new StubModel(sampleConfig(), charTokenizer(), params);

        DecodedTensor t = Decoded.parse(saveAndRead(model)).tensors.get("cube");
        assertArrayEquals(new int[]{2, 3, 4}, t.shape);
        // Element [1][2][3] is at row-major offset 1*12 + 2*4 + 3 = 23.
        assertEquals(tensor.get(1, 2, 3), t.data[23]);
        assertArrayEquals(data, t.data);
    }

    // ------------------------------------------------------------------
    // Config
    // ------------------------------------------------------------------

    @Test
    void optionalConfigKeysAreOmittedWhenAbsent() throws IOException {
        StubConfig config = new StubConfig(4, 8, 2, 1, 1, 8,
                OptionalDouble.empty(), OptionalInt.empty(), OptionalDouble.empty(), Optional.empty());
        StubModel model = new StubModel(config, charTokenizer(), sampleParameters());

        Decoded decoded = Decoded.parse(saveAndRead(model));
        assertEquals(List.of("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "d_ff"),
                new ArrayList<>(decoded.config.keySet()));
    }

    @Test
    void someOptionalKeysCanBePresentWithoutOthers() throws IOException {
        StubConfig config = new StubConfig(4, 8, 2, 1, 1, 8,
                OptionalDouble.empty(), OptionalInt.of(42), OptionalDouble.empty(), Optional.of(""));
        StubModel model = new StubModel(config, charTokenizer(), sampleParameters());

        Decoded decoded = Decoded.parse(saveAndRead(model));
        assertConfig(decoded, "train_steps", 0, 42);
        assertConfig(decoded, "corpus_name", 2, "");
        assertFalse(decoded.config.containsKey("learning_rate"));
        assertFalse(decoded.config.containsKey("final_loss"));
    }

    // ------------------------------------------------------------------
    // Tokenizer
    // ------------------------------------------------------------------

    @Test
    void tokensAreWrittenAsLengthPrefixedUtf8() throws IOException {
        // Multi-byte characters: 2 bytes (é, ſ), 3 bytes (€), 4 bytes (emoji), and empty.
        List<String> tokens = List.of("é", "ſ", "€", "😀", "", "a");
        StubTokenizer tokenizer = new StubTokenizer("char", tokens, List.of());
        StubModel model = new StubModel(configWithVocab(tokens.size()), tokenizer, sampleParameters());

        byte[] bytes = saveAndRead(model);
        Decoded decoded = Decoded.parse(bytes);
        assertEquals(tokens, decoded.tokens);

        // Check the raw bytes for "€" directly: length 3, then E2 82 AC.
        byte[] euro = {0x00, 0x00, 0x00, 0x03, (byte) 0xE2, (byte) 0x82, (byte) 0xAC};
        assertTrue(indexOf(bytes, euro) >= 0, "expected '€' encoded as 3 UTF-8 bytes with a length prefix");
    }

    @Test
    void bpeTokenizerWritesMergesInOrder() throws IOException {
        List<String> tokens = List.of("a", "b", "c", "ab", "abc");
        List<int[]> merges = List.of(new int[]{0, 1}, new int[]{3, 2});
        StubTokenizer tokenizer = new StubTokenizer("bpe", tokens, merges);
        StubModel model = new StubModel(configWithVocab(tokens.size()), tokenizer, sampleParameters());

        Decoded decoded = Decoded.parse(saveAndRead(model));
        assertEquals("bpe", decoded.tokenizerType);
        assertEquals(tokens, decoded.tokens);
        assertEquals(2, decoded.merges.size());
        assertArrayEquals(new int[]{0, 1}, decoded.merges.get(0));
        assertArrayEquals(new int[]{3, 2}, decoded.merges.get(1));
    }

    @Test
    void bpeTokenizerWithNoMergesWritesZeroCount() throws IOException {
        StubTokenizer tokenizer = new StubTokenizer("bpe", List.of("a", "b", "c", "d"), List.of());
        StubModel model = new StubModel(sampleConfig(), tokenizer, sampleParameters());

        Decoded decoded = Decoded.parse(saveAndRead(model));
        assertNotNull(decoded.merges);
        assertTrue(decoded.merges.isEmpty());
    }

    // ------------------------------------------------------------------
    // File behaviour
    // ------------------------------------------------------------------

    @Test
    void savingTwiceProducesIdenticalBytes() throws IOException {
        Path first = dir.resolve("first.tgsw");
        Path second = dir.resolve("second.tgsw");
        dao.save(first.toString(), sampleModel());
        dao.save(second.toString(), sampleModel());

        assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second));
    }

    @Test
    void saveReplacesExistingFile() throws IOException {
        Path target = dir.resolve("model.tgsw");
        Files.write(target, "old contents".getBytes(StandardCharsets.UTF_8));

        dao.save(target.toString(), sampleModel());

        Decoded.parse(Files.readAllBytes(target));
        assertEquals(List.of("model.tgsw"), listDir());
    }

    @Test
    void failureDuringWriteLeavesExistingFileUntouchedAndNoTempFile() throws IOException {
        Path target = dir.resolve("model.tgsw");
        byte[] old = "old contents".getBytes(StandardCharsets.UTF_8);
        Files.write(target, old);

        // A broken tensor whose data does not match its shape fails partway through writing.
        LinkedHashMap<String, Tensor> params = sampleParameters();
        params.put("broken", fakeTensor(new int[]{2, 2}, new float[]{1f, 2f, 3f}));
        StubModel model = new StubModel(sampleConfig(), charTokenizer(), params);

        assertThrows(IllegalStateException.class, () -> dao.save(target.toString(), model));
        assertArrayEquals(old, Files.readAllBytes(target));
        assertEquals(List.of("model.tgsw"), listDir());
    }

    // ------------------------------------------------------------------
    // Invalid models
    // ------------------------------------------------------------------

    @Test
    void rejectsVocabSizeMismatch() {
        StubModel model = new StubModel(configWithVocab(5), charTokenizer(), sampleParameters());
        assertRejected(model, "Tokenizer has 4 tokens but vocab_size is 5");
    }

    @Test
    void rejectsUnknownTokenizerType() {
        StubTokenizer tokenizer = new StubTokenizer("word", List.of("a", "b", "c", "d"), List.of());
        assertRejected(new StubModel(sampleConfig(), tokenizer, sampleParameters()),
                "Unknown tokenizer type 'word'");
    }

    @Test
    void rejectsNullToken() {
        StubTokenizer tokenizer = new StubTokenizer("char", Arrays.asList("a", null, "c", "d"), List.of());
        assertRejected(new StubModel(sampleConfig(), tokenizer, sampleParameters()), "Token 1 is null");
    }

    @Test
    void rejectsCharTokenizerWithMerges() {
        StubTokenizer tokenizer = new StubTokenizer("char", List.of("a", "b", "c", "d"),
                List.<int[]>of(new int[]{0, 1}));
        assertRejected(new StubModel(sampleConfig(), tokenizer, sampleParameters()),
                "A char tokenizer must not have merges");
    }

    @Test
    void rejectsMergeThatIsNotAPair() {
        StubTokenizer tokenizer = new StubTokenizer("bpe", List.of("a", "b", "c", "d"),
                List.<int[]>of(new int[]{0, 1, 2}));
        assertRejected(new StubModel(sampleConfig(), tokenizer, sampleParameters()),
                "Merge 0 is not a pair of token IDs");
    }

    @Test
    void rejectsMergeWithOutOfRangeTokenId() {
        StubTokenizer tokenizer = new StubTokenizer("bpe", List.of("a", "b", "c", "d"),
                List.<int[]>of(new int[]{0, 4}));
        assertRejected(new StubModel(sampleConfig(), tokenizer, sampleParameters()),
                "Merge 0 refers to token ID 4, outside [0, 4)");
    }

    @Test
    void rejectsModelWithNoParameters() {
        StubModel model = new StubModel(sampleConfig(), charTokenizer(), new LinkedHashMap<>());
        assertRejected(model, "Model has no parameters to save");
    }

    @Test
    void rejectsEmptyTensorName() {
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("", vector(1f));
        assertRejected(new StubModel(sampleConfig(), charTokenizer(), params), "Tensor name must be non-empty");
    }

    @Test
    void rejectsRankZeroTensor() {
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("scalar", fakeTensor(new int[0], new float[]{1f}));
        assertRejected(new StubModel(sampleConfig(), charTokenizer(), params), "Tensor 'scalar' has rank 0");
    }

    @Test
    void rejectsTensorWithZeroDimension() {
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("embed.token", fakeTensor(new int[]{0, 64}, new float[0]));
        assertRejected(new StubModel(sampleConfig(), charTokenizer(), params),
                "Tensor 'embed.token' has invalid shape [0, 64]");
    }

    @Test
    void rejectsNullModel() {
        assertThrows(NullPointerException.class, () -> dao.save(dir.resolve("m.tgsw").toString(), null));
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private byte[] saveAndRead(Model model) throws IOException {
        Path file = dir.resolve("model.tgsw");
        dao.save(file.toString(), model);
        return Files.readAllBytes(file);
    }

    /** Asserts the save is rejected with the given message and nothing is left on disk. */
    private void assertRejected(Model model, String message) {
        Path target = dir.resolve("model.tgsw");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> dao.save(target.toString(), model));
        assertEquals(message, e.getMessage());
        try {
            assertEquals(List.of(), listDir());
        } catch (IOException io) {
            fail(io);
        }
    }

    private List<String> listDir() throws IOException {
        try (Stream<Path> files = Files.list(dir)) {
            return files.map(p -> p.getFileName().toString()).sorted().collect(Collectors.toList());
        }
    }

    private static void assertConfig(Decoded decoded, String key, int tag, Object value) {
        ConfigValue v = decoded.config.get(key);
        assertNotNull(v, "missing config key " + key);
        assertEquals(tag, v.tag, "tag of " + key);
        assertEquals(value, v.value, "value of " + key);
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }

    // ---- sample data ----

    private static StubModel sampleModel() {
        return new StubModel(sampleConfig(), charTokenizer(), sampleParameters());
    }

    private static StubConfig sampleConfig() {
        return configWithVocab(4);
    }

    private static StubConfig configWithVocab(int vocabSize) {
        return new StubConfig(vocabSize, 8, 2, 1, 1, 8,
                OptionalDouble.of(0.001), OptionalInt.of(500), OptionalDouble.of(1.25),
                Optional.of("Shakespeare sonnets"));
    }

    private static StubTokenizer charTokenizer() {
        return new StubTokenizer("char", List.of("a", "b", " ", "\n"), List.of());
    }

    private static LinkedHashMap<String, Tensor> sampleParameters() {
        LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();
        params.put("embed.token", TENSORS.create(
                new float[]{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}, new int[]{4, 2}));
        params.put("lm_head.weight", TENSORS.create(
                new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new int[]{2, 4}));
        params.put("ln_final.gamma", TENSORS.create(new float[]{1f, 1f}, new int[]{2}));
        return params;
    }

    private static Tensor vector(float... values) {
        return TENSORS.create(values, new int[]{values.length});
    }

    /**
     * A tensor that only answers {@code shape()} and {@code toArray()}, for
     * shapes and inconsistencies a real tensor would refuse to construct.
     */
    private static Tensor fakeTensor(int[] shape, float[] data) {
        return (Tensor) Proxy.newProxyInstance(Tensor.class.getClassLoader(), new Class<?>[]{Tensor.class},
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "shape":
                            return shape.clone();
                        case "rank":
                            return shape.length;
                        case "toArray":
                            return data.clone();
                        default:
                            throw new UnsupportedOperationException(method.getName());
                    }
                });
    }

    // ---- stubs ----

    private static final class StubModel implements Model {
        private final ModelConfig config;
        private final Tokenizer tokenizer;
        private final LinkedHashMap<String, Tensor> parameters;

        StubModel(ModelConfig config, Tokenizer tokenizer, LinkedHashMap<String, Tensor> parameters) {
            this.config = config;
            this.tokenizer = tokenizer;
            this.parameters = parameters;
        }

        @Override
        public LinkedHashMap<String, Tensor> parameters() {
            return parameters;
        }

        @Override
        public ModelConfig config() {
            return config;
        }

        @Override
        public Tokenizer tokenizer() {
            return tokenizer;
        }
    }

    private static final class StubConfig implements ModelConfig {
        private final int vocabSize;
        private final int contextLength;
        private final int dModel;
        private final int nLayers;
        private final int nHeads;
        private final int dFf;
        private final OptionalDouble learningRate;
        private final OptionalInt trainSteps;
        private final OptionalDouble finalLoss;
        private final Optional<String> corpusName;

        StubConfig(int vocabSize, int contextLength, int dModel, int nLayers, int nHeads, int dFf,
                   OptionalDouble learningRate, OptionalInt trainSteps, OptionalDouble finalLoss,
                   Optional<String> corpusName) {
            this.vocabSize = vocabSize;
            this.contextLength = contextLength;
            this.dModel = dModel;
            this.nLayers = nLayers;
            this.nHeads = nHeads;
            this.dFf = dFf;
            this.learningRate = learningRate;
            this.trainSteps = trainSteps;
            this.finalLoss = finalLoss;
            this.corpusName = corpusName;
        }

        @Override public int getVocabSize() { return vocabSize; }
        @Override public int getContextLength() { return contextLength; }
        @Override public int getDModel() { return dModel; }
        @Override public int getNLayers() { return nLayers; }
        @Override public int getNHeads() { return nHeads; }
        @Override public int getDFf() { return dFf; }
        @Override public OptionalDouble getLearningRate() { return learningRate; }
        @Override public OptionalInt getTrainSteps() { return trainSteps; }
        @Override public OptionalDouble getFinalLoss() { return finalLoss; }
        @Override public Optional<String> getCorpusName() { return corpusName; }
    }

    private static final class StubTokenizer implements Tokenizer {
        private final String type;
        private final List<String> tokens;
        private final List<int[]> merges;

        StubTokenizer(String type, List<String> tokens, List<int[]> merges) {
            this.type = type;
            this.tokens = tokens;
            this.merges = merges;
        }

        @Override public String type() { return type; }
        @Override public List<String> tokens() { return tokens; }
        @Override public List<int[]> merges() { return merges; }
    }

    // ---- independent decoder, written from docs/weight-format.md ----

    private static final class ConfigValue {
        final int tag;
        final Object value;

        ConfigValue(int tag, Object value) {
            this.tag = tag;
            this.value = value;
        }
    }

    private static final class DecodedTensor {
        final int[] shape;
        final float[] data;

        DecodedTensor(int[] shape, float[] data) {
            this.shape = shape;
            this.data = data;
        }
    }

    private static final class Decoded {
        final Map<String, ConfigValue> config = new LinkedHashMap<>();
        String tokenizerType;
        final List<String> tokens = new ArrayList<>();
        List<int[]> merges;
        final Map<String, DecodedTensor> tensors = new LinkedHashMap<>();

        /**
         * Parses a whole file and fails the test if the layout is wrong, the
         * checksum does not match, or bytes remain after the footer.
         */
        static Decoded parse(byte[] bytes) throws IOException {
            DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes));
            Decoded d = new Decoded();

            byte[] magic = new byte[4];
            in.readFully(magic);
            assertArrayEquals("TGSW".getBytes(StandardCharsets.US_ASCII), magic, "magic");
            assertEquals(1, in.readInt(), "version");
            assertEquals(0, in.readInt(), "flags");

            int entryCount = in.readInt();
            for (int i = 0; i < entryCount; i++) {
                String key = readString(in);
                int tag = in.readUnsignedByte();
                Object value;
                switch (tag) {
                    case 0: value = in.readInt(); break;
                    case 1: value = in.readFloat(); break;
                    case 2: value = readString(in); break;
                    default: throw new AssertionError("unknown config tag " + tag + " for " + key);
                }
                assertNull(d.config.put(key, new ConfigValue(tag, value)), "duplicate config key " + key);
            }

            d.tokenizerType = readString(in);
            int vocabSize = in.readInt();
            for (int i = 0; i < vocabSize; i++) {
                d.tokens.add(readString(in));
            }
            if ("bpe".equals(d.tokenizerType)) {
                int mergeCount = in.readInt();
                d.merges = new ArrayList<>();
                for (int i = 0; i < mergeCount; i++) {
                    d.merges.add(new int[]{in.readInt(), in.readInt()});
                }
            }

            int tensorCount = in.readInt();
            for (int i = 0; i < tensorCount; i++) {
                String name = readString(in);
                int rank = in.readInt();
                assertTrue(rank >= 1, "rank of " + name);
                int[] shape = new int[rank];
                int size = 1;
                for (int j = 0; j < rank; j++) {
                    shape[j] = in.readInt();
                    assertTrue(shape[j] >= 1, "dimension of " + name);
                    size *= shape[j];
                }
                float[] data = new float[size];
                for (int j = 0; j < size; j++) {
                    data[j] = in.readFloat();
                }
                assertNull(d.tensors.put(name, new DecodedTensor(shape, data)), "duplicate tensor " + name);
            }

            int consumed = bytes.length - in.available();
            assertEquals(bytes.length - 4, consumed, "footer must be exactly the last 4 bytes");
            CRC32 crc = new CRC32();
            crc.update(bytes, 0, consumed);
            assertEquals((int) crc.getValue(), in.readInt(), "checksum");
            assertThrows(EOFException.class, in::readByte, "data after footer");

            // The tokenizer's vocab size must match the config, as the loader will require.
            ConfigValue vocab = d.config.get("vocab_size");
            assertNotNull(vocab, "vocab_size");
            assertEquals(vocab.value, d.tokens.size(), "tokenizer size vs vocab_size");
            return d;
        }

        private static String readString(DataInputStream in) throws IOException {
            int length = in.readInt();
            assertTrue(length >= 0, "string length");
            byte[] b = new byte[length];
            in.readFully(b);
            return new String(b, StandardCharsets.UTF_8);
        }
    }
}
