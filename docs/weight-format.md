# `.tgsw` Weight File Format — Version 1

This document defines the binary format Text Generation Studio uses to save and load trained models. A `.tgsw` file contains everything needed to rebuild a model and use it for generation without the original corpus:

1. the model **configuration** (sizes needed to construct the layers),
2. the **tokenizer** vocabulary,
3. every **weight tensor**, and
4. a **checksum** to detect corrupted or truncated files.

Anyone implementing the Save/Load Model use case should be able to write a compatible reader and writer from this document alone.

---

## 1. Conventions

| Thing | Encoding |
|---|---|
| Byte order | **Big-endian** throughout. This is what `java.io.DataOutputStream` / `DataInputStream` use, so use those. |
| `i32` | 4-byte signed integer (`writeInt` / `readInt`). |
| `f32` | 4-byte IEEE 754 float (`writeFloat` / `readFloat`). |
| `u8` | 1 byte (`writeByte` / `readUnsignedByte`). |
| `str` | An `i32` byte length **N**, followed by **N** bytes of UTF-8. Length may be 0. |

> ⚠️ **Do not use `writeUTF` / `readUTF`.** They use Java's "modified UTF-8" and cap strings at 65,535 bytes. Write strings manually:
> ```java
> byte[] b = s.getBytes(StandardCharsets.UTF_8);
> out.writeInt(b.length);
> out.write(b);
> ```

Tensor data is stored in **row-major order** (the last index varies fastest), the same layout `NaiveTensor` uses internally.

---

## 2. File Layout

The file is five sections, in this order:

```
+------------------+
| Header           |
+------------------+
| Config           |
+------------------+
| Tokenizer        |
+------------------+
| Tensors          |
+------------------+
| Footer (CRC32)   |
+------------------+
```

### 2.1 Header

| Field | Type | Value |
|---|---|---|
| magic | 4 bytes | ASCII `T` `G` `S` `W` (`0x54 0x47 0x53 0x57`) |
| version | `i32` | `1` |
| flags | `i32` | `0` (reserved for future use; writers must write 0, readers must ignore unknown bits) |

A file always starts with these 12 bytes:

```
54 47 53 57  00 00 00 01  00 00 00 00
 T  G  S  W   version=1    flags=0
```

### 2.2 Config

Key–value pairs describing the model's shape.

| Field | Type |
|---|---|
| entry count | `i32` |
| entries | repeated *entry count* times (below) |

Each entry:

| Field | Type | Notes |
|---|---|---|
| key | `str` | e.g. `"d_model"` |
| type tag | `u8` | `0` = `i32`, `1` = `f32`, `2` = `str` |
| value | `i32` / `f32` / `str` | depends on the type tag |

**Required keys** (all type `i32`):

| Key | Meaning |
|---|---|
| `vocab_size` | number of tokens; must equal the tokenizer's vocab size |
| `context_length` | maximum sequence length the model was trained with |
| `d_model` | embedding / hidden dimension |
| `n_layers` | number of transformer blocks |
| `n_heads` | attention heads per block (`d_model` must be divisible by `n_heads`) |
| `d_ff` | hidden size of each block's feed-forward layer |

**Optional keys** (for display only; not needed to rebuild the model):

| Key | Type | Meaning |
|---|---|---|
| `learning_rate` | `f32` | learning rate used |
| `train_steps` | `i32` | steps completed |
| `final_loss` | `f32` | loss at the end of training |
| `corpus_name` | `str` | human-readable corpus description |

Readers must **ignore unknown keys**, so new optional keys can be added without a version bump. Entries can be in any order.

### 2.3 Tokenizer

| Field | Type | Notes |
|---|---|---|
| tokenizer type | `str` | `"char"` or `"bpe"` |
| vocab size | `i32` | must equal config `vocab_size` |
| tokens | `str` × vocab size | token strings in **ID order** (token 0 first) |

If the type is `"bpe"`, it is followed by:

| Field | Type | Notes |
|---|---|---|
| merge count | `i32` | number of merge rules |
| merges | (`i32`, `i32`) × merge count | each pair is the IDs of the two tokens merged, in the order the merges were learned |

### 2.4 Tensors

| Field | Type |
|---|---|
| tensor count | `i32` |
| tensors | repeated *tensor count* times (below) |

Each tensor:

| Field | Type | Notes |
|---|---|---|
| name | `str` | unique within the file (see naming below) |
| rank | `i32` | number of dimensions, ≥ 1 |
| shape | `i32` × rank | each dimension ≥ 1 |
| data | `f32` × (product of shape) | row-major |

**Naming convention.** Names are dot-separated paths matching the model's structure:

```
embed.token                    [vocab_size, d_model]
embed.position                 [context_length, d_model]
blocks.{i}.ln1.gamma           [d_model]
blocks.{i}.ln1.beta            [d_model]
blocks.{i}.attn.q.weight       [d_model, d_model]
blocks.{i}.attn.q.bias         [d_model]
blocks.{i}.attn.k.weight       [d_model, d_model]
blocks.{i}.attn.k.bias         [d_model]
blocks.{i}.attn.v.weight       [d_model, d_model]
blocks.{i}.attn.v.bias         [d_model]
blocks.{i}.attn.out.weight     [d_model, d_model]
blocks.{i}.attn.out.bias       [d_model]
blocks.{i}.ln2.gamma           [d_model]
blocks.{i}.ln2.beta            [d_model]
blocks.{i}.ff.in.weight        [d_model, d_ff]
blocks.{i}.ff.in.bias          [d_ff]
blocks.{i}.ff.out.weight       [d_ff, d_model]
blocks.{i}.ff.out.bias         [d_model]
ln_final.gamma                 [d_model]
ln_final.beta                  [d_model]
lm_head.weight                 [d_model, vocab_size]
```

`{i}` runs from `0` to `n_layers - 1`. This list will be finalized when the model architecture is; the **rule** is that the names in the file are exactly the keys of `Model.parameters()`.

Tensor **order in the file does not matter**. Loaders match tensors to model parameters **by name**, not by position. Writers should still write them in `parameters()` order so files are reproducible.

### 2.5 Footer

| Field | Type | Notes |
|---|---|---|
| checksum | `i32` | CRC32 of **every byte before the footer**, cast to `int` |

Computing it is easy with `java.util.zip.CheckedOutputStream` / `CheckedInputStream`:

```java
CheckedOutputStream checked = new CheckedOutputStream(fileOut, new CRC32());
DataOutputStream out = new DataOutputStream(new BufferedOutputStream(checked));
// ... write header, config, tokenizer, tensors ...
out.flush();
int crc = (int) checked.getChecksum().getValue();
out.writeInt(crc);   // the footer itself is not included in the checksum
```

---

## 3. Validation Rules (Loading)

A loader must reject the file, throwing an `InvalidWeightFileException` with a clear message, if any of these hold:

| # | Condition | Example message |
|---|---|---|
| 1 | First 4 bytes are not `TGSW` | `Not a .tgsw file` |
| 2 | `version` is greater than the latest supported version | `Unsupported .tgsw version 2 (max supported: 1)` |
| 3 | A required config key is missing or has the wrong type tag | `Missing required config key 'd_model'` |
| 4 | Tokenizer vocab size ≠ config `vocab_size` | `Tokenizer has 80 tokens but vocab_size is 96` |
| 5 | Unknown tokenizer type | `Unknown tokenizer type 'word'` |
| 6 | A tensor has rank < 1 or any dimension < 1 | `Tensor 'embed.token' has invalid shape [0, 64]` |
| 7 | Two tensors share a name | `Duplicate tensor 'lm_head.weight'` |
| 8 | A model parameter has no tensor in the file, or the file has a tensor the model doesn't | `Missing tensor 'blocks.1.ln2.beta'` / `Unexpected tensor 'foo'` |
| 9 | A tensor's shape doesn't match the parameter's expected shape | `Shape mismatch for 'embed.token': file [96, 64], model [96, 128]` |
| 10 | The file ends early | `File is truncated` |
| 11 | Bytes remain after the footer | `Unexpected data after footer` |
| 12 | CRC32 doesn't match | `Checksum mismatch: file is corrupted` |

Genuine I/O failures (file not found, permission denied) should stay `IOException`s. `InvalidWeightFileException` is only for "the file exists but its contents are wrong."

**Sanity limits.** Before allocating arrays, reject absurd sizes (e.g. a string length or element count that is negative or larger than the remaining file size). This stops a corrupted length field from causing an `OutOfMemoryError`.

---

## 4. Where This Lives in Clean Architecture

```
use_case/save_load_model/
    ModelWeightsDataAccessInterface   ← interface the interactors depend on
        void save(String path, Model model) throws IOException
        Model load(String path) throws IOException, InvalidWeightFileException

data_access/
    TgswModelWeightsDataAccessObject  ← implements the interface; the ONLY class
                                        that knows about bytes, CRC32, or this spec
```

The interactors never see bytes. They just call `save` / `load`. If we ever change formats, only the DAO changes.

### Dependencies on other parts of the project

The DAO needs these from the entities. They should be agreed on early so Save/Load can be built and tested with a stub model:

- **`Model.parameters()`** → `Map<String, Tensor>` (use a `LinkedHashMap` so the order is stable).
- **`Model.config()`** → the hyperparameters object holding the required config keys.
- **`Model.tokenizer()`** → exposes the type, the tokens in ID order, and (for BPE) the merges.
- **A way to build a model from a config, tokenizer, and parameter map** (constructor or factory).
- **`Tensor` raw data access.** Writing a tensor with `get(int...)` on every element works but is slow. We should add something like `float[] toArray()` (returns a copy, row-major) to `Tensor`, plus a factory method to build a tensor from an array and shape.

---

## 5. Test Checklist

- [ ] **Round trip:** save a small model, load it back, and assert every config value, every token, and every tensor (shape and every float) is equal.
- [ ] Header bytes of a saved file are exactly `54 47 53 57 00 00 00 01 00 00 00 00`.
- [ ] Wrong magic → rejected (rule 1).
- [ ] Version `2` → rejected (rule 2).
- [ ] Missing required config key → rejected (rule 3).
- [ ] Unknown config key → **accepted** and ignored.
- [ ] Tokenizer/vocab size mismatch → rejected (rule 4).
- [ ] Missing, extra, and wrong-shaped tensors → rejected (rules 8, 9).
- [ ] File cut off halfway → rejected (rule 10).
- [ ] Flip one byte in the middle → checksum mismatch (rule 12).
- [ ] Tensors written in a shuffled order → still loads correctly.

---

## 6. Versioning

- Adding **optional config keys** doesn't need a version bump.
- Anything else that changes the byte layout (new sections, new type tags, changing string encoding, etc.) **must** increment `version`, and this document must be updated with a changelog entry.

### Changelog

| Version | Changes |
|---|---|
| 1 | Initial format. |