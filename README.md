# Text Generation Studio

## Team Information

### Team Members
- Pamela Clements
-   Wolfang van Gulik
- Yining Qiu
- John Shuai


---

## Project Description

### Domain

Text Generation Studio is a desktop application for training and running small transformer language models, written entirely in Java with no machine-learning libraries. The neural network maths runs on our own tensor engine (`torch.Tensor`, `NaiveTensor`).

Our focus is **Early Modern English (1500–1750)**. A user builds a corpus by loading their own text files or by importing public-domain books from Project Gutenberg. They then train a model on that corpus and generate new text in the style of the period from a prompt. Trained models are saved in our own weight format (`.tgsw`), so they can be reloaded and shared without retraining.

---

## User Stories

### User Story 1
As a user, when I open the application, I want a button to load a text file that opens a file chooser, so I can load my corpus text file.

### User Story 2
As a user, I want my corpus text to be validated for size, so I know it can be used for training.

### User Story 3
As a user, I want to search Project Gutenberg for books from certain time periods and add them to my corpus, so I can build a training set without finding files myself.

### User Story 4
As a user, I want to click a "Train" button that runs a training loop on my corpus.

### User Story 5
As a user, I want to set model hyperparameters (e.g. learning rate, number of layers, context length, training steps) before training.

### User Story 6
As a user, once training is complete, I want to save my model. E.g. as a `.tgsw` file.

### User Story 7
As a user, I want to switch to a generation tab (the tab where I run the model) in the UI.

### User Story 8
As a user, I want to press a button that opens a file chooser so I can load a `.tgsw` model.

### User Story 9
As a user, I want to control basic generation settings (number of output tokens, temperature, token write speed).

### User Story 10
As a user, I want to enter a prompt and have my model generate text from it using my settings.

### User Story 11
As a user, I want to save the generated output to a text file so I can share it.

---

## Use Cases

### Use Case 1: Load Corpus
The user selects one or more local text files. The system reads them, checks that the combined text is large enough to train on, and stores it as the active corpus. If validation fails, the user sees an error explaining why.
*(User Stories 1, 2)*

### Use Case 2: Import from Project Gutenberg
The user searches for books (by title, author, or subject) restricted to authors active between certain time periods. The system queries the Gutendex API, shows the results, downloads the plain-text editions the user chooses, strips the Gutenberg header and footer, and adds the text to the corpus.
*(User Story 3)*

### Use Case 3: Train Model
The user chooses hyperparameters and starts training. The system tokenizes the corpus, builds a transformer model, and runs the training loop (forward pass, loss, backward pass, parameter update), reporting the loss as it goes. When training finishes, the trained model becomes the active model.
*(User Stories 4, 5)*

### Use Case 4: Save and Load Model
The user saves the active model to a `.tgsw` file, or loads a previously saved one. The system writes or reads every weight tensor, along with the model's configuration and tokenizer vocabulary.
*(User Stories 6, 8)*

### Use Case 5: Generate Text
The user enters a prompt and sets the output length, temperature, and write speed. The system encodes the prompt, samples tokens from the active model one at a time, and streams the text to the screen. The user can then save the output to a file.
*(User Stories 7, 9, 10, 11)*

### Use Case 6 (Stretch): Train BPE Tokenizer
The user trains a byte-pair encoding tokenizer on the corpus instead of using a character-level vocabulary, for better results at the same model size.

---

## Minimal Viable Product (MVP)

| Team Member | MVP Use Case |
|-------------|--------------|
| Pamela Clements | Train Model |
| Name | Generate Text |
| Name | Import from Project Gutenberg |
| Name | Load Corpus |
| Name | Save and Load Model |
| Name | Train BPE Tokenizer (stretch) / Tensor engine support |

---

## Entities

### Tensor
An immutable n-dimensional array of floats with a shape. It supports matrix multiplication, element-wise maths with NumPy-style broadcasting, reductions along an axis, reshaping, transposing, and slicing. All model maths is built on this. Implementations: `NaiveTensor` (reference) and `FastTensor` (optimized, planned).

### Corpus
The training text, plus metadata about its sources (local files or Gutenberg book IDs, titles, and authors). Knows its total size, which is used for validation.

### Tokenizer
Converts between text and sequences of integer token IDs. Holds the vocabulary. Character-level for the MVP, with BPE as a stretch goal.

### Hyperparameters
The model and training configuration: vocabulary size, embedding dimension, number of layers and attention heads, context length, learning rate, batch size, and number of training steps.

### Model
A transformer language model made of layers (embedding, `Linear`, attention, layer norm), each owning its weight tensors. Given a sequence of tokens, it produces next-token scores. Each layer implements its own forward and backward pass.

### GenerationSettings
The settings for one generation request: prompt, maximum output tokens, temperature, and token write speed.

---

## API Usage

**API Name:** Gutendex (Project Gutenberg catalogue API)

**Documentation:** [https://gutendex.com](https://gutendex.com)

**How it is used in the program:**
The Import from Project Gutenberg use case sends search requests to Gutendex (for example `https://gutendex.com/books/?search=shakespeare&author_year_start=1500&author_year_end=1750`). The JSON response gives each book's ID, title, authors, and available formats. When the user picks a book, the program downloads its `text/plain` edition from the URL in the response, removes the Project Gutenberg licence header and footer, and adds the text to the corpus. Gutendex is reached through a data-access interface in the use case layer, so the interactor never depends on HTTP or JSON details directly.

---

## Design Diagrams

### UML Class Diagram

*(To be added.)*

### Sequence Diagrams

*(To be added — one per use case.)*

---

## How to Run

Requirements: Java 11+ and Maven.

```bash
git clone https://github.com/purpleyrose/team-project-csc207.git
cd team-project-csc207
mvn compile
mvn test      # runs the test suite, including the NaiveTensor tests
```

*(Command to launch the GUI to be added once the main application class exists.)*

---

## Notes

- The project follows Clean Architecture. Entities (including the tensor engine) have no dependencies on the UI, frameworks, or the network.
- **Weight file format (`.tgsw`):** big-endian binary. The file starts with the magic bytes `TGSW`, a format version, and the number of tensors. Each tensor is then stored as its name, rank, shape, and float data.
- Neural network maths uses per-layer forward and backward passes, verified with numerical gradient checking, rather than a general autograd system.
