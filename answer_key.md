# Answer key for the demo notes

Use this to check whether your RAG system retrieved the right chunks and answered correctly.

| # | Question | Correct answer | Expected source | What it tests |
|---|----------|----------------|-----------------|---------------|
| 1 | Deep Learning final exam weight | 35% | deep_learning_notes.pdf, p. 1 | Basic retrieval of a made-up fact |
| 2 | DL late submission policy | Lose 10% per day late; not accepted after 3 days | deep_learning_notes.pdf, p. 1 | Made-up fact the LLM can't guess |
| 3 | The LR halving rule | If validation loss rises for 3 consecutive epochs, halve the LR; stop after 4 halvings | deep_learning_notes.pdf, p. 3 | Made-up fact the LLM can't guess |
| 4 | LSTM gates | Forget (what to erase), input (what to store), output (what to expose as hidden state) | deep_learning_notes.pdf, p. 5 | Standard concept |
| 5 | Stemming vs lemmatization | Stemming chops endings with rules ("studi"); lemmatization returns the dictionary form ("study"), slower but more accurate | nlp_notes.pdf, p. 2 | Standard concept |
| 6 | Why divide by sqrt(d_k) | To keep the attention values stable before softmax | nlp_notes.pdf, p. 4 | Precise detail |
| 7 | NLP project metrics | Precision, recall and F1-score (accuracy alone loses 5 marks) | nlp_notes.pdf, p. 1 | Made-up fact |
| 8 | Roman Urdu difficulty | No standard spelling (acha / achha / accha); fixed with normalization dictionaries or character/subword models | nlp_notes.pdf, p. 2 | Retrieval with local context |
| 9 | Assignment 2 requirements | **Ambiguous on purpose.** DL: dropout of exactly 0.3 (p. 4). NLP: skip-gram, window 5, 100-dim vectors (p. 3) | Both files | **Course filtering.** Without a filter, the answer may mix both courses |
| 10 | What is attention | DL: an add-on to RNN encoder-decoders (p. 5). NLP: self-attention in transformers (p. 4) | Both files | **Overlapping topic** across courses |
| 11 | NLP proposal deadline | Week 6 | nlp_notes.pdf, p. 1 | Made-up fact |
| 12 | Who teaches Deep Learning | **Not in the notes.** The system should say it couldn't find it | none | **Out-of-scope question.** Tests refusal instead of hallucination |

## Follow-up question test (for the chat history feature)

1. Ask: "What is dropout?"
2. Then ask: "What value does Assignment 2 use for it?"

The second question only works if your query-rewriting step turns it into something like "What dropout value does Assignment 2 use?". The correct answer is 0.3 (deep_learning_notes.pdf, p. 4).
