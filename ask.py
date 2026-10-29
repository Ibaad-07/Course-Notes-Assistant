"""
Ask questions about your notes from the terminal.

    python ask.py
    python ask.py --course nlp

Retrieved chunks (with scores and sources) are printed BEFORE the answer, so
you can judge whether a bad answer came from bad retrieval or from the LLM.
"""

import argparse
import sys

import config
import rag


def print_chunks(chunks):
    print("\nRetrieved chunks:")
    for i, c in enumerate(chunks, 1):
        preview = " ".join(c["text"].split())[:150]
        print(f"  {i}. [{c['course']}] {c['file']}, p. {c['page']}  score={c['score']:.3f}")
        print(f"     {preview}...")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your course notes.")
    parser.add_argument("--course", help="only search this course (folder name under notes/)")
    course = parser.parse_args().course

    # Check everything up front, so you don't type a question and then get an error.
    try:
        courses = rag.list_courses()
        if not courses:
            sys.exit("Nothing is indexed yet. Run `python index.py` first.")
        if course and course not in courses:
            sys.exit(f"Unknown course '{course}'. Indexed courses: {', '.join(courses)}")
        rag.check_ollama()
    except RuntimeError as e:
        sys.exit(str(e))

    print(f"Searching: {course or 'all courses'} | model: {config.LLM_MODEL}")
    print("Type a question, or 'quit' to exit. Type 'clear' to forget the conversation.")
    history = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "clear":
            history = []
            print("(history cleared)")
            continue
        if not question:
            continue

        standalone = rag.rewrite_question(question, history)
        if standalone != question:
            print(f"(rewritten as: {standalone})")

        chunks = rag.retrieve(standalone, course=course)
        print_chunks(chunks)

        print("\nAnswer: ", end="", flush=True)
        reply = ""
        for piece in rag.generate(rag.build_prompt(standalone, chunks), stream=True):
            print(piece, end="", flush=True)
            reply += piece
        print()

        # Store the ORIGINAL question: the rewriter should see the conversation
        # as the user actually had it.
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
