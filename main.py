from stats import get_num_words, get_uniq_char_count, sort_dict, generate_report
import sys

def get_book_text(filepath):
    with open(f"{filepath}", 'r', encoding="utf-8") as f:
        file_contents = f.read()
    return file_contents

def main():
    if len(sys.argv) !=2:
        raise Exception("Usage: python3 main.py <path_to_book>")
    else:
        book_path = sys.argv[1]
        book_text = get_book_text(book_path)
        word_count = get_num_words(book_text)
        uniq_char_count = get_uniq_char_count(book_text)
        sorted = sort_dict(uniq_char_count)
        print(generate_report(word_count, sorted,book_path))
main()