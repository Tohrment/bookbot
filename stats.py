def get_num_words(file_contents):
    word_list = []
    word_list = file_contents.split()
    word_count = len(word_list)
    return word_count

def get_uniq_char_count(file_contents):
    char_count = {}
    for i in file_contents.lower():
        if i in char_count:
            char_count[i] += 1
        else:
            char_count[i] = 1
    return char_count

def sort_dict(dict):
    dicts = []
    for d in dict.keys():
        if d.isalpha():
            dicts.append({
                "char": d,
                "num": dict[d]
            })
    dicts.sort(key=sort_on, reverse=True)
    return dicts

def sort_on(items):
    return items["num"]

def generate_report(word_count,sorted_dicts,book_path):
    text = ""
    for s in sorted_dicts:
        if len(text) == 0:
            text += f"{s['char']}: {s['num']}"
        else:
            text += f"\n{s['char']}: {s['num']}"
    report = f"""
============ BOOKBOT ============
Analyzing book found at {book_path}...
----------- Word Count ----------
Found {word_count} total words
--------- Character Count -------
{text}
============= END ===============
"""
    return report