def title_case_apa(title: str) -> str:
    MINOR_WORDS = {
        "a", "an", "the", "and", "at", "by", "for", "in", "of",
        "on", "to", "up", "as", "but", "or", "nor",
    }

    def capitalize_word(word: str) -> str:
        if "-" in word:
            # Capitalize
            parts = word.split("-")
            return "-".join(capitalize_word(part) for part in parts)
        # If the word is already all uppercase (and longer than 1 char),
        if word.isupper() and len(word) > 1:
            return word
        return word.capitalize()

    words = title.strip().split()
    if not words:
        return title

    result = []
    last_index = len(words) - 1

    for i, word in enumerate(words):
        # Check if it's an abbreviation (all uppercase, len > 1) — preserve as-is
        # Strip hyphens for the minor-word check
        bare = word.lower().strip("-")
        if i == 0 or i == last_index:
            # Always capitalize first and last word
            result.append(capitalize_word(word))
        elif bare in MINOR_WORDS and "-" not in word:
            # Minor word that is NOT hyphenated → lowercase
            # But preserve abbreviations
            if word.isupper() and len(word) > 1:
                result.append(word)
            else:
                result.append(word.lower())
        else:
            result.append(capitalize_word(word))

    return " ".join(result)

#==

if __name__ == "__main__":

    # Interactive mode
    print("The quick brown fox jump over a lazy dog")
    while True:
        user_input = input("Enter title ('q' to quit): ").strip()
        if user_input.lower() == "q":
            break
        print(f"  → {title_case_apa(user_input)}\n")
