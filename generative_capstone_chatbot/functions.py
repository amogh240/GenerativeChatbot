from nltk.tokenize import word_tokenize

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def preprocess(user_message):
    input_sentence = user_message.lower()
    input_sentence = re.sub(r'[^\w\s]', '', input_sentence)
    tokens = word_tokenize(input_sentence)
    input_sentence = [i for i in tokens if i not in stop_words]
    return input_sentence


def compare_overlap(user_message, possible_response):
    similar_words = 0
    for word in user_message:
        if word in possible_response:
            similar_words += 1
    return similar_words


def extract_nouns(tagged_message):
    message_nouns = []
    for word in tagged_message:
        if word[1].startswith("N"):
            message_nouns.append(word[0])
    return message_nouns


def compute_similarity(tokens, category):
    output_list = list()
    for token in tokens:
        output_list.append([token.text, category.text, token.similarity(category)])
    return output_list
