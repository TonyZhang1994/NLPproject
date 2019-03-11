# useless, just use it to test the code
import nltk
from gensim.models import Word2Vec
Recipe = {}
f = open("data/recipe")
for i in range(100):
    line = f.readline().splitlines()[0]
    name, recipe = line.split("Recipe:")
    name = name.split("Name:")[1].strip().strip("\"")
    Recipe[name] = recipe
f.close()
print Recipe["Yakisoba"]
text = []
for key in Recipe.keys():
    line = Recipe[key].decode('utf-8')
    tokens = nltk.word_tokenize(line)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    #text.append(" ".join(str(word.encode('utf-8')) for word in tokens))
    text.append(tokens)

model = Word2Vec(text, size=100, window=5, min_count=1, workers=4)
model.train(text, total_examples=len(text), epochs=10)
model.save("data/word2vec.txt")