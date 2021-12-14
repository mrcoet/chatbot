import string
from model import NerualNet, bag_of_words, tokenize
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NerualNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state())
model.eval()

df = pd.read_csv("WikiQASent.pos.ans.tsv", sep="\t")

bot_name = "WikiBot"
print("let's chat! Type 'quite' to exit.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quite":
        print(f"{bot_name}: Goodbye!")
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        filt = df["QuestionID"] == tag
        res = df.loc[filt]
        res = res.iloc[0]
        answer_1 = res["AnswerPhrase1"]
        answer_2 = res["AnswerPhrase2"]
        answer_3 = res["AnswerPhrase3"]

        if not isinstance(answer_1, str):
            answer_1 = " "
        if not isinstance(answer_2, str):
            answer_2 = " "
        if not isinstance(answer_3, str):
            answer_3 = " "

        print("-" * 50)
        print(f'Question: {res["Question"]}')
        print("-" * 50)

        if len(answer_1) > len(answer_2):
            print(f"Short Answer: \n{answer_1}")
        elif len(answer_2) > len(answer_3):
            print(f"Short Answer: \n{answer_2}")
        else:
            print(f"Short Answer: \n{answer_3}")
        print(f"More Details:\n{res['Sentence']}")
        print("-" * 50)
        print(f'See wikipedia For more Details:\nhttps://en.wikipedia.org/wiki/{res["DocumentTitle"]}')
        print("-" * 50)
    else:
        print("Sorry I didn't train enought to answer this question.\nTry another question.\n or write 'quite' to exit.")
        print("-" * 50)
