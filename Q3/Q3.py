import re
import matplotlib.pyplot as plt
import seaborn as sns

# Read the .txt file and store the paragraph as a string object
with open ('text.txt') as f:
    text = f.readlines()[0]

# Split the paragraph to get sentences using the period as separator
# Store the sentences in a list
word_count = list()
sent = text.split('. ')


# Part (a): For each sentence (converted to lower case), check if the word 'data' is present'
# If yes, then data_count will increase by 1
# The ratio of data_count to the length of the list of sentences can be used to compute the probability
data_count = 0
for i in sent:
    if 'data' in i.lower():
        data_count += 1
prob_data = data_count / len(sent) * 100
print(f'The probability of the word "data" appearring in each line is {prob_data:.2f} %.')


# Part (b): For each sentence, use the regex method to remove all characters that are not word/space
# After converting to lower case, the sentence is split into individual words using space as separator and stored in a list
# Convert the list structure to a set structure to remove duplicate entries
# Find the lenght of the set to determine the number of distinct word count, results stored in the distinct llist

distinct = list()
for i in sent:
    i = i.lower()
    i = re.sub(r'[^\w\s]', '', i)
    words = i.split()
    words_set = set(words)
    distinct.append(len(words_set))

# The results can be plotted into a simple histogram
plt.figure()
sns.histplot(data=distinct)
plt.xlabel('Distinct Word Count')
plt.ylabel('Frequency')
plt.show()


# Part (c): Convert the whole paragraph into lower case.
# Count the occurence of 'data' and 'data analytics'
# The ratio of 'data analytics' to 'data' can be used to compute the probability
text2 = text.lower()
data_count2 = text2.count('data')
analytics_count = text2.count('data analytics')
prob_analytics = analytics_count / data_count2 * 100
print(f'The probability of the word “analytics” occurring after the word “data” is {prob_analytics:.2f} %.')
