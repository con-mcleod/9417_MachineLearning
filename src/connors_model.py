"""
COMP9417
Assignment
Author: Connor McLeod (z5058240)
connors_model.py: Main calls on this file for model implementation

"""

import sys, csv, string, re, pprint, math, operator
import pandas as pd
from sklearn.model_selection import train_test_split
from output.scorer_modified import *
from sklearn.ensemble import RandomForestClassifier

ignore_words = ['in','the','to','a','of','and','that','is','was','on','for','he','him','it','his','have','as','has','be','an','are','this','had','us','at','by','her','she','or','i']

def write_to_csv(filename, heading, content):
	"""
	Function to write desired output to csv file
	:param filename: the name of the file
	:param heading: a Python list of the headers for the csv ['Headline','Body ID','Stance']
	:param content: a Python list of lists where each list has the same structure as the header
	:return: created csv filepath
	"""
	with open(filename, mode='w') as file:
		csv_writer = csv.writer(file)
		csv_writer.writerow(heading)
		for row in content:
			csv_writer.writerow(row)
	print (filename, " created")
	return filename



def read_articles(df, read_type):
	"""
	Function to analyse the articles of the training set
	:param df: Pandas dataframe containing training sample
	:return wordcount: Dictionary containing stance, headline word count and body word count for each article (index on Body ID)
	"""

	# initialise manual counters
	num_stances = [0,0,0,0]
	num_words = 0

	# initialise dictionaries for storing article info
	wordcount = {}
	"""
	{
		'Body_ID':
		{
			'Headline_og': ['Headline text...'],
			'Headline': {'word1': #, 'word2': #, ...},
			'Stance': ['unrelated','discuss','agree','disagree'],
			'articleBody': {'word1': #, 'word2': #, ...}
		}
	}
	"""
	if (read_type == "train"):
		stance_count = {}
		stance_count['unrelated'] = {}
		stance_count['discuss'] = {}
		stance_count['agree'] = {}
		stance_count['disagree'] = {}
		"""
		{
			'Stance': {'word1': #, 'word2': #, ...}
		}
		"""

	# for each article
	for index, article in df.iterrows():

		# prepare dictionary for article entry
		Body_ID = article['Body ID']

		# if Body_ID in wordcount.keys():
		# 	print ("duplicate")

		wordcount[Body_ID] = {}
		wordcount[Body_ID]['Headline_og'] = article['Headline']
		wordcount[Body_ID]['Stance'] = article['Stance']

		# for headline and body of article
		for text in ['Headline', 'articleBody']:

			# sanitise article text
			article[text] = article[text].lower()
			article[text] = article[text].translate(str.maketrans('','',string.punctuation))
			article[text] = article[text].replace('\n',' ').replace('\r',' ')
			article[text] = article[text].replace('“','')
			article[text] = article[text].replace('‘','')
			article[text] = re.sub(' +',' ',article[text])

			# separate article words
			article_words = (article[text].split(' '))
			article_words = [word for word in article_words if word not in ignore_words]
			num_words += len(article_words)

			# generate article word count
			wordcount[Body_ID][text] = {}
			wordcount[Body_ID][text] = {word:article_words.count(word) for word in article_words}

			if (read_type == "train"):

				# store word count in stance_count dictionary
				stance = article['Stance']

				# manually create count of training stances
				if text == 'Headline':
					if stance == 'unrelated':
						num_stances[0] += 1
					elif stance == 'discuss':
						num_stances[1] += 1
					elif stance == 'agree':
						num_stances[2] += 1
					elif stance == 'disagree':
						num_stances[3] += 1

				# add or increment word to stance_count
				for word in article_words:
					if (word not in stance_count[stance]):
						stance_count[stance][word] = 1
					else:
						stance_count[stance][word] += 1

	num_articles = len(wordcount)
	print ("Articles read: ", num_articles)

	if (read_type == "train"):
		return wordcount, stance_count, num_articles, num_stances, num_words
	else:
		return wordcount, num_articles, num_words



def stance_prediction(wordcount, stance_count, num_articles, num_stances, num_words):
	"""
	Function to predict article stance given the article headline and word count
	:param wordcount_dict: dictionary containing article's wordcount
	:param stance_count: xxx
	:return prediction: list of lists containing 'Headline', Body ID', 'Stance' where stance is predicted
	"""

	articles_left = num_articles
	prediction = {}
	for bodyID, words in wordcount.items():

		prediction[bodyID] = {}
		P_weight = {}

		relevant_word = 0
		body_words = [body_word for body_word in words['articleBody']]
		for hl_word in words['Headline']:
			if hl_word in body_words:
				relevant_word += 1
		if (relevant_word <= (len(words['Headline'])/3)):
			prediction[bodyID]['Headline'] = words['Headline_og']
			prediction[bodyID]['Stance'] = 'unrelated'
		else:

			for word in words['articleBody']:

				prediction[bodyID][word] = {}

				word_ct = 0
				for i in stance_count.values():
					for j, value in i.items():
						if j == word:
							word_ct += 1
				P_evidence = word_ct / num_words

				for stance in ['discuss', 'agree', 'disagree']:
					P_weight[stance] = 0
					if (stance_count[stance].get(word)):
						P_likelihood = stance_count[stance].get(word) / sum(stance_count[stance].values())

						if stance == 'discuss':
							stance_ct = num_stances[1]
						elif stance == 'agree':
							stance_ct = num_stances[2]
						elif stance == 'disagree':
							stance_ct = num_stances[3]
						P_prior = stance_ct / num_articles

						P_weight[stance] += P_likelihood * P_prior / P_evidence

				prediction[bodyID][word] = max(P_weight.items(), key=operator.itemgetter(1))[0]

			prediction[bodyID]['Headline'] = words['Headline_og']	
			prediction[bodyID]['Stance'] = max(P_weight.items(), key=operator.itemgetter(1))[0]

		articles_left -= 1
		print ("predictions remaining: ", articles_left)

	# pprint.pprint (prediction)
	return prediction


def check_predictions(predictions, wordcount, pred_type):
	"""
	Function description
	:params:
	:return:
	"""

	# initialise column headings
	heading = ['Headline', 'Body ID', 'Stance']

	# set up filenames
	if pred_type == "train":
		pred_fn = "output/cm_train_pred.csv"
		actual_fn = "output/cm_train_actual.csv"
	elif pred_type == "test":
		pred_fn = "output/cm_test_pred.csv"
		actual_fn = "output/cm_test_actual.csv"

	# create predicted stance csv
	test_set = []
	for key, val in predictions.items():
		test_set.append([val['Headline'],key,val['Stance']])
	test_csv = write_to_csv(pred_fn, heading, test_set)

	# create actual stance csv
	true_set = []
	for key, val in predictions.items():
		true_headline = wordcount[key]['Headline_og']
		true_stance = wordcount[key]['Stance']
		true_set.append([true_headline, key, true_stance])
	true_csv = write_to_csv(actual_fn, heading, true_set)

	# print out results
	print("\nNaive-Bayes Results:\n")
	report_score(true_csv, test_csv)


def connors_model():

	train_bodies = pd.read_csv("data/train_bodies.csv")
	train_stances = pd.read_csv("data/train_stances.csv")
	test_bodies = pd.read_csv("data/competition_test_bodies.csv")
	test_stances = pd.read_csv("data/competition_test_stances.csv")

	df = pd.merge(train_bodies, train_stances, on="Body ID") # df.columns.values: ['Body ID' 'articleBody' 'Headline' 'Stance']
	# df.set_index('Body ID', inplace=True)
	df = df.head(3000)
	train_df, validate_df = train_test_split(df, test_size=0.2, random_state=0)

	# read and "learn" from training articles, and check predictions
	wordcount, stance_count, num_articles, num_stances, num_words = read_articles(train_df, "train")
	train_predictions = stance_prediction(wordcount, stance_count, num_articles, num_stances, num_words)
	check_predictions(train_predictions, wordcount, "train")

	# read and check predictions on validation set
	wordcount, num_articles, num_words = read_articles(validate_df, "test")
	test_predictions = stance_prediction(wordcount, stance_count, num_articles, num_stances, num_words)
	check_predictions(test_predictions, wordcount, "test")

	# apply model to competition set
	df_comp = pd.merge(test_bodies, test_stances, on="Body ID") 
	wordcount, num_articles, num_words = read_articles(df_comp, "test")
	test_predictions = stance_prediction(wordcount, stance_count, num_articles, num_stances, num_words)
	check_predictions(test_predictions, wordcount, "test")




