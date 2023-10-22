import pandas as pd
import regex as re
import numpy as np
import os
import datetime
import phik
from phik import resources
from phik.binning import bin_data
from phik.report import plot_correlation_matrix
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
import gensim
from gensim import corpora
import gensim.corpora as corpora
from gensim.models import TfidfModel
import pyLDAvis
import pyLDAvis.gensim
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import figure as fig
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import spacy
import ru_core_news_lg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("start")

###The varaiables 
    #Select the analysis case by deleting the hashtag before the case you want to run and put a hashtag before the eother to disbale it. I originally intended to write a more elegant way to do this, but later thought it would be better to put this time into refinfing the thesis.
#case = "Ukraine" 
case = "Georgia"
print(case)



lda_directory = r'/home/alex/Desktop/Master' #change the path here
vars= ["country","nato","country_nato"]

with open('stopwords_ru.txt', 'r', encoding = "UTF-8") as f:
    my_stop_words = [line.strip() for line in f] #cite https://medium.com/nlplanet/text-analysis-topic-modelling-with-spacy-gensim-4cd92ef06e06#6173

if case == "Georgia":
    df = pd.read_csv('Georgia_all.csv') #georgien
if case == "Ukraine":
    df = pd.read_csv('Ukraine_all.csv') #ukraine
df['date'] = pd.to_datetime(df["date"])
df['month'] = df['date'].dt.to_period('M')
unique_months = df['month'].unique()
df = df.assign(country=0, nato=0, country_nato=0)    #adds three bulean collums to the data frame, one for ua, nato and both, default is False

def confirmatory_data_prep(article):
    
        words = article.split()
        split_words =[]
        cleaned_words = []

        for word in words:
            word = re.sub(r'\W+', ' ', word)
            if ' ' in word:
                split_words.extend(word.split())
            else:
                split_words.append(word)
        words =  split_words
         
        for word in words:
            word = re.sub(r'\W+', ' ', word)
            word = word.replace(" ", "")
            word = stemmer.stem(word)
            word = word.lower()
            word = word.replace("ё", "е")
            word = word.replace("й", "и")
            word = stemmer.stem(word)
            cleaned_words.append(word)
        words =  cleaned_words
        return words

def confirmatory_analysis(df,case): #go through all articles, and code them as 1 and 0 variables for: ua, nato and "ua and nato"
    #
    if case == "Georgia":
        co_set = ("Тбилис",  "Грузи", "грузино", "грузинск","Саакашвили", "Цхинвал", "югоосетинская", "Осетии" ,"Осети", "Абхазия","абхазии","абхазиеи", "Сухум") 
    if case == "Ukraine":
        co_set = ("Украин", "Киев", "Донбас", "Донецк", "дончан", "Луганск", "луганчан", "ДНР", "ЛНР", "Харьков", "Мариупол", "Херсон", "Запорож", "Одес", "Крым", "Севастоп", "Керч", "СВО", "спецоперац") 
    
    nato_set = ("^НАТО","Атлант", "Запад",   "Евросоюз", "евросоюз", "европеиск", "США", "Америк","американск", "Обам", "Баиден","Буш" ) 
    co_set = set(x.lower() for x in co_set)
    nato_set = set(x.lower() for x in nato_set)
    
    index = 0

    for article in df["text"]: #break down text and search them for dictionary words
        words = confirmatory_data_prep(article)
        test = df.loc[index, 'id']
            
        for stem in co_set:
            if re.search(str(stem + '(\w+)?'), str(words)):
                df.loc[index, 'country'] = 1
        for stem in nato_set:
            if re.search(stem + '(\w+)?', str(words)):
                df.loc[index, 'nato'] = 1
        if re.search(r"^нато$" or r"^ес$", str(words)):
            df.loc[index, 'nato'] = 1
        if df.country[index] == 1 and df.nato[index] == 1:
            df.loc[index, 'country_nato'] = 1

        index = index + 1
    
    if case == "Georgia":
        start =  pd.Period('2007-08')
        ends = [ pd.Period("2008-08"),  pd.Period("2008-09"),  pd.Period("2009-09")] #+1 cuz < in if start < x < end
    if case == "Ukraine":
        start =  pd.Period('2021-02')
        ends = [ pd.Period("2022-02"),  pd.Period("2022-03"),  pd.Period("2023-03")] #+1 cuz < in if start < x < end

    #phi correlate the two binary variables to anser H1
    corr_data = pd.DataFrame()
    corr_data =  corr_data.assign(country= df["country"],nato= df["nato"])
    phi_corr = phik.phik_matrix(corr_data)

    print("phi over all periods")
    print(phi_corr) # = 0.339966 - for rought data



    corr_data_escalation = pd.DataFrame()
    corr_data_escalation =  corr_data_escalation.assign(country= df.loc[df['month'] == ends[0],"country"],nato= df.loc[df['month'] == ends[0],"nato"])
    phi_escalation = corr_data_escalation.phik_matrix()
    print("phi for escaltion period")
    print(phi_escalation)


    df_by_month = pd.DataFrame(columns=["month", "country", "nato", "country_nato", "total"])
    for e in vars:
        #count total occurances
        raw_occurances = df[e].sum()
        print(e)
        print(raw_occurances)
        total_data = df[e].size
        print(total_data)
        percent_of_data = raw_occurances / total_data
        percent_of_data = percent_of_data * 100
        print("Words belongign to the dirctionary of the variable " + e + " occure in " + str(percent_of_data) + "% of all publications in the selected timespan" )
        #plot occurances by month
        for month in unique_months:
            df_by_month.loc[month, 'month'] = month
            #print(df_by_month.loc[:, 'month'])
            df_by_month.loc[:, e] = df.groupby('month')[e].sum()
            df_by_month.loc[:,"total"] = df.groupby('month').size()

    df_by_month[['nato', 'country_nato','total']] = df_by_month[['nato', 'country_nato','total']].astype(float)
    
    if case == "Georgia":
        start =  pd.Period('2007-08')
        ends = [ pd.Period("2008-08"),  pd.Period("2008-09"),  pd.Period("2009-09")] #+1 cuz < in if start < x < end
    if case == "Ukraine":
        start =  pd.Period('2021-02')
        ends = [ pd.Period("2022-02"),  pd.Period("2022-03"),  pd.Period("2023-03")] #+1 cuz < in if start < x < end

    period_data = pd.DataFrame(columns=["start", "end", "country","country_percent", "nato","nato_percent",  "country_nato", "country_nato_percent"])
    a = 1
    for end in ends:
        country_sum = 0
        nato_sum = 0
        country_nato_sum = 0
        total = 0
        for month in df_by_month['month']:
        
            if start <= month < end:
                country_sum = country_sum + df_by_month.loc[month, 'country'].sum()
                nato_sum = nato_sum + df_by_month.loc[month, 'nato'].sum()
                country_nato_sum =  country_nato_sum + df_by_month.loc[month, 'country_nato'].sum()
                total = total + df_by_month.loc[month, 'total'].sum()
            
        period_data.loc[a] = [start , end , country_sum, country_sum / total , nato_sum, nato_sum / total , country_nato_sum, country_nato_sum / total ]
        start = end
        a = a + 1
    print(period_data)

    # plot the data, figure 1
    plt.figure()
        #excluding the first and last month as they are incomplete and therefore lead to outliers in the visualisation. This does not impact the calcuation which stays complete and accurate. 
    
    x_values = df_by_month.loc[:, 'month'].astype(str)
    y_values = df_by_month.loc[:, 'total'] / 100
    figure(figsize=(20, 20), dpi=300)
    fig = plt.figure()
    ax = plt.axes()
    ax.set_facecolor('#F5F5F5')
    plt.plot(x_values, df_by_month.loc[:, 'country']/y_values, label = "occurances "+ case +" dictionary", linestyle="-.", color='#505050')
    plt.plot(x_values, df_by_month.loc[:, 'nato']/y_values, label = "occurances NATO dictionary", color='#808080', linestyle="solid")
    plt.plot(x_values, df_by_month.loc[:, 'country_nato']/y_values, label = "co-occurances of both "+ case +" and NATO dictionarys", linestyle="--", color='#101010')
    ticks = list(df_by_month.loc[:, 'month'].astype(str))
    plt.xticks([ticks[i] for i in range(len(ticks)) if i % 2 == 0], rotation=45)
    plt.title('Percentage of topic from all texts by month for the '+ case +' case')
    plt.xlabel('Month')
    plt.ylabel("Percent of texts")

    if case == "Georgia":
        ax.axvspan("2007-08","2008-07",  hatch = "/" , facecolor='#D3D3D3',alpha = 0.3)
        ax.axvspan("2008-07", "2008-09", hatch = "", facecolor='#808080',alpha = 0.2)
        ax.axvspan("2008-09", "2009-08", hatch = "\\" , facecolor='#D3D3D3',alpha = 0.3)
        plt.text("2007-08", 24, 'Pre-escalation', fontsize = 10)
        plt.text("2008-07", 24, 'Escalation', fontsize = 10)
        plt.text("2008-10", 24, 'Post-escalation', fontsize = 10)

    if case == "Ukraine":
        ax.axvspan("2021-02","2022-01",  hatch = "/" , facecolor='#D3D3D3',alpha = 0.3)
        ax.axvspan("2022-01", "2022-03", hatch = "", facecolor='#808080',alpha = 0.2)
        ax.axvspan("2022-03", "2023-02", hatch = "\\" , facecolor='#D3D3D3',alpha = 0.3)
        plt.text("2021-02", 2, 'Pre-escalation', fontsize = 10)
        plt.text("2022-01", 2, 'Escalation', fontsize = 10)
        plt.text("2022-03", 2, 'Post-escalation', fontsize = 10)

    ax.legend( )
    plt.show()
    # save to png
    fig.savefig('png_new_graph1.png', dpi=300)

    return(df)

nlp = spacy.load('ru_core_news_lg', disable=['parser', 'ner','textcat']) #here i disable parser and NER
nlp.max_length = 20000000 

def apply_topic_modeling(subject_of_analysis): #takes a list of lists of  as input
    
    #cleaning the data further
    clean_sub_analysis = []
    for sentence in subject_of_analysis:
        sentence_str = " ".join(sentence)
        sentence_str = sentence_str.replace("ё", "е")
        sentence_str = sentence_str.replace("й", "и")
        sentence_str = re.sub(r'\W+', ' ', sentence_str)
        sentence = sentence_str.split()
        clean_sub_analysis.append(sentence)

   #legmmatisation
    doc = ""
   
    lemma_sub_analysis = []
    for sentence in clean_sub_analysis:
        pattern = r"^(\d+)$"#|([0[0-9]|1[0-9]|2[0-3]):[0-5][0-9])$"
        doc = " ".join(sentence)
        doc = nlp(doc)
        lemma_list = []
        for token in doc:
            if (token.lemma_ not in my_stop_words) and not re.search(r'\d', token.lemma_):
                lemma_list.append(token.lemma_)
        lemma_sub_analysis.append(lemma_list)
    
    # Build the bigrams and trigrams 
    bigram = gensim.models.Phrases(lemma_sub_analysis, min_count=5, threshold=90) 
    trigram = gensim.models.Phrases(bigram[lemma_sub_analysis], threshold=90)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(lemma_sub_analysis):
        return(bigram[lemma_sub_analysis])

    def make_trigrams(lemma_sub_analysis):
        return (trigram[bigram[lemma_sub_analysis]])
    
    data_bigrams = make_bigrams(lemma_sub_analysis)
    data_bigrams_trigrams = make_trigrams(data_bigrams)

    
    #create dictionary
        #TF-IDF REMOVAL
    texts = data_bigrams_trigrams
    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.05
    words  = []
    words_missing_in_tfidf = []
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow
   
    #lda moddel
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                                num_topics=10,random_state=100,update_every=1,chunksize=100,passes=10,iterations =100, alpha="auto")
 
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=50)
    return  vis

def model_overall_topics(df):

    articles = " ".join(article.lower() for article in df["text"])
    sentences = nltk.sent_tokenize(articles)
    sentences_list = [nltk.word_tokenize(sentence) for sentence in sentences]
    vis =apply_topic_modeling(sentences_list)

        #save the file
    now = datetime.datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    pyLDAvis.save_html(vis, f"{time_string}lda.html")    
    
    return


def sentiment_analysis_per_article(df):

    neutral_list = []
    negative_list = []
    positive_list = []

    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        
        #sentiment analysis
    results = model.predict(df["text"], k=5) 
    
    for sentiment in results: 

        neutral = sentiment.get('neutral')
        negative = sentiment.get('negative')
        positive = sentiment.get('positive')
        if neutral is None:
            neutral_list.append(0)
        else:
            neutral_list.append(sentiment.get('neutral'))
        if negative is None:
            negative_list.append(0)
        else:
            negative_list.append(sentiment.get('negative'))
        if positive is None:
            positive_list.append(0)
        else:
            positive_list.append(sentiment.get('positive'))
    df['neutral'] = neutral_list
    df['negative'] = negative_list
    df['positive'] = positive_list
    return df  
    
def compare_sentiment(df):
    for variable in vars:
        negativity = df.groupby(variable)["negative"].sum() / df.groupby(variable)["negative"].size() 
        positivity = df.groupby(variable)["positive"].sum() / df.groupby(variable)["positive"].size()
        print("Comparison of the average negativity and posititvity between the articles containing words of the " + variable + " dictionary (1) in contrast to all other articles (0)")
        print(negativity)
        print(positivity)

    return


df = confirmatory_analysis(df,case)

df = sentiment_analysis_per_article(df) #print(topic)
compare_sentiment(df)

model_overall_topics(df)

print("end")
