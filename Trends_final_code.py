# Trends MarketPlace
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import speech_recognition as sr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
import timeit


def survey_analysis(survey_file):


    warnings.filterwarnings('ignore')
    data = pd.read_csv(survey_file)
    data.describe()
    data.head()
    
    #Missing values check
    data.isnull().sum()
    
    # View the distribution of data
    num_bins = 10
    data.hist(bins=num_bins, figsize=(20,15))
    plt.savefig("Histogram plots")
    plt.show()
    
    data['Attrition'].value_counts()
    
    data.groupby('Attrition').mean()
    
    # Plot the distribution of churn employee
    num_bins = 10
    data_churn = data[data['Attrition'] == 'Yes'].hist(bins=num_bins, figsize=(20,15))
    plt.savefig("Histogram plots_Churn")
    plt.show()
    
    # Transfer attrition: Yes = 1, No = 0
    attrition_val = {'Yes': 1, 'No': 0}
    data['Attrition'] =  data['Attrition'].apply(attrition_val.get)
    
    # Select categorical data and turn them into dummy variables, deleted "Over18" for it's the same for all rows
    data_cat = data[['Attrition', 'BusinessTravel','Department', 'EducationField',
                     'Gender', 'JobRole', 'MaritalStatus', 'OverTime']].copy()
    data_cat = pd.get_dummies(data_cat)
    data_cat.shape
    
    # Pick numerical variables
    data_num = data[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction',
                 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
    
    # Check correlation of numerical variables
    sns.set(style="white")
    
    # Compute the correlation matrix
    corr = data_num.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
    # Remove column with same value in all rows (EmployeeCount, StandardHour), unrelated (EmployeeNumber)
    # Remove highly correlated variables: MonlhlyIncome, TotalWorkingYears, PercentSalaryHike, YearsWithCurrManager
    data_num = data[["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
                     "JobLevel", "JobSatisfaction", "MonthlyRate", "NumCompaniesWorked", "PerformanceRating",
                     "RelationshipSatisfaction", "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
                     "YearsSinceLastPromotion"]].copy()
    data_num.shape
    
    
    # Merge final data
    final_data = pd.concat([data_num, data_cat], axis=1)
    final_data.shape
    
    # Select independent variables and dependent variable
    data_vars=final_data.columns.values.tolist()
    y_var=['Attrition']
    x_var=[var for var in data_vars if var not in y_var]
    
    x=final_data[x_var]
    y=final_data['Attrition']
    
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    train, test, target_train, target_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Resolve imbalanced data issue
    from imblearn.over_sampling import SMOTE
    oversampler=SMOTE(random_state=42)
    smote_train, smote_target = oversampler.fit_sample(train,target_train)
    
    
    # Logistic Regression
    
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(smote_train)
    train_std = sc.transform(smote_train)
    test_std = sc.transform(test)
    logreg = LogisticRegression()
    logreg.fit(smote_train, smote_target)
    
    from sklearn.metrics import accuracy_score
    print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(target_test, logreg.predict(test))))
    
    coefs = logreg.coef_.transpose()
    df = pd.DataFrame(coefs, columns=['Coef'])
    df['Variables'] = x_var
    df['abs_Coef'] = np.absolute(coefs)
    final_df = df.sort_values(by=['abs_Coef'], ascending=False)
    final_df
    
    
    # Random Forest
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(smote_train, smote_target)
    
    print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(target_test, rf.predict(test))))
    
    feat_importances = pd.DataFrame(rf.feature_importances_, index=x_var, columns=['Importance']).sort_values(by=['Importance'], ascending=False)
    top_10_factors = feat_importances[0:10]
    top_10_factors.plot.barh().invert_yaxis()
    plt.title('Random Forest Top 10 Important Factors')
    plt.savefig('Top 10 Important Factors')
    plt.show()
    
    # SVM
    
    from sklearn.svm import SVC
    svc = SVC(random_state=42)
    svc.fit(train,target_train)
    print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(target_test, svc.predict(test))))
    
    # Cross Validation
    
    # Cross Validation for Logistic Regression
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, smote_train, smote_target, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy for logistic regression: %.3f" % (results.mean()))
    
    # Cross Validation for Logistic Regression
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model = RandomForestClassifier()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, smote_train, smote_target, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy for Random Forest: %.3f" % (results.mean()))
    
    
    # Classification report for logistic regression
    from sklearn.metrics import classification_report
    print(classification_report(target_test, logreg.predict(test)))
    
    # Classification report for Random Forest
    from sklearn.metrics import classification_report
    print(classification_report(target_test, rf.predict(test)))
    
    # Confusion matrix for logistic regression
    logreg_y_pred = logreg.predict(test)
    logreg_cm = metrics.confusion_matrix(logreg_y_pred, target_test, [1,0])
    sns.heatmap(logreg_cm, cmap="YlGnBu", annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Logistic Regression')
    plt.savefig('logistic_regression')
    plt.show()
    
    
    # Confusion matrix for random forest
    rf_y_pred = rf.predict(test)
    rf_cm = metrics.confusion_matrix(rf_y_pred, target_test, [1,0])
    sns.heatmap(rf_cm, cmap="YlGnBu", annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Logistic Regression')
    plt.savefig('logistic_regression')
    plt.show()
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(target_test, logreg.predict(test))
    fpr, tpr, thresholds = roc_curve(target_test, logreg.predict_proba(test)[:,1])
    rf_roc_auc = roc_auc_score(target_test, rf.predict(test))
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(target_test, rf.predict_proba(test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC')
    plt.show()
    
    # Results of different models
    
    print('\n')
    
    print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(target_test, logreg.predict(test))))
    
    print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(target_test, rf.predict(test))))
    
    print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(target_test, svc.predict(test))))
    
    
    feat_importances = pd.DataFrame(rf.feature_importances_, index=x_var, columns=['Importance']).sort_values(by=['Importance'], ascending=False)
    top_10_factors = feat_importances[0:10]
    top_10_factors.plot.barh().invert_yaxis()
    plt.title('Random Forest Top 10 Important Factors')
    plt.savefig('Top 10 Important Factors')
    plt.show()


# Audio to text

def audio_txt(audio_file):
    r = sr.Recognizer()
    counter = 0
    
    with open(audio_file, 'r') as audio_name:
        
            
        for line3 in audio_name:
            data2 = line3.strip()
            counter +=1
            
            data3 = "exit_interview" + str(counter) + ".txt"
            
            
            with sr.AudioFile(data2) as source:
                audio = r.record(source)
            
            try:
                s = r.recognize_google(audio)
                #print("Text: "+s)
                
                with open(data3, 'w' ) as myfile:
                    myfile.writelines(s)
                    
            except Exception as e:
                print("Exception: "+str(e))
        

# combine all the text files

def read_file(filename):   
    
    with open(filename, 'r') as fin1:
        with open("Output.txt", 'a' ) as f_out:
            for line in fin1:
                data = line.rstrip('\n')
                with open(data, 'r') as fin2:
                    for line2 in fin2:
                        f_out.writelines(line2)
                    f_out.writelines('\n \n')
                    
# text analytics
                  
def text_analytics(filename1):
    with open(filename1, 'r' ) as fin:
        sentence = ""
        for line in fin:
             sentence += line.strip()
             
    # token
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    # lower capitalization
    tokens = [word.lower() for word in tokens]
    
    # Adding stopwords   
    
    my_stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['could','would', 'company','change','always','really','great', 'like','new']
    my_stopwords.extend(newStopWords)
    
    # Removing stopwords  
    filtered_words = [word for word in tokens if word not in my_stopwords]
    
    #Lemmatization
    lmtzr = WordNetLemmatizer()
    lem_list = [lmtzr.lemmatize(word) for word in filtered_words] 
    
#    # stem
#    snowball_stemmer = SnowballStemmer('english')
#    stemmed_list = [snowball_stemmer.stem(word) for word in filtered_words]
     
    # frequency of each word
    fdist1 = FreqDist(lem_list)
    #fdist1.plot(15,cumulative=False)
    #plt.show()
    
#    fdist2 = FreqDist(stemmed_list)
#    fdist2.plot(15,cumulative=False)
#    plt.show()
    
    # frequency of each word
    fdist1 = FreqDist(lem_list)
    df_fdist1 = pd.DataFrame.from_dict(fdist1, orient='index')
    df_fdist1.columns = ['Frequency']
    df_fdist1.index.name = 'Term'
    df_fdist1['word'] = df_fdist1.index
    df_fdist1.sort_values(by = ['Frequency'], axis = 0 , ascending = False , inplace = True)
    
    
    
#    fdist2 = FreqDist(stemmed_list)
#    df_fdist2 = pd.DataFrame.from_dict(fdist2, orient='index')
#    df_fdist2.columns = ['Frequency']
#    df_fdist2.index.name = 'Term'
#    df_fdist2['word'] = df_fdist2.index
#    df_fdist2.sort_values(by = ['Frequency'], axis = 0 , ascending = False , inplace = True)
    
    
    
    # Word Cloud
    word_cld = df_fdist1.head(30)
    new_t = (word_cld['word'] + ' ') * word_cld['Frequency']
    
    my_string = ''
    for word in new_t:
        my_string += word
    
    # Create and generate a word cloud image:
    
    import wordcloud as wc
    
    wordcloud = wc.WordCloud(background_color='black', max_words=300, collocations=False).generate(my_string)
    
    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show

    
def main():
    
    survey_analysis('HR data.csv')
    
    audio_txt("audio_files.txt")
    #print("Audio to text completed")
    
    read_file("exit_interview_list.txt")
    #print("All exit interviews clubbed together")
    
    #print("Results")
    text_analytics("Output.txt")
    
    
    

main()
    
# run file
# Exit interview audio file
# Demo
# Presentation




