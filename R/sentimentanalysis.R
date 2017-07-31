#######################################################################################
################################### Sentiment Analysis        #########################
#######################################################################################

# loading packages
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(purrrlyr)

getSentimentAnalysis <- function(tweets.df, time.break = "day"){

  tweets <- tweets.df$tweet
  created <- tweets.df$tweetcreated

  # Read the dictionary for valence computation
  dictionary <- read.csv("utils/dictionary.csv")
  #VALENCE: 1= sad, 9= happy
  #AROUSAL: 1=calm, 9=excited
  #DOMINANCE: 1=controlled, 9=in control

  #Now recode all columns so that neutral equals 0
  dictionary[,2:4] <- sapply(dictionary[,2:4],function(x) x-5)

  # calcualte valence score for each tweet
  scoretweet <- numeric(length(tweets))
  for (i in 1:length(tweets)){

    tweetsplit <- tryCatch({

      strsplit(tweets[i],split=" ")[[1]]

    },error = function(e){
        print(e)
    })

    #find the positions of the words in the Tweet in the dictionary
    m <- match(tweetsplit, dictionary$Word)

    #which words are present in the dictionary?
    present <- !is.na(m)

    #of the words that are present, select their valence
    wordvalences <- dictionary$VALENCE[m[present]]

    #compute the total valence of the tweet
    scoretweet[i] <- mean(wordvalences, na.rm=TRUE)
    if (is.na(scoretweet[i])) scoretweet[i] <- 0 else scoretweet[i] <- scoretweet[i]

  }

  #Group in minutes and take the average per hour / day / min
  #handle time zone
  created.time <- as.POSIXct(created, format="%Y-%m-%d %H:%M:%S",tz="UTC")
  attributes(created.time)$tzone <- "GMT"

  # time.breaks <- round.POSIXt(created.time, units=time.break)
  time.breaks <- lubridate::ceiling_date(created.time, unit=time.break)

  #' Set options as stringAsFactors = FALSE
  options(stringsAsFactors = FALSE)
  #paste tweet, sentiment and timebreak together into a dataframe
  sentiment.df <- as.data.frame(cbind(tweets = tweets, scoretweet = as.numeric(scoretweet),
                                      timebreak = as.character(time.breaks)))

  sentiment.df <- sentiment.df[!is.na(sentiment.df$timebreak),]
  time.breaks <- na.omit(time.breaks)

  # typecast scoretweet as numeric.
  sentiment.df$scoretweet <- as.numeric(sentiment.df$scoretweet)

  # Aggregate the sentiment score over time breaks
  sentiment.agg <- aggregate(scoretweet ~ timebreak,
                             data = sentiment.df,
                             mean,
                             na.action = na.omit)

  # Compute tweet frequency table
  tweet.freq <- table(sentiment.df$timebreak)

  #convert timebreak into POSIXlt time format
  sentiment.agg$timebreak <- as.POSIXlt(sentiment.agg$timebreak)
  sentiment.agg$tweet.freq <- tweet.freq

  # Getting sentiment data into time series data that can be used for plotting
  # create time_index to order data chronologically
  time_index <- seq(from = min(time.breaks),
                    to = max(time.breaks), by = time.break)

  if(nrow(sentiment.agg) != length(time.break)){
    #missing data so we need to interpolate using zoo
    # sentiment time series
    sentiment.ts <- zoo(sentiment.agg$scoretweet, order.by = time_index)
    # frequnecy time series
    freq.ts <- zoo(sentiment.agg$tweet.freq, order.by = time_index)

  }else{
    # sentiment time series
    sentiment.ts <- xts(sentiment.agg$scoretweet, order.by = time_index)
    # frequnecy time series
    freq.ts <- xts(sentiment.agg$tweet.freq, order.by = time_index)
  }

  #paste sentiment and freq time series
  plot.data <- cbind(sentiment.data = sentiment.ts,freq.data = freq.ts)

  return(plot.data)

}


#Assignment

#Up to now we have determined the sentiment of a Tweet over time by looking at single words.
#These are called unigrams. We only looked at the valence. One could also determine the sentiment
#of a Tweet over time by looking at combinations of two words. These are called bigrams.

#Determine the sentiment of a corpus of 500 tweets. Once only based on unigrams and once
#only based on bigrams. Plot both curves in the same plot. Also compute the correlation between
#both curves.

#Do this for valence, arousal and dominance. In total you should have 3 plots with each two curves.

#To help you, here is how you extract bigrams from a string. You also need to change the dictionary.



#' Function to prepare text for sentiment analysis
#' removing special characters, punctuation, numbers links etc..
#'
#' @param textvector vector of text to perform sentiment analysis on
#'
#' @return  textvector cleaned text vector
#'
#' @export
prepareTextForSentimentAnalysis <- function(textvector) {

  # remove retweet entities
  textvector = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", textvector)
  # remove at people
  textvector = gsub("@\\w+", "", textvector)
  # remove punctuation
  textvector = gsub("[[:punct:]]", "", textvector)
  # remove numbers
  textvector = gsub("[[:digit:]]", "", textvector)
  # remove html links
  textvector = gsub("http\\w+", "", textvector)
  # remove unnecessary spaces
  textvector = gsub("[ \t]{2,}", "", textvector)
  textvector = gsub("^\\s+|\\s+$", "", textvector)

  # define "tolower error handling" function
  try.error = function(x)
  {
    # create missing value
    y = NA
    # tryCatch error
    try_error = tryCatch(tolower(x), error=function(e) e)
    # if not an error
    if (!inherits(try_error, "error"))
      y = tolower(x)
    # result
    return(y)
  }
  # lower case using try.error with sapply
  textvector = sapply(textvector, try.error)

  # remove NAs in textvector
  textvector = textvector[!is.na(textvector)]
  names(textvector) = NULL

  return(textvector)

}


#' #' function to get sentiment data.frame from cleaned text vector
#' #'
#' #' @param textvector a text vetor with clened text data
#' #'
#' #' @return sent_df data.frame of text comments with sentiment analysis columns
#' #'
#' #' @export
#' getSentimentAnalysisDF <- function(textvector) {
#'
#'   # classify emotion
#'   class_emo = classify_emotion(textvector, algorithm="bayes", prior=1.0)
#'   # get emotion best fit
#'   emotion = class_emo[,7]
#'   # substitute NA's by "unknown"
#'   emotion[is.na(emotion)] = "unknown"
#'
#'   # classify polarity
#'   class_pol = classify_polarity(textvector, algorithm="bayes")
#'   # get polarity best fit
#'   polarity = class_pol[,4]
#'
#'
#'   # data frame with results
#'   sent_df = data.frame(text=textvector, emotion=emotion,
#'                        polarity=polarity, stringsAsFactors=FALSE)
#'
#'   # sort data frame
#'   sent_df = within(sent_df,
#'                    emotion <- factor(emotion, levels=names(sort(table(emotion), decreasing=TRUE))))
#'
#'   return(sent_df)
#'
#' }


#' function to get comparion wordcloud of sentiment analysis
#'
#' @param sent_df data.frame of sentiment analysed content with emotion
#' and polarity columns
#'
#' @return cloud comparison wordcloud to be plot
#'
#' @export
getSentimentAnalysisWordCloud <- function(sent_df) {

  # separating text by emotion
  emos = levels(factor(sent_df$emotion))
  nemo = length(emos)
  emo.docs = rep("", nemo)
  for (i in 1:nemo)
  {
    tmp = some_txt[emotion == emos[i]]
    emo.docs[i] = paste(tmp, collapse=" ")
  }

  # remove stopwords
  emo.docs = removeWords(emo.docs, stopwords("english"))
  # create corpus
  corpus = Corpus(VectorSource(emo.docs))
  tdm = TermDocumentMatrix(corpus)
  tdm = as.matrix(tdm)
  colnames(tdm) = emos

  # comparison word cloud
  cloud <- comparison.cloud(tdm, colors = brewer.pal(nemo, "Dark2"),
                            scale = c(3,.5), random.order = FALSE, title.size = 1.5)

  return(cloud)
}



#' function to train model for sentiment clasification
#'
#' @param textvector a text vetor with clened text data
#'
#' @return trained.model glmnet classified model
#'
#' @export
trainModelForSentimentAnalysis <- function(tweets_classified = NULL) {
  ### loading and preprocessing a training set of tweets
  # function for converting some symbols
  conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

  ##### loading classified tweets ######
  # source: http://help.sentiment140.com/for-students/
  # 0 - the polarity of the tweet (0 = negative, 4 = positive)
  # 1 - the id of the tweet
  # 2 - the date of the tweet
  # 3 - the query. If there is no query, then this value is NO_QUERY.
  # 4 - the user that tweeted
  # 5 - the text of the tweet

  tweets_classified <- read_csv('data/training.1600000.processed.noemoticon.csv',
                                col_names = c('sentiment', 'id', 'date', 'query', 'user', 'text')) %>%
  # converting some symbols
  dmap_at('text', conv_fun) %>%
  # replacing class values
  mutate(sentiment = ifelse(sentiment == 0, 0, 1))

  # data splitting on train and test
  set.seed(2340)
  trainIndex <- createDataPartition(tweets_classified$sentiment, p = 0.8,
                                    list = FALSE,
                                    times = 1)
  tweets_train <- tweets_classified[trainIndex, ]
  tweets_test <- tweets_classified[-trainIndex, ]

  ##### doc2vec #####
  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- word_tokenizer

  it_train <- itoken(tweets_train$text,
                     preprocessor = prep_fun,
                     tokenizer = tok_fun,
                     ids = tweets_train$id,
                     progressbar = TRUE)
  it_test <- itoken(tweets_test$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = tweets_test$id,
                    progressbar = TRUE)

  # creating vocabulary and document-term matrix
  vocab <- create_vocabulary(it_train)
  vectorizer <- vocab_vectorizer(vocab)
  dtm_train <- create_dtm(it_train, vectorizer)
  dtm_test <- create_dtm(it_test, vectorizer)
  # define tf-idf model
  tfidf <- TfIdf$new()
  # fit the model to the train data and transform it with the fitted model
  dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
  dtm_test_tfidf <- fit_transform(dtm_test, tfidf)

  # train the model
  t1 <- Sys.time()
  glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = tweets_train[['sentiment']],
                                 family = 'binomial',
                                 # L1 penalty
                                 alpha = 1,
                                 # interested in the area under ROC curve
                                 type.measure = "auc",
                                 # 5-fold cross-validation
                                 nfolds = 5,
                                 # high value is less accurate, but has faster training
                                 thresh = 1e-3,
                                 # again lower number of iterations for faster training
                                 maxit = 1e3)
  print(difftime(Sys.time(), t1, units = 'mins'))

  plot(glmnet_classifier)
  print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

  preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
  glmnet:::auc(as.numeric(tweets_test$sentiment), preds)

  # save the model for future using
  saveRDS(glmnet_classifier, 'model/glmnet_classifier.RDS')
  saveRDS(vectorizer, 'model/vectorizer.RDS')
  return(vectorizer)

}


#' function to apply trained model to tweets data.frame
#'
#' @param df_tweets data.frame of tweet comments
#' @param glmnet_classifier tained glm_net model
#'
#' @return sentiment_df sentiment analyzed data.frame
#'
#' @export
getSentimentAnalysisDF <- function(df_tweets, glmnet_classifier=NULL) {

  # function for converting some symbols
  conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

  df_tweets %>%
  # converting some symbols
  dmap_at('text', conv_fun)

  ##### doc2vec #####
  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- word_tokenizer

  # preprocessing and tokenization
  it_tweets <- itoken(df_tweets$tweet,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = df_tweets$id,
                      progressbar = TRUE)

  # get vectorizer that was used for developing model
  vectorizer <- readRDS('model/vectorizer.RDS')

  # creating vocabulary and document-term matrix
  dtm_tweets <- create_dtm(it_tweets, vectorizer)

  # define tf-idf model
  tfidf <- TfIdf$new()
  # transforming data with tf-idf
  dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)

  # loading classification model
  glmnet_classifier <- readRDS('model/glmnet_classifier.RDS')

  # predict probabilities of positiveness
  preds_tweets <- glmnet::predict.cv.glmnet(glmnet_classifier, dtm_tweets_tfidf, type = 'response')[ ,1]

  # adding rates to initial dataset
  df_tweets$sentiment <- preds_tweets

  return(df_tweets)

}

#' function to get plot for sentiment analysis data.frame
#'
#' @param df_tweets data.frame of glm_net calssified tweets
#'
#' @return sentimentplot
#'
#' @export
getSentimentAnalysisPlot <- function(df_tweets) {
  # color palette
  cols <- c("#ce472e", "#f05336", "#ffd73e", "#eec73a", "#4ab04a")

  #' format tweetcreated to date time format
  df_tweets$tweetcreated <- as.POSIXct(df_tweets$tweetcreated, format="%Y-%m-%d %H:%M:%S",tz="UTC")

  set.seed(932)
  samp_ind <- sample(c(1:nrow(df_tweets)), nrow(df_tweets) * 0.1) # 10% for labeling

  # plotting
  x <- ggplot(df_tweets, aes(x = tweetcreated, y = sentiment, color = sentiment)) +
          theme_minimal() +
          scale_color_gradientn(colors = cols, limits = c(0, 1),
                                breaks = seq(0, 1, by = 1/4),
                                labels = c("0", round(1/4*1, 1), round(1/4*2, 1), round(1/4*3, 1), round(1/4*4, 1)),
                                guide = guide_colourbar(ticks = T, nbin = 50, barheight = .5, label = T, barwidth = 10)) +
          geom_point(aes(color = sentiment), alpha = 0.8) +
          geom_hline(yintercept = 0.65, color = "#4ab04a", size = 1.5, alpha = 0.6, linetype = "longdash") +
          geom_hline(yintercept = 0.35, color = "#f05336", size = 1.5, alpha = 0.6, linetype = "longdash") +
          geom_smooth(size = 1.2, alpha = 0.2) +
          ggrepel::geom_label_repel(data = df_tweets[samp_ind, ],
                           aes(label = round(sentiment, 2)),
                           fontface = 'bold',
                           size = 2.5,
                           max.iter = 100) +
          theme(legend.position = 'bottom',
                legend.direction = "horizontal",
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                plot.title = element_text(size = 20, face = "bold", vjust = 2, color = 'black', lineheight = 0.8),
                axis.title.x = element_text(size = 16),
                axis.title.y = element_text(size = 16),
                axis.text.y = element_text(size = 8, face = "bold", color = 'black'),
                axis.text.x = element_text(size = 8, face = "bold", color = 'black')) +
          ggtitle("Tweets Sentiment rate (probability of positiveness)")

    return(x)
}
