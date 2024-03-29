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

getSentimentAnalysis <- function(texts.df, time.break = "day"){

  texts <- texts.df$text
  created <- texts.df$textcreated

  # Read the dictionary for valence computation
  dictionary <- read.csv("utils/dictionary.csv")
  #VALENCE: 1= sad, 9= happy
  #AROUSAL: 1=calm, 9=excited
  #DOMINANCE: 1=controlled, 9=in control

  #Now recode all columns so that neutral equals 0
  dictionary[,2:4] <- sapply(dictionary[,2:4],function(x) x-5)

  # calcualte valence score for each text
  scoretext <- numeric(length(texts))
  for (i in 1:length(texts)){

    textsplit <- tryCatch({

      strsplit(texts[i],split=" ")[[1]]

    },error = function(e){
        print(e)
    })

    #find the positions of the words in the text in the dictionary
    m <- match(textsplit, dictionary$Word)

    #which words are present in the dictionary?
    present <- !is.na(m)

    #of the words that are present, select their valence
    wordvalences <- dictionary$VALENCE[m[present]]

    #compute the total valence of the text
    scoretext[i] <- mean(wordvalences, na.rm=TRUE)
    if (is.na(scoretext[i])) scoretext[i] <- 0 else scoretext[i] <- scoretext[i]

  }

  #Group in minutes and take the average per hour / day / min
  #handle time zone
  created.time <- as.POSIXct(created, format="%Y-%m-%d %H:%M:%S",tz="UTC")
  attributes(created.time)$tzone <- "GMT"

  # time.breaks <- round.POSIXt(created.time, units=time.break)
  time.breaks <- lubridate::ceiling_date(created.time, unit=time.break)

  #' Set options as stringAsFactors = FALSE
  options(stringsAsFactors = FALSE)
  #paste text, sentiment and timebreak together into a dataframe
  sentiment.df <- as.data.frame(cbind(texts = texts, scoretext = as.numeric(scoretext),
                                      timebreak = as.character(time.breaks)))

  sentiment.df <- sentiment.df[!is.na(sentiment.df$timebreak),]
  time.breaks <- na.omit(time.breaks)

  # typecast scoretext as numeric.
  sentiment.df$scoretext <- as.numeric(sentiment.df$scoretext)

  # Aggregate the sentiment score over time breaks
  sentiment.agg <- aggregate(scoretext ~ timebreak,
                             data = sentiment.df,
                             mean,
                             na.action = na.omit)

  # Compute text frequency table
  text.freq <- table(sentiment.df$timebreak)

  #convert timebreak into POSIXlt time format
  sentiment.agg$timebreak <- as.POSIXlt(sentiment.agg$timebreak)
  sentiment.agg$text.freq <- text.freq

  # Getting sentiment data into time series data that can be used for plotting
  # create time_index to order data chronologically
  time_index <- seq(from = min(time.breaks),
                    to = max(time.breaks), by = time.break)

  if(nrow(sentiment.agg) != length(time.break)){
    #missing data so we need to interpolate using zoo
    # sentiment time series
    sentiment.ts <- zoo(sentiment.agg$scoretext, order.by = time_index)
    # frequnecy time series
    freq.ts <- zoo(sentiment.agg$text.freq, order.by = time_index)

  }else{
    # sentiment time series
    sentiment.ts <- xts(sentiment.agg$scoretext, order.by = time_index)
    # frequnecy time series
    freq.ts <- xts(sentiment.agg$text.freq, order.by = time_index)
  }

  #paste sentiment and freq time series
  plot.data <- cbind(sentiment.data = sentiment.ts,freq.data = freq.ts)

  return(plot.data)

}


#Assignment

#Up to now we have determined the sentiment of a text over time by looking at single words.
#These are called unigrams. We only looked at the valence. One could also determine the sentiment
#of a text over time by looking at combinations of two words. These are called bigrams.

#Determine the sentiment of a corpus of 500 texts. Once only based on unigrams and once
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

  # remove retext entities
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
trainModelForSentimentAnalysis <- function(texts_classified = NULL) {
  ### loading and preprocessing a training set of texts
  # function for converting some symbols
  conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

  ##### loading classified texts ######
  # source: http://help.sentiment140.com/for-students/
  # 0 - the polarity of the text (0 = negative, 4 = positive)
  # 1 - the id of the text
  # 2 - the date of the text
  # 3 - the query. If there is no query, then this value is NO_QUERY.
  # 4 - the user that texted
  # 5 - the text

  texts_classified <- read.csv('data/training.1600000.processed.noemoticon.csv', header=FALSE)
  names(texts_classified) <- c('sentiment', 'id', 'date', 'query', 'user', 'text')
  texts_classified <- texts_classified %>%
                              # converting some symbols
                              purrrlyr::dmap_at('text', conv_fun) %>%
                              # replacing class values
                              dplyr::mutate(sentiment = ifelse(sentiment == 0, 0, 1))


  # data splitting on train and test
  set.seed(2340)
  trainIndex <- caret::createDataPartition(texts_classified$sentiment, p = 0.8,
                                    list = FALSE,
                                    times = 1)
  texts_train <- texts_classified[trainIndex, ]
  texts_test <- texts_classified[-trainIndex, ]

  ##### doc2vec #####
  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- text2vec::word_tokenizer

  it_train <- text2vec::itoken(texts_train$text,
                     preprocessor = prep_fun,
                     tokenizer = tok_fun,
                     ids = texts_train$id,
                     progressbar = TRUE)
  it_test <- text2vec::itoken(texts_test$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = texts_test$id,
                    progressbar = TRUE)

  # creating vocabulary and document-term matrix
  vocab <- text2vec::create_vocabulary(it_train)
  vectorizer <- text2vec::vocab_vectorizer(vocab)
  dtm_train <- text2vec::create_dtm(it_train, vectorizer)
  dtm_test <- text2vec::create_dtm(it_test, vectorizer)
  # define tf-idf model
  tfidf <- text2vec::TfIdf$new()
  # fit the model to the train data and transform it with the fitted model
  dtm_train_tfidf <- text2vec::fit_transform(dtm_train, tfidf)
  dtm_test_tfidf <- text2vec::fit_transform(dtm_test, tfidf)

  # train the model
  t1 <- Sys.time()
  glmnet_classifier <- glmnet::cv.glmnet(x = dtm_train_tfidf, y = texts_train[['sentiment']],
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
  glmnet:::auc(as.numeric(texts_test$sentiment), preds)

  # save the model for future using
  saveRDS(glmnet_classifier, 'model/glmnet_classifier.RDS')
  saveRDS(vectorizer, 'model/vectorizer.RDS')
  return(vectorizer)

}


#' function to apply trained model to texts data.frame
#'
#' @param df_texts data.frame of text comments
#' @param glmnet_classifier tained glm_net model
#'
#' @return sentiment_df sentiment analyzed data.frame
#'
#' @export
getSentimentAnalysisDF <- function(df_texts, glmnet_classifier=NULL) {

  # function for converting some symbols
  conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

  df_texts <- df_texts %>%
              # converting some symbols
              purrrlyr::dmap_at('text', conv_fun)

  ##### doc2vec #####
  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- text2vec::word_tokenizer

  # preprocessing and tokenization
  it_texts <- text2vec::itoken(df_texts$text,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = df_texts$id,
                      progressbar = TRUE)

  # get vectorizer that was used for developing model
  vectorizer <- readRDS('model/vectorizer.RDS')

  # creating vocabulary and document-term matrix
  dtm_texts <- text2vec::create_dtm(it_texts, vectorizer)
  #dtm_texts <- text2vec::create_dtm(it_texts, vectorizer, grow_dtm = FALSE, skip_grams_window=5, skip_grams_window_context = "symmetric", window_size = 0, weights = numeric(0))

  # define tf-idf model
  tfidf <- text2vec::TfIdf$new()
  # transforming data with tf-idf
  dtm_texts_tfidf <- text2vec::fit_transform(dtm_texts, tfidf)

  # loading classification model
  glmnet_classifier <- readRDS('model/glmnet_classifier.RDS')

  # predict probabilities of positiveness
  # @TODO update with non cv predict.
  preds_texts <- glmnet:::predict.cv.glmnet(glmnet_classifier, dtm_texts_tfidf, type = 'response')[ ,1]

  # adding rates to initial dataset
  df_texts$sentiment <- preds_texts

  return(df_texts)

}

#' function to get plot for sentiment analysis data.frame
#'
#' @param df_texts data.frame of glm_net calssified texts
#'
#' @return sentimentplot
#'
#' @export
getSentimentAnalysisPlot <- function(df_texts) {
  # color palette
  cols <- c("#ce472e", "#f05336", "#ffd73e", "#eec73a", "#4ab04a")

  #' format textcreated to date time format
  df_texts$textcreated <- as.POSIXct(df_texts$textcreated, format="%Y-%m-%d %H:%M:%S",tz="UTC")

  set.seed(932)
  samp_ind <- sample(c(1:nrow(df_texts)), nrow(df_texts) * 0.1) # 10% for labeling

  # plotting
  x <- ggplot2::ggplot(df_texts, ggplot2::aes(x = textcreated, y = sentiment, color = sentiment)) +
          ggplot2::theme_minimal() +
          ggplot2::scale_color_gradientn(colors = cols, limits = c(0, 1),
                                breaks = seq(0, 1, by = 1/4),
                                labels = c("0", round(1/4*1, 1), round(1/4*2, 1), round(1/4*3, 1), round(1/4*4, 1)),
                                guide = ggplot2::guide_colourbar(ticks = T, nbin = 50, barheight = .5, label = T, barwidth = 10)) +
    ggplot2::geom_point(ggplot2::aes(color = sentiment), alpha = 0.8) +
    ggplot2::geom_hline(yintercept = 0.65, color = "#4ab04a", size = 1.5, alpha = 0.6, linetype = "longdash") +
    ggplot2::geom_hline(yintercept = 0.35, color = "#f05336", size = 1.5, alpha = 0.6, linetype = "longdash") +
    ggplot2::geom_smooth(size = 1.2, alpha = 0.2) +
    ggrepel::geom_label_repel(data = df_texts[samp_ind, ],
                           ggplot2::aes(label = round(sentiment, 2)),
                           fontface = 'bold',
                           size = 2.5,
                           max.iter = 100) +
    ggplot2::theme(legend.position = 'bottom',
                legend.direction = "horizontal",
                panel.grid.major = ggplot2::element_blank(),
                panel.grid.minor = ggplot2::element_blank(),
                plot.title = ggplot2::element_text(size = 20, face = "bold", vjust = 2, color = 'black', lineheight = 0.8),
                axis.title.x = ggplot2::element_text(size = 16),
                axis.title.y = ggplot2::element_text(size = 16),
                axis.text.y = ggplot2::element_text(size = 8, face = "bold", color = 'black'),
                axis.text.x = ggplot2::element_text(size = 8, face = "bold", color = 'black')) +
          ggplot2::ggtitle("texts Sentiment rate (probability of positiveness)")

    return(x)
}
