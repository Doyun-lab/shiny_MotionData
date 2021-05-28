library(shiny)
library(tidyverse)
library(caret)
library(xgboost)
library(randomForest)
library(keras)

ui <- fluidPage(
  titlePanel(title = "Motion Data Analysis"),
  sidebarPanel(
    fileInput(inputId = "file",
              label = "Please select csv file or txt file.",
              multiple = FALSE,
              buttonLabel = icon(name = "file"),
              placeholder = "No files have been selected yet."),
    
    radioButtons(input = 'sep',
                 label = 'Please select a separator.',
                 choices = c('Comma' = ',', 'Semi-colon' = ';',
                             'Tab' = '\t', 'Space' = ' '),
                 selected = ',',
                 inline = TRUE),
    
    checkboxInput(inputId = "header",
                  label = "The first row is the header.",
                  value = T),
    
    selectInput(inputId = "var",
                label = "Please select a variable.",
                choices = NULL,
                multiple = T), 
    
    radioButtons(inputId = "algorithm",
                 label = "Select an algorithm.",
                 choices = c("XGBoost"="XGB",
                             "Random Forest"="RF",
                             "DNN"="dnn")),
    
    submitButton(text = "Applying Changes",
                 icon = icon(name = "sync"))
  ),
  
  mainPanel(
    uiOutput(outputId = "mainUI")
  )
)

server <- function(input, output, session){
  df <- reactive({
    if(is.null(x = input$file)) return()
    read.csv(file = input$file$datapath, header = input$header,
             sep = input$sep, stringsAsFactors = TRUE)
  })
  
  observe({
    cols <- colnames(x=df())
    updateSelectInput(session = session, inputId = "var", choices = cols)
  })
  
  output$xgb <- renderPrint({
    if(is.null(x=df())) {
      return()
      
    }else{
      library(xgboost)
      library(caret)
      library(e1071)
      cat(" Please wait ...\n")
      
      xgb_data <- df()[c(input$var,"activity")]
      
      set.seed(234)
      
      x = xgb_data %>% select(-activity) %>% data.matrix()
      y = as.factor(xgb_data$activity)
      
      cv_model1 = xgb.cv(data = x, label = as.numeric(y)-1, num_class = levels(y) %>% length,
                         nfold = 10, nrounds = 500, early_stopping_rounds = 200, verbose = F,
                         objective = "multi:softprob", prediction = T, eta = 0.2, gamma = 0)
      
      
      pred_df = cv_model1$pred %>% as.data.frame %>%
        mutate(pred = levels(y)[max.col(.)] %>% as.factor, actual = y)
      
      pred_df %>% select(pred, actual) %>% table
      
      cat("\n\n\n   - 10-fold result -  \n")
      cat("   - Confusion Matrix -   \n")
      print(caret::confusionMatrix(pred_df$pred, pred_df$actual))
      
    }
  })
  
  output$rf <- renderPrint({
    if(is.null(x=df())) {
      return()
      
    }else{
      library(randomForest)
      cat(" Please wait ...\n")
      rf_data <- df()[c(input$var,"activity")]
      
      rf_m <- randomForest::randomForest(as.factor(activity) ~ ., data = rf_data)
      
      importance_value <- data.frame(randomForest::importance(rf_m)) %>%
        arrange(desc(MeanDecreaseGini))
      
      unimp <- subset(importance_value, importance_value$MeanDecreaseGini < 1)
      
      except_name <- rownames(unimp)
      model_data_importance <- rf_data %>% select(-except_name)
      
      set.seed(234)
      cv_list <- createFolds(model_data_importance$activity, k = 10)
      
      cv_accuracy_total <- c()
      for(i in 1:length(cv_list)) {
        valid_index <- cv_list[[i]]
        
        # test 데이터
        cv_valid_set <- model_data_importance[valid_index,]
        
        #train 데이터
        cv_train_set <- model_data_importance[-valid_index,]
        
        # 모델 생성
        rf_m <- randomForest::randomForest(as.factor(activity) ~ ., data = cv_train_set)
        
        # predict
        rf_p <- predict(rf_m, newdata = cv_valid_set, type = "class")
        
        # model acurracy 생성
        cv_accuracy <- sum(cv_valid_set$activity == rf_p, na.rm = T)/nrow(cv_valid_set)
        
        cv_accuracy_total <- c(cv_accuracy_total, cv_accuracy)
      }
      cat("- Random Forest by extracting critical variables -\n")
      cat("- 10-fold result (accuracy) -\n")
      print(mean(cv_accuracy_total))
    }
  })
  
  output$dnn1 <- renderPrint({
    if(is.null(x=df())) {
      return()
      
    }else{
      library(keras)
      library(caret)
      library(e1071)
      cat(" Please wait ...\n\n")
      dnn_data <- df()[c(input$var,"activity")]
      
      set.seed(234)
      inTrain=createDataPartition(1:nrow(dnn_data),p=0.8,list=FALSE)
      train <- dnn_data[inTrain,]
      test <- dnn_data[-inTrain,]
      
      train_x <- train %>% 
        data.frame() %>% 
        select(-activity) %>% 
        as.matrix()
      
      test_x <- test %>% 
        data.frame() %>% 
        select(-activity) %>% 
        as.matrix()
      
      train$activity <- as.factor(train$activity)
      test$activity <- as.factor(test$activity)
      
      train_y <- train$activity %>% as.numeric()-1
      train_y <- train_y %>% to_categorical()
      
      test_y <- test$activity %>% as.numeric()-1
      test_y <- test_y %>% to_categorical()
      
      model <- keras_model_sequential()
      
      model %>% 
        layer_dense(units = 512, activation = 'relu', input_shape = ncol(dnn_data)-1) %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units = 256, activation = 'relu') %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units = 128, activation = 'relu') %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 6, activation = 'softmax')
      
      model %>% compile(
        loss = "categorical_crossentropy",
        optimizer=optimizer_adam(lr = 0.005),
        metrics = "accuracy"
      )
      
      history <- model %>% fit(
        train_x,
        train_y,
        epochs=100,
        validation_split=0.1
      )
      
      pred <- predict(model, test_x) %>% round()
      cat("    - DNN result -     \n\n")
      conf <- confusionMatrix(factor(pred), factor(test_y))
      print(conf$overall[1])
      
      cat("\n\n     - Confusion Matrix - \n")
      classes <- model %>% predict_classes(test_x, batch_size = 10)
      print(table(test$activity, classes))
      
      
    }
  })
  
  output$table <- renderTable({
    if(is.null(x=df())) return() else df()[input$var] %>% head()
  })
  output$glimpse <- renderPrint({
    if(is.null(x=df())) return() else glimpse(x=df()[input$var])
  })
  output$mainUI <- renderUI({
    if(is.null(x=df())) h4("There is no content to display.")
    else tabsetPanel(
      tabPanel(title="Data",
               tableOutput(outputId = "table"), 
               verbatimTextOutput(outputId = "glimpse")),
      if(input$algorithm=="XGB"){
        tabPanel(title="XGBoost",
                 verbatimTextOutput(outputId = "xgb"))
      } else if(input$algorithm=="RF"){
        tabPanel(title="Random Forest",
                 verbatimTextOutput(outputId = "rf"))
      } else{
        tabPanel(title="DNN",
                 verbatimTextOutput(outputId = "dnn1"))
      }
    )
  })
}

shinyApp(ui, server)

