# Last Updated: 02/12/24
rm(list = ls())

# Load libraries
library(rstudioapi)
library(tidyverse)
library(tidyr)
library(dplyr)
library(igraph)
library(readxl)
library(glmnet)
# Set Path
setwd("~/RProjects/RED_dialogue/REM")

trans <- read.csv("trans.csv")
pred <- read.csv("trans_preds.csv")


df <- cbind(trans, pred) %>%
  select(-X) %>%
  rename("outcome" = "BC_Trans_Preds")
df %>% glimpse()

unique_sessions <- unique(df$session)

session_dfs <- list()

for (session_id in unique_sessions) {

  session_data <- subset(df, session == session_id)

  session_dfs[[paste0("df_", session_id)]] <- session_data }

#session_dfs %>%  saveRDS(file = "all_sessions.RData")
s1903 <- select(-session) %>% session_dfs[["df_1903"]]

session_dfs %>% glimpse() %>% head(1)