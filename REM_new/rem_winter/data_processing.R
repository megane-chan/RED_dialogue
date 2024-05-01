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
setwd("~/RProjects/SNA_REM/REM_new/")

trans <- read.csv("data/trans.csv")
pred <- read.csv("data/trans_preds.csv")
perf <- read.csv("data/perfs.csv")

df <- cbind(trans, pred) %>%
  rename("outcome" = "BC_Trans_Preds")
df %>% glimpse()

unique_sessions <- unique(df$session)

session_dfs <- list()

for (session_id in unique_sessions) {

  session_data <- subset(df, session == session_id)

  session_dfs[[paste0("df_", session_id)]] <- session_data }
