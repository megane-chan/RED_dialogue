# Last Updated: 10/3/23
# This script is for testing different REM approaches

# Clear environment
rm(list = ls())

# Load libraries
library(rstudioapi)
library(tidyverse)
library(tidyr)
library(dplyr)
library(igraph)
library(readxl)

# Set Path
setwd(dirname(getActiveDocumentContext()$path))
setwd("..")

# Load chat and transcript data
nek19_chat <- read.csv(paste0(getwd(),"/results/nek19_chat_df.csv"))
nek21_chat <- read.csv(paste0(getwd(),"/results/nek21_chat_df.csv"))
nek19_trans <- read.csv(paste0(getwd(),"/results/nek19_trans_df.csv")) |>
  mutate(session=paste0("NEK19_",session))
nek21_trans <- read.csv(paste0(getwd(),"/results/nek21_trans_df.csv"))|>
  mutate(session=paste0("NEK21_",session))

# Visualize the transcript data

nek_trans <- full_join(nek19_trans,nek21_trans) |>
  select(speaker, session, labels_h) |>
  filter(session=="NEK21_2") |>
  select(-session)

# Create a sequential dataframe with leading speaker
trans_seq <- nek_trans |>
  mutate(speaker = case_when(speaker == "Sala" | speaker == "Salah:" ~ "Salah",
                             speaker == "Maybe Katya" ~ "Katya",
                             speaker == "Maybe Ashley" | speaker == "Ashley:" ~ "Ashley",
                             speaker == "Igo" ~ "Igor",
                             .default=speaker)) |>
  filter(speaker!="Legen") |>
  mutate(receiver = lead(speaker))

# Make sure that there are the appropriate number of speakers
unique(trans_seq$speaker)

# convert to igraph format
graph_df <- trans_seq |>
  select(speaker,receiver,labels_h) |>
  filter(row_number() <= n()-1) |>
  as.matrix()

seq_graph <- graph_from_edgelist(graph_df[,1:2]) |>
  set_edge_attr("dialog_act",value=graph_df[,3])

E(seq_graph)$color <- as.factor(graph_df[,3])

plot(seq_graph,layout=layout_in_circle, edge.width=0.5,edge.arrow.size=0.5,
     vertex.size=degree(seq_graph,mode="in")/5)

# now convert to number of messages sent
graph_df <- trans_seq |>
  group_by(speaker,receiver,labels_h)|>
  summarise(n = n()) |>
  na.omit() |>
  as.matrix()

seq_graph <- graph_from_edgelist(graph_df[,1:2]) |>
  set_edge_attr("dialog_act",value=graph_df[,3])

plot(seq_graph,layout=layout_in_circle,
     edge.color = as.factor(graph_df[,3]),
     edge.width = as.numeric(graph_df[,4])/7,
     vertex.size=strength(seq_graph,mode="in",weights=graph_df[,4])/5) # currently incorrect node sizes to nodes

# now ignore labels
graph_df <- trans_seq |>
  select(-labels_h) |>
  group_by(speaker,receiver)|>
  summarise(n = n()) |>
  na.omit() |>
  as.matrix()

seq_graph <- graph_from_edgelist(graph_df[,1:2]) 

plot(seq_graph,layout=layout_in_circle,
     edge.width = as.numeric(graph_df[,3])/5,
     arrow.size = .25,
     vertex.size=strength(seq_graph,mode="in",weights=graph_df[,3])/5)


# Simple REM
library(relevent)

node_ids <- tibble(unique(trans_seq$speaker)) |>
  rename(speaker = `unique(trans_seq$speaker)`) |>
  mutate(id = row_number())

rem_df <- trans_seq |>
  mutate(x = row_number()) |>
  select(x,speaker,receiver) |>
  full_join(node_ids) |>
  mutate(speaker = id) |>
  select(-id) |>
  full_join(node_ids,by=join_by(receiver==speaker)) |>
  mutate(receiver = id) |>
  select(-id) |>
  na.omit()
# need to remove self loops

rem_1 <- rem.dyad(rem_df,8,effects=c("NIDSnd","RRecSnd"),ordinal=T,hessian=T)
summary(rem_1)
