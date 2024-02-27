---
output:
pdf_document: default
html_document: default
---
```{r, message = FALSE}
# Last Updated: 02/12
rm(list = ls())

# Load libraries
library(rstudioapi)
library(tidyverse)
library(tidyr)
library(RColorBrewer)
library(dplyr)
library(igraph)
library(readxl)
library(igraph)
library(glmnet)
library(remsonic)
library(dplyr)
library(relevent)
# Set Path
setwd("~/RProjects/SNA_REM/REM_new/")
```

```{r}
#session_dfs %>%  saveRDS(file = "all_sessions.RData")
df_1903 <- readRDS("data/all_sessions.RData") %>%
  .[["df_1903"]] %>%
  select(-session) %>%
  rename("dialog_act" = "outcome")


#df_1903 %>% glimpse()
df_1903$speaker %>% unique()

head(df_1903[grepl(" to ", df_1903$speaker),])

```


```{r}

# Remove rows where speaker name is likely an error
df_1903 <- df_1903[!df_1903$speaker %in% c("-", "N/A", "[Audio from 0"),]
df_1903 <- df_1903[!grepl(" to ", df_1903$speaker),]


# Correct speaker names
name_corrections <- c("Nasty" = "Nastya",
                      "Rya" = "Ryan",
                      "Zheny" = "Zhenya",
                      "Al" = "Ali")

df_1903$speaker <- ifelse(df_1903$speaker %in% names(name_corrections),
                          name_corrections[df_1903$speaker],
                          df_1903$speaker)

df_1903$speaker %>% unique()
# df_1903 %>% saveRDS("data/df_1903.RData")
```

```{r}
summary(df_1903)
```


```{r}
names(df_1903)
```

```{r}
head(df_1903)
```
```{r}
df_1903$dialog_act %>% unique()
```

```{r}
# df_1903$dialog_act  
```



```{r}
par(mfrow = c(1, 1), mar = c(1, 1, 1, 1))
unique_speakers <- unique(df_1903$speaker)

# add  vertices based on speakers' names
network_1903 <- graph.empty()
for (speaker in unique_speakers) {
  network_1903 <- add_vertices(network_1903, 1, name = speaker)
}

for (i in 1:(nrow(df_1903) - 1)) {

  if (df_1903$speaker[i] %in% V(network_1903)$name && df_1903$speaker[i + 1] %in% V(network_1903)$name) {

    network_1903 <- add_edges(network_1903, c(which(V(network_1903)$name == df_1903$speaker[i]),
                                              which(V(network_1903)$name == df_1903$speaker[i + 1])))
  }
}
```


```{r}
# Plot the network
plot(network_1903,
     layout = layout_nicely(network_1903),
     vertex.size = 30,
     vertex.label = V(network_1903)$name,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     vertex.color = "lightblue",
     edge.width = .1,
     edge.arrow.size = .4,
     edge.color = "grey50")

```




```{r}
graph_1903 <- df_1903 %>%
  mutate(
    sender = speaker,
    receiver = lead(speaker, default = NA), # shift
    event_time = row_number() # creates a sequence number for each event
  ) %>%
  filter(!is.na(receiver)) %>%
  select(c(event_time, sender, receiver, dialog_act))


people_list <- unique(c(graph_1903$sender, graph_1903$receiver))

lookup_table <- setNames(seq_along(people_list), people_list)
graph_1903[, 'sender_id'] <- lookup_table[graph_1903$sender]
graph_1903[, 'receiver_id'] <- lookup_table[graph_1903$receiver]

dialog_act <- sort(unique(graph_1903$dialog_act)) %>% as.factor()
graph_1903
```


```{r, warning=FALSE, message=FALSE}

par(mfrow = c(1, 1), mar = c(1, 1, 1, 1))
legend <- unique(graph_1903[, 5])
colors <- rainbow(length(legend)) # Generate a set of colors
plot(graph_from_edgelist(),
     edge.arrow.size = 0.25,
     vertex.label = V(network_1903)$name,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     vertex.color = 0,
     vertex.size = 40,
     edge.color = colors[as.factor(E(network_1903)$dialog_act)])


legend("topright",
       legend = legend,
       col = colors,
       pch = 15,
       title = "Dialog Acts",
       cex = 0.75,
       pt.cex = 0.75,
       lwd = 0.5)
```

## REM Model
```{r, warning=FALSE}

data <- data.frame(sid = df_1903$sender_id, rid = df_1903$receiver_id, time = df_1903$event_time, dialog_act = df_1903$sen)


# Model 1 -----------------------------------------------------------------

# Calculate the first set of statistics: the intercept, the effects for repetition (RSndSnd), 
# and reciprocity (RRecSnd)

stats.intercept <- Constant(data)
stats.rrecsnd <- RRecSnd(data)
stats.rsndsnd <- RSndSnd(data)


stats1 <- combine.stats(
  '[Intercept]' = stats.intercept,
  'RRecSnd' = stats.rrecsnd,
  'RSndSnd' = stats.rsndsnd
)

model1 <- FitEventNetworkCore(data, stats1, ordinal = FALSE)
summary(model1)
```


```{r}
# Model 2 -----------------------------------------------------------------

# Adding the second term: the Normalized Total Degree Received (NTDRec)
stats.ntdegrec <- NTDRec(data)
stats2 <- combine.stats(
  '[Intercept]' <- stats.intercept,
  'RRecSnd' = stats.rrecsnd,
  'RSndSnd' = stats.rsndsnd,
  'NTDegRec' = stats.ntdegrec
)

# Run the second model and check the transript_data
model2 <- FitEventNetworkCore(data, stats2, ordinal = FALSE)
summary(model2)
```


```{r}
# Model 3 -----------------------------------------------------------------
stats.sameconstgroup.da <- SameConstGroup(data, dialog_act)

stats3 <- combine.stats(
  '[Intercept]' = stats.intercept,
  'RRecSnd' = stats.rrecsnd,
  'RSndSnd' = stats.rsndsnd,
  'NTDegRec' = stats.ntdegrec,
  'SameConstGroup' = stats.sameconstgroup.da
)

# Run the third model and check the transript_data
model3 <- FitEventNetworkCore(data, stats3, ordinal = FALSE)
summary(model3)
```


