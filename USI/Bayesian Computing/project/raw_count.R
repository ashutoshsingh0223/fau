# Reading the datafile
set.seed(12345) 

raw.data <- read.csv('~/Downloads/results-20210426-141638.csv')

dim(raw.data)
head(raw.data)

# Getting unique users. Total number rows in transformed dataset
users <- unique(raw.data$user_id)
length(users)

# Getting unique events
events <- unique(raw.data$event_name)
events
length(events)


## Transforming data

# Matrix with users as rows and events as columns. This matrix stores raw count for each event 
# for a user by the times he triggered the event.

raw_count_matrix <- matrix(0, byrow=TRUE, nrow=length(users),
                           ncol=length(events), dimnames=list(users, events))

for (user in users){
  user_events <- raw.data[raw.data$user_id == user, ]$event_name
  for (index in 1: length(user_events)){
    raw_count_matrix[user, user_events[index]]  <- raw_count_matrix[user, user_events[index]] + 1
  }
}

library("intRinsic")

raw_hid <- Hidalgo(X=raw_count_matrix, nsim = 5000, burn_in = 2500)
post_process_raw_hid <- Hidalgo_postpr_chains(output=raw_hid, all_chains = F)
autoplot(post_process_raw_hid) + ggtitle("Conjugate Prior - Raw Count - Sparse")


# Verifying Markov Property
event_seq <- raw.data$event_name
createSequenceMatrix(event_seq)




