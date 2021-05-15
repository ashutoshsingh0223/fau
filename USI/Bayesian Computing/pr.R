# Reading the datafile
set.seed(12345)
raw.data <- read.csv('~/Downloads/results-20210426-141638.csv')

dim(raw.data)
head(raw.data)

######## Data Augmentation ##############

##### Augmentation Step 1: Finding what's missing $$$$$

# Getting a sequence of events to get a Markov Chain.
# I am planning on using the stationary distribution if it exists to fill blank values in user vectors like:
"""
     event_1 event_2 ...     ....   ....  event_25
user  1        0                              1
"""

# Also I am planning on giving scores to events visited by users based on transition matrix of scores


users <- unique(raw.data$user_id)
length(users)

# Getting unique events
events <- unique(raw.data$event_name)
events
length(events)

# Empty transition matrix
transition_matrix <- matrix(0, byrow=TRUE,nrow=25,ncol=25, dimnames=list(events, events))
transition_matrix

# Iterate over users and build transition matrix
for (user in users){
  user_events <- raw.data[raw.data$user_id == user, ]$event_name
  if(length(user_events) > 1){
    for (index in 2: length(user_events)){
      transition_matrix[user_events[index - 1], user_events[index]]  <- transition_matrix[user_events[index - 1], user_events[index]] + 1
    }
  }
}
row.sum <- rowSums(transition_matrix)
for (event in events){
  transition_matrix[event,] <- transition_matrix[event,] / row.sum[event]
}

transition_matrix['login',]

library(markovchain)

McEvent <- new('markovchain', states=events,
              byrow=TRUE,
              transitionMatrix=transition_matrix,
              name='User Behaviour')
McEvent
summary(McEvent) # Irreducible, All classes are recurrent
period(McEvent) # Period = 1 implying chain is aperiodic


# Since Markov chain is irreducible and aperiodic a unique limiting distribution exists
limiting.dist <- steadyStates(McEvent)
limiting.dist <- as.data.frame(limiting.dist)
a <-limiting.dist["login"][1, 1]
a
##### Augmentation Step 2: Adding a time dependency
# Accounting for: How frequent are user's visits
# In the series of events there are different time intervals for different users(Scale is not uniform)



####### Generating vectors for users #########

### Ignoring Time ###

### Here I am only considering scores from transition matrix.
### Scoring scheme:
#  1. For the first event user gets a score of 1
#  2. For the second event users gets a score of McEvent[first event, second event]
#  3. If there is no transition score user gets a value from stationary distribution
#  4. Not considering time. I ignore the time between events completely.

# Empty user matrix
user_matrix <- matrix(0, byrow=TRUE,nrow=length(users),ncol=length(events), dimnames=list(users, events))
user_matrix

for (user in users){
    user_events <- raw.data[raw.data$user_id == user, ]$event_name
    for (index in 1: length(user_events)){
      if(index == 1){
        user_matrix[user, user_events[index]] = 1
      }
      else{
        transition_score <- transition_matrix[user_events[index - 1], user_events[index]]
        if(transition_score > 0){
          user_matrix[user, user_events[index]]  <- user_matrix[user, user_events[index]] + transition_matrix[user_events[index - 1], user_events[index]]
        }
      }
    }
}

# Storing sparse matrix in a seprate variable
user_matrix_sparse <- user_matrix

# Now filling the empty values for events on basis on values from staionary distribution 
for (user in users){
  for (event in events){
    if(user_matrix[user, event] == 0){
      user_matrix[user, event] <- limiting.dist[event][1,1]
    }
  }

}

library("intRinsic")

user_hid <- Hidalgo(X=user_matrix, nsim = 5000, burn_in = 2500)
post_process_user_hid <- Hidalgo_postpr_chains(output=user_hid, all_chains = F)
autoplot(post_process_user_hid) + ggtitle("Conjugate Prior - Dense")



user_hid_sparse <- Hidalgo(X=user_matrix_sparse, nsim = 5000, burn_in = 2500)
post_process_user_hid_sparse <- Hidalgo_postpr_chains(output=user_hid_sparse, all_chains = F)
autoplot(post_process_user_hid_sparse) + ggtitle("Conjugate Prior - Sparse") 





