# Reading the datafile
raw.data <- read.csv('~/Downloads/results-20210426-141638.csv')

dim(raw.data)
head(raw.data)

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
