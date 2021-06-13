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

library("intRinsic")


######## Data Augmentation ##############

##### Augmentation Step 1: Finding what's missing $$$$$

# Getting a sequence of events to get a Markov Chain.
# I am planning on using the stationary distribution if it exists to fill blank values in user vectors like:
"""
     event_1 event_2 ...     ....   ....  event_25
user  1        0                              1
"""

# Also I am planning on giving scores to events visited by users based on transition matrix of scores





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


# Storing sparse matrix in a separate variable
user_matrix_sparse <- user_matrix

# Now filling the empty values for events on basis on values from staionary distribution 
for (user in users){
  for (event in events){
    if(user_matrix[user, event] == 0){
      user_matrix[user, event] <- limiting.dist[event][1,1]
    }
  }

}


# Filtering out duplicate users and storing them
dups <- duplicated(user_matrix)

# Finding distinct
user_matrix <- unique(user_matrix)
user_matrix_sparse <- unique(user_matrix_sparse)

users_list <- row.names(user_matrix)

user_hid <- Hidalgo(X=user_matrix, nsim = 5000, burn_in = 2500)
post_process_user_hid <- Hidalgo_postpr_chains(output=user_hid, all_chains = F)
autoplot(post_process_user_hid) + ggtitle("Conjugate Prior - Dense")



user_hid_sparse <- Hidalgo(X=user_matrix_sparse, nsim = 5000, burn_in = 2500)
post_process_user_hid_sparse <- Hidalgo_postpr_chains(output=user_hid_sparse, all_chains = F)
autoplot(post_process_user_hid_sparse) + ggtitle("Conjugate Prior - Sparse")

post_process_user_hid_sparse$ID_summary



sparse_clusters <- Hidalgo_coclustering_matrix(user_hid_sparse, greed = T)
dense_clusters <- Hidalgo_coclustering_matrix(user_hid, greed = T)



dense_cluster_indices <- dense_clusters$optimalCL
sparse_cluster_indices <- sparse_clusters$optimalCL

dense_cluster_indices <- as.data.frame(dense_cluster_indices)
sparse_cluster_indices <- as.data.frame(sparse_cluster_indices)


df.result <- data.frame(user_id=users_list,
                 dense_cluster_id=as.numeric(dense_cluster_indices$dense_cluster_indices),
                 sparse_cluster_id=as.numeric(sparse_cluster_indices$sparse_cluster_indices))
colnames(df.result) <- c("users", "dense_cluster_id", "sparse_cluster_id")



diff.result <- df.result[df.result$dense_cluster_id != df.result$sparse_cluster_id,]

# Results of users except for users in diff.result
agree.result <- df.result[df.result$dense_cluster_id == df.result$sparse_cluster_id,]


dense_clusters_psm <- dense_clusters$psm
# Set dim names for easier access 
dimnames(dense_clusters_psm) <- list(users_list, users_list)

sparse_clusters_psm <- sparse_clusters$psm
# Set dim names for easier access 
dimnames(sparse_clusters_psm) <- list(users_list, users_list)



# Observing the diff.result we can see that
cluster_3_users_sparse <- df.result[df.result$sparse_cluster_id == 3,]$users
cluster_3_users_dense <- df.result[df.result$dense_cluster_id == 3,]$users
cluster_3_users_dense == cluster_3_users_sparse
# Comparing Dense and Sparse matrix prediction results we see that - 
# 1. Both model iterations return the same number of clusters
# 2. Both model iterations return same members for cluster level 3
# 3. Observing the diff.result we see that both versions predict only slightly different results for cluster 1 and 2
# 4. Study the co-clustering matrix for more information

# Comparing co-clustering matrices of bot sparse and trial

# For each user in diff.result we compare the posterior co-clustering probablities to members in the same cluster


# Separating users for each cluster id.
# Only take take users except the users which have different results.
cluster_1_users <- agree.result[agree.result$sparse_cluster_id == 1,]$users
cluster_2_users <- agree.result[agree.result$sparse_cluster_id == 2,]$users

# 1. Take post probs for predicted cluster users
# 2. Take posterior probs for cluster predicted by the other method
# 3. Sort the scores. 
# 4. Length check to take equal from both lists
# 5. Get Diff.
# 6. Want to compare this diff to do a basic comparision between discriminating power of each method between the two clusters

# selecting an example user and calling it user_prime

count = 0
for (user_prime in diff.result$users){
  temp <- df.result[df.result$users != user_prime,]
  cluster_1_users <- temp[temp$dense_cluster_id == 1,]$users
  cluster_2_users <- temp[temp$dense_cluster_id == 2,]$users
  scores_1 <- sort(dense_clusters_psm[user_prime, cluster_1_users], decreasing = T)
  scores_2 <- sort(dense_clusters_psm[user_prime, cluster_2_users], decreasing = T)
  min_len <- min(length(scores_1), length(scores_2))
  x_m <- mean(abs(scores_1[1:min_len] - scores_2[1:min_len]))
  
  cluster_1_users <- temp[temp$sparse_cluster_id == 1,]$users
  cluster_2_users <- temp[temp$sparse_cluster_id == 2,]$users

  scores_1 <- sort(sparse_clusters_psm[user_prime, cluster_2_users], decreasing = T)
  scores_2 <- sort(sparse_clusters_psm[user_prime, cluster_1_users], decreasing = T)
  min_len <- min(length(scores_1), length(scores_2))
  y_m <- mean(abs(scores_1[1:min_len] - scores_2[1:min_len]))
  
  if(x_m > y_m){
    count <- count + 1
  }
  
}

user_row.data<- read.csv('~/Downloads/user_rows.csv')
dimnames(user_row.data) <- list(user_row.data$users, 1:99)


cluster_3_users_sparse <- df.result[df.result$sparse_cluster_id == 3,]$users
cluster_3_users_dense <- df.result[df.result$dense_cluster_id == 3,]$users

cluster_2_users_sparse <- df.result[df.result$sparse_cluster_id == 2,]$users
cluster_2_users_dense <- df.result[df.result$dense_cluster_id == 2,]$users

cluster_1_users_sparse <- df.result[df.result$sparse_cluster_id == 1,]$users
cluster_1_users_dense <- df.result[df.result$dense_cluster_id == 1,]$users




cluster_3_users_sparse_ <- user_row.data[cluster_3_users_sparse,]
cluster_3_users_dense_ <- user_row.data[cluster_3_users_dense,]

cluster_2_users_sparse_ <- user_row.data[cluster_2_users_sparse,]
cluster_2_users_dense_ <- user_row.data[cluster_2_users_dense,]

cluster_1_users_sparse_ <- user_row.data[cluster_1_users_sparse,]
cluster_1_users_dense_ <- user_row.data[cluster_1_users_dense,]

write.csv(cluster_3_users_sparse_,"cluster_3_users_sparse.csv", row.names = FALSE)
write.csv(cluster_3_users_dense_,"cluster_3_users_dense.csv", row.names = FALSE)

write.csv(cluster_2_users_sparse_,"cluster_2_users_sparse.csv", row.names = FALSE)
write.csv(cluster_2_users_dense_,"cluster_2_users_dense.csv", row.names = FALSE)

write.csv(cluster_1_users_sparse_,"cluster_1_users_sparse.csv", row.names = FALSE)
write.csv(cluster_1_users_dense_,"cluster_1_users_dense.csv", row.names = FALSE)






