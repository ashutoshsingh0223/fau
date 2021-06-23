# Reading the datafile

base_path <- "~/Downloads/ashutosh/fau/USI/Bayesian Computing/project/"



set.seed(12345)
raw.data <- read.csv(paste(base_path,'results-20210426-141638.csv', sep = ''))

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
# Calculate sum of each row to convert transition scores into transition probabilities
row.sum <- rowSums(transition_matrix)
for (event in events){
  transition_matrix[event,] <- transition_matrix[event,] / row.sum[event]
}

transition_matrix['login',]

# Import Markov Chain Library
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


##### Augmentation Step 2: Adding a time dependency
# Accounting for: How frequent are user's visits
# In the series of events there are different time intervals for different users(Scale is not uniform)
### Future work



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
# This matrix will be used for sparse trail
user_matrix_sparse <- user_matrix

# Now filling the empty values for events on basis on values from stationary distribution of the events
# This modified matric will be used for dense-trail
for (user in users){
  for (event in events){
    if(user_matrix[user, event] == 0){
      user_matrix[user, event] <- limiting.dist[event][1,1]
    }
  }

}

user_matrix_sparse_original <- user_matrix
user_matrix_original <- user_matrix
# Filtering out duplicate users and storing them
dups <- as.data.frame(duplicated(user_matrix_sparse_original))

# Finding users with different behaviour on website for dense and sparse-trail data
user_matrix <- unique(user_matrix)
user_matrix_sparse <- unique(user_matrix_sparse)

# Getting a list of users with distics behaviours
users_list <- row.names(user_matrix)


#### Dense Trial
# Running Hidalgo with a conjugate prior
user_hid <- Hidalgo(X=user_matrix, nsim = 5000, burn_in = 2500)
post_process_user_hid <- Hidalgo_postpr_chains(output=user_hid, all_chains = F)
autoplot(post_process_user_hid) + ggtitle("Conjugate Prior - Dense")

# Getting optimal clusters using mean for VI algo
dense_clusters <- Hidalgo_coclustering_matrix(user_hid, greed = T)
dense_cluster_indices <- dense_clusters$optimalCL
# Storing as dataframe
dense_cluster_indices <- as.data.frame(dense_cluster_indices)

# Getting the posterior co-clustering matrix
dense_clusters_psm <- dense_clusters$psm
# Set dim names for easier access 
dimnames(dense_clusters_psm) <- list(users_list, users_list)



#### Sparse Trial
user_hid_sparse <- Hidalgo(X=user_matrix_sparse, nsim = 5000, burn_in = 2500)
post_process_user_hid_sparse <- Hidalgo_postpr_chains(output=user_hid_sparse, all_chains = F)
autoplot(post_process_user_hid_sparse) + ggtitle("Conjugate Prior - Sparse")

# Getting optimal clusters using mean for VI algo
sparse_clusters <- Hidalgo_coclustering_matrix(user_hid_sparse, greed = T)
sparse_cluster_indices <- sparse_clusters$optimalCL
# Storing as dataframe
sparse_cluster_indices <- as.data.frame(sparse_cluster_indices)

# Getting the posterior co-clustering matrix
sparse_clusters_psm <- sparse_clusters$psm
# Set dim names for easier access 
dimnames(sparse_clusters_psm) <- list(users_list, users_list)




### Storing result. This result dataframe contains only distinct 108 users(not 136)
df.result <- data.frame(user_id=users_list,
                 dense_cluster_id=as.numeric(dense_cluster_indices$dense_cluster_indices),
                 sparse_cluster_id=as.numeric(sparse_cluster_indices$sparse_cluster_indices))

colnames(df.result) <- c("users", "dense_cluster_id", "sparse_cluster_id")


### Storing final results and saving to disk
library(lsa)
dup_users <- c()
dup_dense_cluster <- c()
dup_sparse_cluster <- c()
for (user in df.result$users){
    for (user_ in users){
      if(cosine(user_matrix_original[user_, ], user_matrix_original[user, ]) == 1){
        if(user != user_){
          dup_users <- c(dup_users, user_)
          dup_dense_cluster <- c(dup_dense_cluster, df.result[df.result$users == user, ]$dense_cluster_id[1])
          dup_sparse_cluster <- c(dup_sparse_cluster, df.result[df.result$users == user, ]$sparse_cluster_id[1])
        }
      }
    } 
  
}

# Storing final results. Last 28 elememts are duplicated users
df.result.final <- data.frame(user_id=c(users_list, dup_users),
                        dense_cluster_id=as.numeric(c(dense_cluster_indices$dense_cluster_indices, dup_dense_cluster)),
                        sparse_cluster_id=as.numeric(c(sparse_cluster_indices$sparse_cluster_indices, dup_sparse_cluster)))
write.csv(df.result.final, paste(base_path,'results/Final_Result.csv', sep = ''))



##### Comparing results

# Storing data for users for different results in dense and sparse trail
diff.result <- df.result[df.result$dense_cluster_id != df.result$sparse_cluster_id,]

# Results of users except for users in diff.result
agree.result <- df.result[df.result$dense_cluster_id == df.result$sparse_cluster_id,]





# Observing the diff.result we can see that
cluster_3_users_sparse <- df.result[df.result$sparse_cluster_id == 3,]$users
cluster_3_users_dense <- df.result[df.result$dense_cluster_id == 3,]$users
cluster_3_users_dense == cluster_3_users_sparse
# Comparing Dense and Sparse matrix prediction results we see that - 
# 1. Both model iterations return the same number of clusters
# 2. Both model iterations return same members for cluster level 3
# 3. Observing the diff.result we see that both versions predict only slightly different results for cluster 1 and 2
# 4. Study the co-clustering matrix for more information



##### Comparing co-clustering matrices of both sparse and dense trial ####

# For each user we calculate discriminative or segmenting power based posterior co-clustering probabilities with the following steps

# 1. For each user
# 2. For dense trial
#    2.1  Get predicted_cluster_id for the user
#    2.2. Get top-5 neighbours within its own cluster(predicted_cluster_id) using the co-clustring matrix
#    2.3.  Calculate mean of scores of these neighbours as `mean_score_of_top_neighbours_same_cluster`
#    2.4.  Initialize `mean_scores_of_top_neighbours_other_clusters` <- c()
#    2.5. for `cluster` in  clusters != predicted_cluster_id
#         2.5.1. Get top-5 neighbours of user from this `cluster` from co-clustering matrix
#         2.5.2. Calculate mean of scores from previous step and store in `mean_scores_of_top_neighbours_other_clusters`
#    2.6  calculate segmenting_power_dense <- abs(max(mean_scores_of_top_neighbours_other_clusters) -  mean_score_of_top_neighbours_same_cluster)
# 3. For sparse trial
#    3.1  Get predicted_cluster_id for the user
#    3.2. Get top-5 neighbours within its own cluster(predicted_cluster_id) using the co-clustring matrix
#    3.3  Calculate mean of scores of these neighbours as `mean_score_of_top_neighbours_same_cluster`
#    3.4  Initialize `mean_scores_of_top_neighbours_other_clusters` <- c()
#    3.5. for `cluster` in  clusters != predicted_cluster_id
#         3.5.1. Get top-5 neighbours of user from this `cluster` from co-clustering matrix
#         3.5.2. Calculate mean of scores from previous step and store in `mean_scores_of_top_neighbours_other_clusters`
#    3.6  calculate segmenting_power_sparse <- abs(max(mean_scores_of_top_neighbours_other_clusters) -  mean_score_of_top_neighbours_same_cluster)
# 4. Compare segmenting_power_sparse and segmenting_power_dense
# 5. Increment count if segmenting_power_dense > segmenting_power_sparse


clusters = c(1, 2, 3)
count <- 0

for (user_prime in df.result$users){


  temp <- df.result[df.result$users != user_prime,]
  predicted_cluster_id <- df.result[df.result$users == user_prime,]$dense_cluster_id
  other_clusters <- clusters[clusters != predicted_cluster_id]
  predicted_cluster_users <- temp[temp$dense_cluster_id == predicted_cluster_id,]$users
  
  mean_score_of_top_neighbours_same_cluster <- mean(sort(dense_clusters_psm[user_prime, predicted_cluster_users], decreasing = T)[1: 5])
  mean_scores_of_top_neighbours_other_clusters <- c()

  for (cluster_id in other_clusters){
      cluster_users <- temp[temp$dense_cluster_id == cluster_id,]$users
      co_clustering_scores_mean <- mean(sort(dense_clusters_psm[user_prime, cluster_users], decreasing = T)[1: 5])
      mean_scores_of_top_neighbours_other_clusters <- c(mean_scores_of_top_neighbours_other_clusters, co_clustering_scores_mean)
  }
  mean_scores_of_top_neighbours_other_clusters
  mean_score_of_top_neighbours_same_cluster
  segmenting_power_dense <- abs(max(mean_scores_of_top_neighbours_other_clusters) -  mean_score_of_top_neighbours_same_cluster)
  


  predicted_cluster_id <- df.result[df.result$users == user_prime,]$sparse_cluster_id
  other_clusters <- clusters[clusters != predicted_cluster_id]
  predicted_cluster_users <- temp[temp$sparse_cluster_id == predicted_cluster_id,]$users

  mean_score_of_top_neighbours_same_cluster_sparse <- mean(sort(sparse_clusters_psm[user_prime, predicted_cluster_users], decreasing = T)[1: 5])
  mean_scores_of_top_neighbours_other_clusters <- c()
  for (cluster_id in other_clusters){
      cluster_users <- temp[temp$sparse_cluster_id == cluster_id,]$users
      co_clustering_scores_mean <- mean(sort(sparse_clusters_psm[user_prime, cluster_users], decreasing = T)[1: 5])
      mean_scores_of_top_neighbours_other_clusters <- c(mean_scores_of_top_neighbours_other_clusters, co_clustering_scores_mean)
  }
  mean_scores_of_top_neighbours_other_clusters
  mean_score_of_top_neighbours_same_cluster_sparse
  segmenting_power_sparse <- abs(max(mean_scores_of_top_neighbours_other_clusters) -  mean_score_of_top_neighbours_same_cluster_sparse)

  if (segmenting_power_dense >= segmenting_power_sparse){
    count <- count + 1
}
  


}

count


  











######## Getting out data for manual user analysis
user_row.data<- read.csv(paste(base_path, 'user_rows.csv', sep = ''))
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

write.csv(cluster_3_users_sparse_,paste(base_path, "cluster_3_users_sparse.csv", sep=''), row.names = FALSE)
write.csv(cluster_3_users_dense_,paste(base_path, "cluster_3_users_dense.csv", sep=''), row.names = FALSE)

write.csv(cluster_2_users_sparse_,paste(base_path, "cluster_2_users_sparse.csv", sep=''), row.names = FALSE)
write.csv(cluster_2_users_dense_,paste(base_path, "cluster_2_users_dense.csv", sep=''), row.names = FALSE)

write.csv(cluster_1_users_sparse_,paste(base_path, "cluster_1_users_sparse.csv", sep=''), row.names = FALSE)
write.csv(cluster_1_users_dense_,paste(base_path, "cluster_1_users_dense.csv", sep=''), row.names = FALSE)






