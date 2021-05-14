# Reading the datafile
raw.data <- read.csv('~/Downloads/Data.csv')

dim(raw.data)
head(raw.data)

# Take out data vars without time information
data.vars_time <- raw.data[, 2:23]
data.vars_time

# First Break the dataframe depending on the `hour` column
d0 <- data.vars_time[data.vars_time$hour == 0, ][, 1:20]
d1 <- data.vars_time[data.vars_time$hour == 1, ][, 1:20]
d2 <- data.vars_time[data.vars_time$hour == 2, ][, 1:20]

d3 <- data.vars_time[data.vars_time$hour == 3, ][, 1:20]
d4 <- data.vars_time[data.vars_time$hour == 4, ][, 1:20]
d5 <- data.vars_time[data.vars_time$hour == 5, ][, 1:20]

d6 <- data.vars_time[data.vars_time$hour == 6, ][, 1:20]
d7 <- data.vars_time[data.vars_time$hour == 7, ][, 1:20]
d8 <- data.vars_time[data.vars_time$hour == 8, ][, 1:20]


d0

d_check <- raw.data[raw.data$hour == 0, ][, 2:21]

d_check
# Using HidAlgo model for data segmentation.
# The labels we have are very coarse, we just have 0 1 or 2 for sites. Our assumption is that we can do a much deeper analysis
# be segmenting the sites based on the the ID of the data we have from the recordings

library("intRinsic")

hid0 <- Hidalgo(X=d0)

dim(hid0$membership_labels)

post_process0 <- Hidalgo_postpr_chains(output=hid0, all_chains = F)

autoplot(post_process0) + ggtitle("Conjugate Prior - Hour 0")




hid1 <- Hidalgo(X=d1)


post_process1 <- Hidalgo_postpr_chains(output=hid1, all_chains = F)

autoplot(post_process1) + ggtitle("Conjugate Prior - Hour 1")



hid7 <- Hidalgo(X=d7)


post_process7 <- Hidalgo_postpr_chains(output=hid7, all_chains = F)

autoplot(post_process7) + ggtitle("Conjugate Prior - Hour 7")


