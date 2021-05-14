# Reading the datafile
raw.data <- read.csv('~/Downloads/Data.csv')

dim(raw.data)
head(raw.data)

# Take out data vars with time information
data.vars <- raw.data[, 2:21]



# Using HidAlgo model for data segmentation.
# The labels we have are very coarse, we just have 0 1 or 2 for sites. Our assumption is that we can do a much deeper analysis
# be segmenting the sites based on the the ID of the data we have from the recordings

library("intRinsic")

hid <- Hidalgo(X=data.vars)


post_process <- Hidalgo_postpr_chains(output=hid, all_chains = F)

autoplot(post_process) + ggtitle("Conjugate Prior")
