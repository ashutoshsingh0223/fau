set.seed(12345)
# Reading the datafile
raw.data <- read.csv('~/Downloads/Data.csv')

dim(raw.data)
head(raw.data)

# Get all the sites
sites <- raw.data$site

unique_sites <- unique(sites)

unique_sites




# First Break the dataframe depending on the `site` column
d0 <- raw.data[raw.data$site == '2/MIRAMEDI3788', ][, 2: 21]

d1 <- raw.data[raw.data$site == "12/M0S5067__0_", ][, 2: 21]
d2 <- raw.data[raw.data$site == "5070MIRALT__0_", ][, 2: 21]



# Using HidAlgo model for data segmentation.
# The labels we have are very coarse, we just have 0 1 or 2 for sites. Our assumption is that we can do a much deeper analysis
# be segmenting the sites based on the the ID of the data we have from the recordings

library("intRinsic")

hid0 <- Hidalgo(X=d0)

dim(hid0$membership_labels)

post_process0 <- Hidalgo_postpr_chains(output=hid0, all_chains = F)

autoplot(post_process0) + ggtitle("Conjugate Prior - 2/MIRAMEDI3788")




hid1 <- Hidalgo(X=d1)


post_process1 <- Hidalgo_postpr_chains(output=hid1, all_chains = F)

autoplot(post_process1) + ggtitle("Conjugate Prior - 12/M0S5067__0_")



hid2 <- Hidalgo(X=d2)


post_process2 <- Hidalgo_postpr_chains(output=hid2, all_chains = F)

autoplot(post_process2) + ggtitle("Conjugate Prior - 5070MIRALT__0_")


