set.seed(12345)
# Reading the datafile
raw.data <- read.csv('~/Downloads/Data.csv')

dim(raw.data)
head(raw.data)

# Get all the sites
sites <- raw.data$site

unique_sites <- unique(sites)

unique_sites
#write.csv(unique_sites, '~/Desktop/ashutosh/fau/USI/Bayesian Computing/unique-sites.csv')



# First Break the dataframe depending on the `site` column
d0 <- raw.data[raw.data$site == '2/MIRAMEDI3788', ]
#[, 2: 21]

d1 <- raw.data[raw.data$site == "12/M0S5067__0_", ]
d2 <- raw.data[raw.data$site == "5070MIRALT__0_", ]



# Using HidAlgo model for data segmentation.
# The labels we have are very coarse, we just have 0 1 or 2 for sites. Our assumption is that we can do a much deeper analysis
# be segmenting the sites based on the the ID of the data we have from the recordings

library("intRinsic")


run_hidalgo <- function(site, index, prior="Conjugate", burn_in=1000, nsim=1000) {
  
  d0 <- raw.data[raw.data$site == site, ]
  hid0 <- Hidalgo(X=d0[, 2: 21])
  post_process0 <- Hidalgo_postpr_chains(output=hid0, all_chains = F)
  d0$ID_SUMMARY <- post_process0$ID_summary
  path = paste("/Users/ankitsharma/Downloads/Matlab/", index, sep="")
  write.csv(d0, paste(path, "csv", sep="."), row.names = TRUE)
  autoplot(post_process0) + ggtitle(paste("Conjugate Prior", site, sep="-"))
  return()
}

for (i in 1: length(unique_sites)){

run_hidalgo(site=unique_sites[i], index=i)

}








