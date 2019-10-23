library(ggplot2)
library(reshape)
#script to automatically summarise data from patella-strings python macro

#set session diretory to current file location, we normally do this using session/setwd...
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


python_data_files <- list.files("./05_python_data",full.names = T)

data <- data.frame()
for(i in 1:length(python_data_files)){
  csv <- python_data_files[i]
  t.dat <- read.csv(csv, header = T, stringsAsFactors = F)
  t.out <- data.frame()
  t.name <- basename(csv)
  t.out<- data.frame(substr(t.name,1,nchar(t.name)-4))
  colnames(t.out) <- "im_name"
  if(nchar(unlist(base::strsplit(as.character(t.out$im_name),split = "_"))[5]) == 1){
    t.out$im_name_short <- paste(unlist(base::strsplit(as.character(t.out$im_name),split = "_"))[3:5],collapse = "_")
  }else{
    t.out$im_name_short <- paste(unlist(base::strsplit(as.character(t.out$im_name),split = "_"))[3:4],collapse = "_")
  }
  
  
  t.out$num_strings <- nrow(t.dat)
  
  t.out$'num_strings_<10' <- nrow(subset(t.dat, feret <= 10.0))
  t.out$'num_strings_10-20' <- nrow(subset(t.dat, feret > 10.0 & feret <= 20.0))
  t.out$'num_strings_20-50' <- nrow(subset(t.dat, feret > 20.0 & feret <= 50.0))
  t.out$'num_strings_50+' <- nrow(subset(t.dat, feret > 50.0))
  t.out$'num_strings_20+' <- nrow(subset(t.dat, feret > 20.0))
  t.out$mean_string_feret <- mean(t.dat$feret)
  t.out$mean_string_area <- mean(t.dat$area)
  t.out$mean_string_perimeter <- mean(t.dat$perimeter)
  t.out$sd_string_feret <- sd(t.dat$feret)
  
  data <- rbind(data, t.out)
}


#make data table long-skinny (easier to ggplot with)
data_long <- melt(data, id=c("im_name","im_name_short") )


dat <- subset(data_long, variable == "num_strings")
ggplot(data = dat, aes()) +
  geom_col(aes(x = im_name_short, y = value, fill=im_name_short), 
               col="black", 
               size=.1) +  # change binwidth
  labs(title="Number of strings observed") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))
ggsave("05_R_results/num_strings.png",device = "png")


dat <- subset(data_long, variable == "mean_string_feret")
dat2 <- subset(data_long, variable == "sd_string_feret")
ggplot(data = dat, aes()) +
  geom_col(aes(x = im_name_short, y = value, fill=im_name_short), 
           col="black", 
           size=.1) +  # change binwidth
  geom_point(data = dat2, aes(x = im_name_short, y = value)) +
  labs(title="Mean string feret + sd as points",
       subtit) +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))
ggsave("05_R_results/mean_string_feret.png",device = "png")



dat <- subset(data_long, variable %in% c("num_strings_<10", "num_strings_10-20","num_strings_20-50", "num_strings_50+"))
# Grouped
ggplot(dat, aes(fill=im_name_short, y=value, x=variable)) + 
  geom_bar(position="dodge", stat="identity") +
  labs(title="String size distribution")
ggsave("05_R_results/string_size_dist.png",device = "png")



dat <- subset(data_long, variable == "num_strings_20+")
ggplot(data = dat, aes()) +
  geom_col(aes(x = im_name_short, y = value, fill=im_name_short), 
           col="black", 
           size=.1) +  # change binwidth
  labs(title="Strings longer than 20") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))
ggsave("05_R_results/num_strings_20+.png",device = "png")


