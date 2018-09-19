el <- read.csv("edgelist.csv")
Replace<-read.csv('source_replace.csv')

el$to<-tolower(el$to)

df<-merge(el, Replace, by.x="to", by.y="link", all=TRUE)
df$to <- ifelse(!is.na(df$name), as.character(df$name), as.character(df$to))
df <- subset(df, select=c(from, to, weight))

write.csv(df, "edgelist_cleaned_renamed.csv", row.names=FALSE)