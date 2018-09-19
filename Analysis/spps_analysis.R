library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(gridExtra)
library(ggrepel)
library(rstanarm)
library(lme4)
library(bayesplot)
library(ggmcmc)
library(tidytext)
library(fuzzyjoin)
library(GGally)
library(grid)
df_sources <- read.csv('../processing_analysis/source_scoring/allsides.csv')
df_dict <- read.csv('../output/dict_methods.csv')
df_mentions <- read.csv('../output/mentions.csv', stringsAsFactors = F)
df_men_adj <- read.csv('../output/mentions_adjectives.csv')
df_sent <- read.csv('../output/sentiment.csv')
df_sim <- read.csv('../output/similarity.csv')
df_preds <- read.csv('../output/art_predictions.csv')
df_m_preds <- read.csv('../output/merged_predictions.csv')
df_source_demog <- read.csv('../processing_analysis/source_scoring/sourcedemog.csv')
df_combined_dat <- read.csv('../output/combined_dat.csv')
df_militarized <- read.csv('../output/TF_milclassifier.csv')
df_clean_edges <- read.csv('../output/edgelist_cleaned_renamed.csv', stringsAsFactors = F)
df_mturk <- read.csv('../FergusonMedia/DATA/mturk ratings/media_questions3.csv')
flaxman <- read.csv('../processing_analysis/source_scoring/fgr_dat.csv')
df_mturk <- read.csv('../output/mturk_modeling_data.csv')
df_combined_dat$ArticleID <- str_trim(df_combined_dat$ArticleID)
df_sent$ArticleID <- str_trim(df_sent$ArticleID)
names(df_men_adj)[4] <- 'ent_key'
names(df_mentions)[1] <- 'ArticleID'
options(mc.cores = parallel::detectCores())


df_sources$leaning <- factor(df_sources$leaning, levels(df_sources$leaning)[c(2, 1, 3, 4, 5)])
df_source_demog %>% filter(Demog=='African-American') -> source_demos
names(source_demos)[2] <- 'leaning'
source_info <- rbind(df_sources[,c(1,2)], source_demos)
df_dict <- left_join(df_dict, source_info)

df_mentions %<>% 
  group_by(ArticleID, ent_key) %>%
  mutate(brown = str_detect(ent_text, 'Brown')) %>%
  mutate(wilson = str_detect(ent_text, 'Wilson')) %>%
  ungroup() %>%
  filter(brown==TRUE & wilson==FALSE) %>%
  select(ArticleID, ent_key, brown) %>%
  distinct() %>%
  right_join(df_mentions) %>%
  mutate(brown = ifelse(is.na(brown), FALSE, TRUE))

df_mentions %<>%
  group_by(ArticleID, ent_key) %>%
  mutate(wilson = str_detect(ent_text, 'Wilson')) %>%
  ungroup() %>%
  filter(brown==FALSE & wilson==TRUE) %>%
  select(ArticleID, ent_key, wilson) %>%
  distinct() %>%
  right_join(df_mentions) %>%
  mutate(wilson = ifelse(is.na(wilson), FALSE, TRUE))

names(df_mentions)[8] <- 'Source'

df_mentions <- left_join(df_mentions, source_info)  


###################### estimating source-level metrics for network analysis
## Positivity
df_sent %>% 
  mutate(pos_prop = pos/words,
         neg_prop = neg/words) %>%
  mutate(posneg_diff = pos_prop - neg_prop) %>%
  select(Source, ArticleID, positivity, posneg_ratio, posneg_diff) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_posit <- stan_lmer(positivity~1+(1|Source), prior_intercept = normal(0, 5), 
                     prior=normal(0,5), data=mod.dat)

# m_posit_ratio <- stan_lmer(posneg_ratio~1+(1|Source), prior_intercept = normal(0, 5), 
#                           prior=normal(0,5), data=mod.dat) #undefined for many. not possible to run.

m_posit_diff <- stan_lmer(posneg_diff~1+(1|Source), prior_intercept = normal(0, 5), 
                           prior=normal(0,5), data=mod.dat)


posterior <- as.data.frame(m_posit)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  group_by(Source) %>% 
  summarise(bayesian_est_positivity = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_positivity=mean(positivity, na.rm=T)) -> out.dat

posterior <- as.data.frame(m_posit_diff)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  group_by(Source) %>% 
  summarise(bayesian_est_positivity_diff = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_positivity_diff=mean(posneg_diff, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## subjectivity
df_sent %>% 
  select(Source, ArticleID, subjectivity) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat
mod.dat$subjectivity <- mod.dat$subjectivity*100

m_subj <- stan_lmer(subjectivity~1+(1|Source), prior_intercept = normal(0, 5), 
                    prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_subj)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  group_by(Source) %>% 
  summarise(bayesian_est_subjectivity = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_subjectivity=mean(subjectivity, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## vader_sentiment
df_sent %>% 
  select(Source, ArticleID, vader_compound) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_vader <- stan_lmer(vader_compound~1+(1|Source), prior_intercept = normal(0, 5), 
                     prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_vader)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  group_by(Source) %>% 
  summarise(bayesian_est_vader = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_vader=mean(vader_compound, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## race
df_dict %>% 
  select(Source, ArticleID, race_w2v, race_nodiv_w2v) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_race <- stan_lmer(race_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                    prior=normal(0,5), data=mod.dat)

m_race_nodiv <- stan_lmer(race_nodiv_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                             prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_race)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_race = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_race=mean(race_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

posterior <- as.data.frame(m_race_nodiv)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_race_nodiv = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_race_nodiv=mean(race_nodiv_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## youth
df_dict %>% 
  select(Source, ArticleID, youth_w2v) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_youth <- stan_lmer(youth_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                    prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_youth)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_at(vars(-intercept), funs(estimated_val = (intercept+.))) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_youth = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_youth=mean(youth_w2v, na.rm=T)) %>%
  select(Source, bayesian_est_youth, mean_youth) %>%
  distinct() %>%
  right_join(out.dat) -> out.dat

## ind
df_dict %>% 
  select(Source, ArticleID, ind_w2v, ind_postreview_w2v) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_ind <- stan_lmer(ind_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                   prior=normal(0,5), data=mod.dat)

m_ind_postreview <- stan_lmer(ind_postreview_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                              prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_ind)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_ind = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_ind=mean(ind_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

posterior <- as.data.frame(m_ind_postreview)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_ind_postreview = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_ind_postreview=mean(ind_postreview_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## egal
df_dict %>% 
  select(Source, ArticleID, egal_w2v, egal_postreview_w2v) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_egal <- stan_lmer(egal_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                    prior=normal(0,5), data=mod.dat)

m_egal_postrev <- stan_lmer(egal_postreview_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                    prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_egal)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_egal = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_egal=mean(egal_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

posterior <- as.data.frame(m_egal_postrev)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_egal_postreview = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_egal_postreview=mean(egal_postreview_w2v, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

## concrete
df_dict %>% 
  select(Source, ArticleID, concrete) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_concrete <- stan_lmer(concrete~1+(1|Source), prior_intercept = normal(0, 5), 
                        prior=normal(0,5), data=mod.dat)

posterior <- as.data.frame(m_concrete)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_each(funs(estimated_val = (intercept+.)), -intercept) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  group_by(Source) %>% 
  summarise(bayesian_est_concrete = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_concrete=mean(concrete, na.rm=T)) %>%
  right_join(out.dat) -> out.dat

out.dat %>% 
  ungroup() %>%
  select(Source, bayesian_est_concrete, mean_concrete, bayesian_est_vader,
         mean_vader, bayesian_est_positivity, mean_positivity) %>%
  unique() %>%
  filter(!is.na(mean_concrete)) %>%
  mutate(conc_cat = bayesian_est_concrete>median(bayesian_est_concrete, na.rm=T),
         vader_cat = bayesian_est_vader>median(bayesian_est_vader, na.rm=T),
         liwc_cat = bayesian_est_positivity>median(bayesian_est_positivity, na.rm=T)) %>%
  mutate(conc_vader = paste(conc_cat, vader_cat, sep = '_'),
         conc_liwc = paste(conc_cat, liwc_cat, sep='_')) %>%
  mutate(conc_vader = factor(conc_vader, levels = c('TRUE_TRUE', 'TRUE_FALSE',
                                                    'FALSE_TRUE','FALSE_FALSE'),
                             labels=c('HIGH_HIGH', 'HIGH_LOW', 'LOW_HIGH',
                                      'LOW_LOW')),
         conc_liwc = factor(conc_liwc, levels = c('TRUE_TRUE', 'TRUE_FALSE',
                                                  'FALSE_TRUE', 'FALSE_FALSE'),
                            labels=c('HIGH_HIGH', 'HIGH_LOW', 'LOW_HIGH',
                                     'LOW_LOW'))) %>%
  select(Source, conc_vader, conc_liwc) %>%
  left_join(out.dat) %>%
  select(-ArticleID, -concrete, -egal_w2v, -egal_postreview_w2v, -ind_w2v, 
         -ind_postreview_w2v, -race_w2v, -race_nodiv_w2v, -vader_compound, 
         -subjectivity, -positivity, -posneg_ratio, -posneg_diff) %>%
  filter(!is.na(bayesian_est_concrete)) %>%
  unique() -> out.dat

## cigarillos
df_dict %>% 
  select(Source, ArticleID, cigarillos_w2v) -> mod.dat

df_combined_dat %>%
  select(Source, ArticleID) %>%
  right_join(mod.dat) -> mod.dat

m_cig <- stan_lmer(cigarillos_w2v*10~1+(1|Source), prior_intercept = normal(0, 5), 
                   prior=normal(0,5), data=mod.dat)


posterior <- as.data.frame(m_cig)
names(posterior)[1] <- 'intercept'
posterior %>%
  mutate_at(vars(-intercept), funs(estimated_val = (intercept+.))) %>%
  select(ends_with('estimated_val')) %>%
  gather(estimate, value) %>%
  mutate(estimate = str_replace_all(estimate, '_', '')) %>%
  filter(estimate != 'intercept') %>%
  filter(estimate != 'sigmaestimatedval') %>%
  extract(estimate, into='Source', ':([[:alnum:]]+)\\]') -> plot.dat

plot.dat %>% 
  mutate(value=value/10) %>%
  group_by(Source) %>% 
  summarise(bayesian_est_cig = mean(value)) %>%
  right_join(mod.dat) %>%
  group_by(Source) %>%
  mutate(mean_cig=mean(cigarillos_w2v, na.rm=T)) %>%
  select(Source, bayesian_est_cig, mean_cig) %>%
  ungroup() %>%
  distinct() %>%
  right_join(out.dat) -> out.dat


## source correlations - concrete + valence
df_sent %>% 
  select(Source, ArticleID, positivity, vader_compound) -> mod.dat

df_dict %>%select(Source, ArticleID, concrete) %>%
  left_join(mod.dat) -> mod.dat

mod.dat %>%
  group_by(Source) %>%
  summarise(vader_conc_cor = cor(concrete, vader_compound, use='complete.obs'),
            liwc_conc_cor = cor(concrete, positivity, use='complete.obs')) %>%
  right_join(out.dat) -> out.dat


## Source political orientation
df_mturk <- read.csv('../output/mturk_modeling_data.csv')

m.freq <- stan_lmer(Freq_num~1+(PolBeliefExt|Source) + (1|ResponseId), 
                    prior=normal(0,1), prior_intercept = normal(0,1),
                    prior_covariance = decov(1.5, 1, 1, 1), data=df_mturk)

write.csv(out.dat, '../output/source_estimated_values.csv', row.names = F)

##### computing internal and external edges
df_clean_edges %>%
  rowwise() %>%
  summarise(from = list(rep(from, weight)),
            to = to, count=1) %>%
  unnest() -> out.dat

write.csv(out.dat, '../output/expanded_full.csv', row.names = F)
  
df_clean_edges %>%
  filter(from!=to) %>%
  rowwise() %>%
  summarise(from = list(rep(from, weight)),
            to = to, count=1) %>%
  unnest() -> out.dat
write.csv(out.dat, '../output/expanded_no_internal.csv', row.names = F)

df_clean_edges %>%
  mutate(internal = from==to) %>%
  group_by(from, internal) %>%
  summarise(total = sum(weight)) %>%
  spread(internal, total, fill=0) -> in_v_out

names(in_v_out) <- c('Source', 'external', 'internal')
in_v_out <- left_join(in_v_out, df_source_demog)
names(in_v_out)[4] <- "Category"

print(xtable(in_v_out[,c(1,4,2,3 )], digits = 0), include.rownames=F)

###
df <- read.csv('../output/source_estimated_values.csv')
df_lean <- read.csv('../output/source_estimates_mturk.csv')

df %>%
  left_join(df_lean) -> df

df %>%
  select(Source, bayesian_est_race, liwc_conc_cor, bayesian_est_cig, 
         bayesian_est_youth, bayesian_est_subjectivity, bayesian_est_vader, 
         bayesian_est_positivity, trust_effect) %>%
  filter(!is.na(liwc_conc_cor)) %>%
  distinct() %>%
  select(-Source) -> plot.dat

names(plot.dat) <- c("Race", "Intergroup_Bias", "Crime", "Youth", 
                     "Emotionality", "VADER_Valence", "LIWC_Valence", 'Political_leaning')

p <- ggpairs(plot.dat, 
             upper=list(continuous=wrap('cor', color='black'))) + 
  theme_classic() + 
  theme(axis.text = element_text(size=8))
subplot <- getPlot(p, 8, 3)
subplot <- subplot + scale_x_continuous(breaks=c(.057, .059, .061))
p <- putPlot(p, subplot, 8, 3)
subplot <- getPlot(p, 3, 1)
subplot <- subplot + scale_y_continuous(breaks=c(.057, .059, .061))
p <- putPlot(p, subplot, 3, 1)

ggsave('../output/figs/correl_matrix_vaderbias.jpeg', p, width=15, height=9)



jpeg('../output/figs/correl_matrix_liwcbias.jpeg', width=800, height=600)
print(p, left=.3, bottom=.25)
dev.off()

g <- grid.ls(print=FALSE)
idx <- g$name[grep('text', g$name)]

for(i in idx[1:7]) {
  grid.edit(gPath(i), rot=45, hjust=0.25)
}


jpeg('../output/figs/correl_matrix.jpeg', width=800, height=600)
for(i in idx[1:7]) {
  grid.edit(gPath(i), rot=45, hjust=0.25)
}
dev.off()
  
