
########################################################################################################

import graphlab
import matplotlib

#######################################################################################################

# Explore the data

song_data = graphlab.SFrame('song_data.gl/')

song_data.head()

#graphlab.canvas.set_target('ipynb')

song_data['song'].show()

# Count number of users

users = song_data['user_id'].unique()

len(users)

# Create a song recommender

train_data, test_data = song_data.random_split(.8, seed=0)

# Simple popularity based Recommender

popularity_model = graphlab.popularity_recommender.create(train_data, 
                                                         user_id = 'user_id',
                                                          item_id = 'song')


popularity_model


# Use some Popularity model to make some predictions

print popularity_model.recommend(users = [users[0]])

print popularity_model.recommend(users = [users[1]])

# Build a song recommendation with Personalization

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')

# Applying the Personalized model to make song recommendations

print personalized_model.recommend(users=[users[0]])

print personalized_model.recommend(users=[users[1]])

##song_data

print personalized_model.get_similar_items(['With Or Without You - U2'])

print personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])

# Quantitative comparison between the models

#get_ipython().magic(u'matplotlib inline')
model_performance = graphlab.recommender.util.compare_models(test_data, 
                                                            [popularity_model, personalized_model],
                                                            user_sample=0.05)
print model_performance

print model_performance[:-1]

