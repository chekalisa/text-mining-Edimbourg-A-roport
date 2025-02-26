import pandas as pd


def transformation (df):
   '''
   transformation of categories
   '''

   df['Overall, were you satisfied with your most recent experience at Edinburgh Airport?  '].replace({'Yes': '1', 'No': '0'}, inplace=True)

   df["When do you think you will fly next from Edinburgh Airport?"].replace({'Within* 6 months *to* 1 year *':'6-12', 'Within *3-6 months *':'3-6', 'Within *1-3 months*':'1-3', 'More than* 1 year from *now':'>12'}, inplace=True)

   df["How did you get to and/or from the airport?"].replace({'I drove': 'Car', 'I took the bus': 'Bus', 'I took the tram': 'Tram', 'I was dropped off by friends/family':'Drop-off', 'I took a taxi':'Taxi', 'I took an uber':'Uber', 'I got the train':'Train'}, inplace=True)


   #transformation of variables with modalities : Very unlikely, unlikely etc...
   def treat_likely(df, variable):
       df[variable].replace({'Very unlikely':'1', 'Unlikely':'2','Somewhat likely':'3', 'Likely':'4', 'Very likely':'5'}, inplace=True)

   treat_likely(df,"Meet and greet / airport concierge service")
   treat_likely(df,"Luggage porterage")
   treat_likely(df,"Private terminal with airside vehicle collection/pick-up")
   treat_likely(df,"Home collection baggage services")
   treat_likely(df, "Priority queuing")

   #transformation of variables with modalities : Strongly disagree, disagree etc....
   def treat_agree(df, variable):
       df[variable].replace({'Strongly disagree':'1', 'Disagree':'2','Somewhat disagree':'3','Somewhat agree':'4', 'Agree':'5', 'Strongly agree':'6'}, inplace=True)

   treat_agree(df,"It was easy to get to / from the airport")
   treat_agree(df,"It was easy to find information about my flight")
   treat_agree(df,"It was easy to find my way around the airport")
   treat_agree(df,"I was satisfied with the staff service I received")
   treat_agree(df,"I was satisfied with the cleanliness of the airport")
   treat_agree(df,"It was easy to find a seat at the airport")
   treat_agree(df,"It was easy to recycle at Edinburgh Airport")
   
   df.rename(columns={'Overall, were you satisfied with your most recent experience at Edinburgh Airport?  ': 'overall_satisfaction_binary'}, inplace=True)
   df.rename(columns={'What country do you live in? ':'residency_country'}, inplace=True)
   df.rename(columns={'What age group do you fall into?  ':'age'}, inplace=True)
   df.rename(columns={'Why were you travelling? ':'travelling_reason'}, inplace=True)
   df.rename(columns={"Do you have any other feedback or suggestions?" : "feedbacks"}, inplace=True)
   df.rename(columns={"Are there any other premium services that you would like to see introduced at Edinburgh?" : "services"}, inplace=True)