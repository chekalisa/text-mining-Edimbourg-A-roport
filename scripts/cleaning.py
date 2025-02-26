import pandas as pd

def cleaning(df):
    
    ''' this function makes some pretreatment, especially fills NaN variables
    
    Arguments
      df : data frame

    Returns : cleaned data frame
    '''

    #Drop duplications if the dataset has any
    df.drop_duplicates(keep = 'first', inplace = True)
 
    #Datetime
    
    df['Start Date (UTC)'] = pd.to_datetime(df['Start Date (UTC)'], format="%d/%m/%Y %H:%M")
    df['Submit Date (UTC)'] = pd.to_datetime(df['Submit Date (UTC)'], format="%d/%m/%Y %H:%M")
    
    # What time (roughly) did you arrive at Edinburgh Airport for your *departing *flight -> missing values are because the customers only arrived, not departed
    # We change to 0?
    df["What time (roughly) did you arrive at Edinburgh Airport for your *departing *flight"] = df["What time (roughly) did you arrive at Edinburgh Airport for your *departing *flight"].fillna(0) 
    
    #What country were you travelling to? -> same problem
    #replace by 'No destination'
    df["What country were you travelling to? "] = df["What country were you travelling to? "].fillna("No destination")

    #Why were you travelling? 
    #replace by 'Others'
    df["Why were you travelling? "] = df["Why were you travelling? "].fillna("Others")

    #What country do you live in? 
    #replace by 'Others'
    df["What country do you live in? "] = df["What country do you live in? "].fillna("Others")

    #they didn't answer since they travel alone
    df["Including yourself, how many people were in your travel party?"] = df["Including yourself, how many people were in your travel party?"].fillna(0)

    df["Do you have any other feedback or suggestions?"] = df["Do you have any other feedback or suggestions?"].fillna("No")
    
    df["Are there any other premium services that you would like to see introduced at Edinburgh?"] = df["Are there any other premium services that you would like to see introduced at Edinburgh?"].fillna("No")
    
    def fillna(df, variable):
         df[variable] = df[variable].apply(lambda x: 0 if pd.isna(x) else 1)


    #Family restaurant, Grab & Go, Premium bar, Quick service restaurant: they didn't answer as they didn't use these services
    fillna(df,'Family restaurant') 
    fillna(df, 'Grab & Go')
    fillna (df, 'Premium bar')
    fillna(df,'Quick service restaurant')

    #"It was easy to get to / from the airport" options, "0" if they didn't choose this option
    fillna(df,'It was hard to find the car park')
    fillna(df,'Not enough transport options')
    fillna (df, 'It was hard to find my way to the terminal') 
    fillna(df, 'Something else')

    #"It was easy to find information about my flight" options, "0" if they didn't choose this option
    fillna(df, 'Not enough announcements')
    fillna(df, 'Could not find flight information screens')
    fillna(df, 'Flight information screens did not have the right info')
    fillna(df, 'No one around to help')
    fillna(df, 'Something else.1')
    
    #"It was easy to find my way around the airport" options
    fillna(df, 'Not enough signs')
    fillna(df, 'Signs were confusing')
    fillna(df, 'Too many signs')
    fillna(df, 'No one around to direct me')
    fillna(df, 'Something else.2')

    #"I was satisfied with the staff service I received" options
    fillna(df, 'Staff were unable to answer my question')
    fillna(df, "Staff were unhelpful/rude")
    fillna(df, "I couldn't find anyone to talk to")
    fillna(df, 'Something else.3')

    #"I was satisfied with the cleanliness of the airport" options 
    fillna(df, 'Litter')
    fillna(df, 'Bins not emptied')
    fillna(df, 'Dirty washrooms')
    fillna(df, 'Foul odour')
    fillna(df, 'Something else.4')
    
    #"It was easy to find a seat at the airport" options
    fillna(df, 'Not enough seats before Security')
    fillna(df, 'Not enough seats at restaurants')
    fillna(df, 'Not enough seats at gates')
    fillna(df, 'Something else.5')

    #"It was easy to recycle at Edinburgh Airport" options
    fillna(df, "Couldn't find recycling bin")
    fillna(df, 'Bin labelling was unclear')
    fillna(df, 'Not enough waste separation')
    fillna(df, 'Something else.6')

    df.rename(columns=lambda x: x.replace('"', ''), inplace=True)

    
def prepare_dataframe(df):
    '''
    This function renames the columns, fills NaN values in 'feedback_suggestions' column with 'No'
    It also filters the DataFrame to include only rows where 'overall_satisfaction_binary' is 'No' and selects relevant columns
    So, the goal is to analyze the suggestions of people who weren't satisfied with their experience

    Arguments:
    df : the DataFrame that contains feedback and satisfaction information

    Returns:
    DataFrame : a filtered clean DataFrame 
    '''
    df.rename(columns={'Do you have any other feedback or suggestions?':'feedback_suggestions'}, inplace=True)
    df['feedback_suggestions'] = df['feedback_suggestions'].fillna("No")
    df.rename(columns={'Overall, were you satisfied with your most recent experience at Edinburgh Airport?  ':"overall_satisfaction_binary"}, inplace=True)
    
    df = df[df['overall_satisfaction_binary'] == 'No']
    df = df[['overall_satisfaction_binary', 'feedback_suggestions']]
    
    return df
