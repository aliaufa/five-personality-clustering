import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle

with open('pipeline.pkl', 'rb') as file:
  kmeans = pickle.load(file)

st.set_page_config(
    page_title = 'Five Personality Trait' ,
    layout= 'wide',
    page_icon= '0978Tatsugiri.png'
)

def run():
    # Questions
    questions = {
        'EXT1': 'I am the life of the party.',
        'EXT2': "I don't talk a lot.",
        'EXT3': 'I feel comfortable around people.',
        'EXT4': 'I keep in the background.',
        'EXT5': 'I start conversations.',
        'EXT6': 'I have little to say.',
        'EXT7': 'I talk to a lot of different people at parties.',
        'EXT8': "I don't like to draw attention to myself.",
        'EXT9': "I don't mind being the center of attention.",
        'EXT10': 'I am quiet around strangers.',
        'EST1': 'I get stressed out easily.',
        'EST2': "I am relaxed most of the time.",
        'EST3': 'I worry about things.',
        'EST4': 'I seldom feel blue.',
        'EST5': 'I am easily disturbed.',
        'EST6': 'I get upset easily.',
        'EST7': 'I change my mood a lot.',
        'EST8': "I have frequent mood swings.",
        'EST9': "I get irritated easily.",
        'EST10': 'I often feel blue.',
        'AGR1': 'I feel little concern for others.',
        'AGR2': "I am interested in people.",
        'AGR3': 'I insult people.',
        'AGR4': 'I sympathize with others feelings.',
        'AGR5': "I am not interested in other people's problems.",
        'AGR6': 'I have a soft heart.',
        'AGR7': 'I am not really interested in others.',
        'AGR8': "I take time out for others.",
        'AGR9': "I feel others' emotions.",
        'AGR10': 'I make people feel at ease.',
        'CSN1': 'I am always prepared.',
        'CSN2': "I leave my belongings around.",
        'CSN3': 'I pay attention to details.',
        'CSN4': 'I make a mess of things.',
        'CSN5': 'I get chores done right away.',
        'CSN6': 'I often forget to put things back in their proper place.',
        'CSN7': 'I like order.',
        'CSN8': "I shirk my duties.",
        'CSN9': "I follow a schedule.",
        'CSN10': 'I am exacting in my work.',
        'OPN1': 'I have a rich vocabulary.',
        'OPN2': "I have difficulty understanding abstract ideas.",
        'OPN3': 'I have a vivid imagination.',
        'OPN4': 'I am not interested in abstract ideas.',
        'OPN5': 'I have excellent ideas.',
        'OPN6': 'I do not have a good imagination.',
        'OPN7': 'I am quick to understand things.',
        'OPN8': "I use difficult words.",
        'OPN9': "I spend time reflecting on things.",
        'OPN10': 'I am full of ideas.'
    }

    # Initialize empty lists to store responses
    responses = {key: [] for key in questions}

    # Buat form
    with st.form(key='questionnaire'):

        # Display the form
        st.title('Personality Questionnaire')

        for key, statement in questions.items():
            response_map = {
                'Disagree': 1,
                'Slightly Disagree': 2,
                'Neutral': 3,
                'Slightly Agree': 4,
                'Agree': 5
            }
            response = st.radio(statement, ['Disagree', 'Slightly Disagree', 'Neutral', 'Slightly Agree', 'Agree'])
            responses[key].append(response_map[response])


        submitted = st.form_submit_button('Predict')

    # dataframe

    data_inf = pd.DataFrame(responses)
    st.dataframe(data_inf)
    

    if submitted:
        st.write('# Personality')
        st.write('___')
        # Predict using Linear Regression
        clusters = kmeans.predict(data_inf)
        if clusters == 0:
            st.write('# Tranquil Introverts')
            st.write('Represents individuals who are moderately reserved and inclined towards introspection')
        elif clusters == 1:
            st.write('# Harmony Seekers')
            st.write('Signifies a balanced and open-minded group seeking harmony and new experiences')
        elif clusters == 2:
            st.write('# Versatile Moderates')
            st.write('Describes a flexible and adaptable group with moderate traits across the board')
        elif clusters == 3:
            st.write('# Serene Stoics')
            st.write('Depicts reserved yet emotionally stable individuals with a calm demeanor')
        elif clusters == 4:
            st.write('# Radiant Collaborators')
            st.write('Captures the vibrancy and collaborative nature of socially active and stable individuals.')

        # For Visualization
        questionaire = data_inf.columns.tolist()
        ext = questionaire[0:10]
        est = questionaire[10:20]
        agr = questionaire[20:30]
        csn = questionaire[30:40]
        opn = questionaire[40:50]

        sum_df = pd.DataFrame()
        sum_df['extroversion'] = data_inf[ext].mean(axis=1)
        sum_df['neurotic'] = data_inf[est].mean(axis=1)
        sum_df['agreeable'] = data_inf[agr].mean(axis=1)
        sum_df['conscientious'] = data_inf[csn].mean(axis=1)
        sum_df['open'] = data_inf[opn].mean(axis=1)

        st.write('### Radar Chart')
        st.dataframe(sum_df)

        # Radar chart creation with values
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=sum_df.values.tolist()[0],
            theta=['Extroversion', 'Neurotic', 'Agreeable', 'Conscientious', 'Open'],
            fill='toself'
        ))

        # Update layout
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])), 
            showlegend=False,  
            title='Personality Traits Radar Chart',
        )

        # Display in Streamlit
        st.plotly_chart(fig)

if __name__ == '__main__':
    run()