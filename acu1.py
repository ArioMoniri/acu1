#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os 
import streamlit as st
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
#url = "https://github.com/ArioMoniri/acu1/blob/e15fc2a85dc3615de43b0ddda144f4b1cdf36bb4/MED212%20online-program%2023.02.23%5B2872%5D.pdf"
#r_data = pd.read_csv(url)


# In[6]:

# Import Streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import tabula
# Define the app
def app():
    st.title("ACU Class Schedule")
    st.write("Welcome! Please upload your class schedule in PDF format.")
# Create a Streamlit file uploader widget for the PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

name = st.text_input("Enter your name:")
surname = st.text_input("Enter your surname:")
email = st.text_input("Enter your email:")
# Convert the PDF to a DataFrame using the convert_pdf_to_csv function
import streamlit as st
import pandas as pd
import tabula

# Add a title to the app
st.title("PDF to CSV Converter")

# Create a Streamlit file uploader widget for the PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Convert the PDF to a DataFrame using the convert_pdf_to_csv function
if pdf_file is not None:
    try:
        data = tabula.convert_into_df(pdf_file, output_format="csv")
        #st.write(df)
    except:
        st.error("Unable to convert PDF file. Please try again with a different file.")
else:
    st.warning("Please upload a PDF file.")



# Import Module



# In[7]:





# In[8]:


data1 = data.columns.to_frame().T.append(data, ignore_index=True)
data1.columns = range(len(data1.columns))


# In[9]:


coll = int(len(data1.columns))
coll2 = int(len(data1.columns)) - 1
while (coll2 >= 3):
 if (coll > 3):
         data1.iloc[:,2].combine_first(data1.iloc[:,coll2])
 coll2 = coll2 - 1
 


# In[10]:


#data1.iloc[:,2] = datax


# In[11]:


#data1 = data1.drop(
    #labels = [3],
    #axis = 1
#)


# In[12]:


data1.columns = ['TT','Subject','Instructor']
#data1['Date'] = data1.loc[:, "TT"]


# In[13]:


#data1.dropna(
    #axis=0,
    #how = 'all',
    #subset=None,
    #inplace=True
#)


# In[14]:


#data1.fillna("",inplace=True)


# In[15]:


#data1.Date = ''
data1['Date'] = ''


# In[16]:


data1['Date'] = data1.apply(lambda x: ','.join([str(cell) for cell in x if 'DAY' in str(cell)]), axis=1)

for column in data1.columns:
    if column != 'Date':
        data1[column] = data1[column].apply(lambda x: '' if str(x) in data1['Date'].values else x)


# In[17]:


data1.dropna(
    axis=0,
    how = 'all',
    subset=None,
    inplace=True
)


# In[18]:


x=1
y=11
z=0
i=1
xx = int(len(data1)/12) + 1
while i in range(xx):
   
    data1.loc[x:y,'Date']= data1.iloc[z,3]
    x=x+12
    y=y+12
    z=z+12
    i=i+1


# In[19]:


#data1.iloc[0,1]= ''
#data1.iloc[0,0]= ''


# In[20]:


#data2 = data1.replace(np. NaN,'',regex=True)
data1 = data1.applymap(lambda x: '' if pd.isna(x) else '' if 'Unname' in str(x) else x)


# In[21]:


data1[['Date', 'Day']] = data1.Date.str.split(" ", expand = True)


# In[22]:


data1.columns = ['TT','Subject','Instructor','Date','Day']


# In[23]:


data1[['Sart Time', 'End Time']] = data1.TT.str.split(" - ", expand = True)


# In[24]:


z=0
i=1
while i in range(xx):
   
    data1.loc[z,"End Time"]= ''
    z=z+12
    i=i+1


z=0
i=1
while i in range(xx):
   
    data1.loc[z,"Date"]= ''
    z=z+12
    i=i+1

    
z=0
i=1
while i in range(xx):
   
    data1.loc[z,"Day"]= ''
    z=z+12
    i=i+1

z=0
i=1
while i in range(xx):
   
    data1.loc[z,"Subject"]= ''
    z=z+12
    i=i+1


# In[25]:


#data2.fillna("",inplace=True)


# In[26]:


data1.drop('TT', inplace=True, axis=1)


# In[27]:


data1.columns = ['Subject','Description','Start Date','Day','Start Time','End Time']


# In[28]:


data1['End Date'] = data1.loc[:,'Start Date'] 


# In[29]:


data4 = data1.replace(regex=['Study Time'],value= '')


# In[30]:


data4 = data4.replace(regex=['Lunch Time'],value= '')


# In[31]:


#data4.replace('', pd.np.na, inplace=True)


# In[32]:


word_part = 'EXAMINATION'
for index, row in data4.iterrows():

    # Check if the word part is present in any of the cells in the row
    if any(word_part in str(cell) for cell in row.values):

        # If the word part is found, fill all empty cells with the contents of the cell where the word part was found
        for i in range(len(row)):
            if pd.isnull(row[i]):
                row[i] = row[row.astype(str).str.contains(word_part)].iloc[0]


# In[33]:


word_part = 'Medicine'
for index, row in data4.iterrows():

    # Check if the word part is present in any of the cells in the row
    if any(word_part in str(cell) for cell in row.values):

        # If the word part is found, fill all empty cells with the contents of the cell where the word part was found
        for i in range(len(row)):
            if pd.isnull(row[i]):
                row[i] = row[row.astype(str).str.contains(word_part)].iloc[0]


# In[34]:


word_part = 'Exam'
for index, row in data4.iterrows():

    # Check if the word part is present in any of the cells in the row
    if any(word_part in str(cell) for cell in row.values):

        # If the word part is found, fill all empty cells with the contents of the cell where the word part was found
        for i in range(len(row)):
            if pd.isnull(row[i]):
                row[i] = row[row.astype(str).str.contains(word_part)].iloc[0]


# In[35]:


word_part = 'exam'
for index, row in data4.iterrows():

    # Check if the word part is present in any of the cells in the row
    if any(word_part in str(cell) for cell in row.values):

        # If the word part is found, fill all empty cells with the contents of the cell where the word part was found
        for i in range(len(row)):
            if pd.isnull(row[i]):
                row[i] = row[row.astype(str).str.contains(word_part)].iloc[0]


# In[36]:


data4.dropna(
    axis=0,
    how = 'any',
    subset=None,
    inplace=True
)


# In[37]:


word_part = 'THEORETICAL'
if data4.apply(lambda x: x.astype(str).str.contains(word_part).any()).any():
    print(f"The word part '{word_part}' exists in the DataFrame")
else:
    print(f"The word part '{word_part}' does not exist in the DataFrame")
    # pause the code using input() function
    input("Press Enter to continue...")


# In[38]:


data4.Description = data4.Description + '    https://zoom.us/j/95698119943'


# In[39]:


data4 = data4.iloc[1:, :]


# In[40]:


empty_rows = data4.index[data4['Start Date'] == ''].tolist()

# drop rows where cell 'B' is empty
data4 = data4.drop(index=empty_rows)

empty_rows = data4.index[data4['Subject'] == ''].tolist()

# drop rows where cell 'B' is empty
data4 = data4.drop(index=empty_rows)


# In[41]:


#data4 = data4.replace(regex=['02.03.23'],value= '02.03.2023')


# In[42]:


data4['Start Date'] = pd.to_datetime(data4['Start Date'], format='%d.%m.%Y')
data4['Start Date'] = data4['Start Date'].dt.strftime('%m/%d/%Y')


# In[43]:


data4['End Date'] = pd.to_datetime(data4['End Date'], format='%d.%m.%Y')
data4['End Date'] = data4['End Date'].dt.strftime('%m/%d/%Y')


# In[44]:


#data4['End Date'] = pd.to_datetime(data4['End Date'], format='%d.%m.%Y')
#data4['End Date'] = data4['End Date'].dt.strftime('%Y-%m-%d')


# In[45]:


#data4['Start Date'] = pd.to_datetime(data4['Start Date'], format='%d.%m.%Y')
#data4['Start Date'] = data4['Start Date'].dt.strftime('%Y-%m-%d')


# In[46]:


#data4['End Time'] = pd.to_datetime(data4['End Time'], format='%H:%M')
#data4['End Time'] = data4['End Time'].dt.strftime('%HH.%MM')


# In[47]:


#data4['Start Time'] = pd.to_datetime(data4['Start Time'], format='%H:%M')
#data4['Start Time'] = data4['Start Time'].dt.strftime('%HH.%MM')


# In[49]:


# find the indices of rows where 'World' is in the 'Subject' column
indices = data4.index[data4['Subject'].str.contains('English')].tolist()

# empty the cells in the 'Description' column for those rows
data4.loc[indices, 'Description'] = ''

# add the string 'Hello' to the previously emptied cells in the 'Description' column for those rows
data4.loc[indices, 'Description'] = 'https://zoom.us/j/97669955181'


# In[50]:


indices = data4.index[data4['Subject'].str.contains('Study')].tolist()

# empty the cells in the 'Description' column for those rows
data4.loc[indices, 'Subject'] = ''

empty_rows = data4.index[data4['Subject'] == ''].tolist()

# drop rows where cell 'B' is empty
data4 = data4.drop(index=empty_rows)


# In[54]:


data4 = tabula.convert_into_df(data4, output_format="csv")
data4 = pd.DataFrame(data4[1:], columns=data4[0])

# In[52]:


#data4.to_csv(r'c:\\Users\\ASUS\\AppData\\Local\\Programs\\Microsoft VS Code\zahed1.csv', index=False, header=True)


# In[55]:

import streamlit as st
import pandas as pd
import tabula
import icalendar
from datetime import datetime, timedelta
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


# Set up the app
st.set_page_config(page_title="ACU Class Schedule", page_icon=":books:", layout="wide")

# Define a function to convert the DataFrame to an iCalendar file
def create_ics_file(df):
    calendar = icalendar.Calendar()
    calendar.add('prodid', '-//My Reminder Calendar//example.com//')
    calendar.add('version', '2.0')
    for index, row in df.iterrows():
        event = icalendar.Event()
        event.add('summary', row['Subject'])
        event.add('description', row['Description'])
        start_date_str = row['Start Date']
        start_time_str = row['Start Time']
        end_date_str = row['End Date']
        end_time_str = row['End Time']
        start_dt = datetime.strptime(start_date_str + start_time_str, '%m/%d/%Y%H:%M')
        end_dt = datetime.strptime(end_date_str + end_time_str, '%m/%d/%Y%H:%M')
        event.add('dtstart', start_dt)
        event.add('dtend', end_dt)
        alarm = icalendar.Alarm()
        alarm.add('action', 'DISPLAY')
        alarm.add('description', 'Reminder: ' + row['Subject'])
        alarm.add('trigger', timedelta(minutes=-15))
        event.add_component(alarm)
        calendar.add_component(event)
    return calendar.to_ical()

# Define a function to send email
def send_email(email, file_data):
    msg = MIMEMultipart()
    msg['From'] = 'your_email@example.com'
    msg['To'] = email
    msg['Subject'] = 'Schedule'

    text = MIMEText('Please find attached your class schedule in iCalendar format.')
    msg.attach(text)

    file_part = MIMEApplication(file_data)
    file_part.add_header('Content-Disposition', 'attachment', filename='schedule.ics')
    msg.attach(file_part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('Ariorad2020@gmail.com', 'Ariorad2020')
    server.sendmail('Ariorad2020@gmail.com', email, msg.as_string())
    server.quit()


 calcal = create_ics_file(data4)
 send_email(email,calcal)
 


  


