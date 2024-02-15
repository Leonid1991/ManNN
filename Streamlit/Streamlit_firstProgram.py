import pandas as pd
import streamlit as st

st.write("Here is the first attempt to create a table:")
st.write(pd.DataFrame({
    'first column' : [1,2,3,4],
    'second column': [10, 20, 30, 40]
})) 