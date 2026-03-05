import streamlit as st

pages = [
		st.Page('./pages/clusters_page.py', title='Cluster Analysis'),
		st.Page('./pages/predictive_page.py', title='Cluster Prediction'),
    	st.Page('./pages/order_predictive_page.py', title='Order Count Prediction'),
	]
	

pg = st.navigation(pages, position='top')
pg.run()
