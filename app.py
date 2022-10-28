
from shiny import App, render, ui
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import linear_model
import csv
from scipy.optimize import minimize_scalar
from functions import *
import matplotlib.style
import matplotlib as mpl
from pathlib import Path
from htmltools import css, div

ratings = pd.read_csv(Path(__file__).parent / "underwood.csv")
worster = pd.read_csv(Path(__file__).parent / "worster.csv")
fbs = pd.read_csv(Path(__file__).parent / "FBS.csv")

hyperparameters = {
    'home_adv': 2.8498763422370907
    , 'decay_rate': 0.0003998040916498869
}


graph_data = get_graph_data(2022, hyperparameters = hyperparameters)
y_max = (graph_data.groupby(['team','hypothetical rating']).sum()).reset_index().groupby(['team'])['error'].mean().reset_index()['error'].max() #determines where the max should be set for the plot

options_teams = fbs.sort_values(by = 'team')['team'].tolist()

app_ui = ui.page_fluid(
    ui.navset_tab_card(
        ui.nav("Underwood",
               ui.row(
                   ui.column(4,
                             ui.output_ui("underwood", container = ui.div, 
                                          style=css(
                                              height="800px",
                                              overflow="auto"
                                          )
                                         )
                            ),
                   ui.column(6,
                             ui.input_selectize("team", "Teams", options_teams, multiple=False),
                             ui.output_plot("stackplot", width = "700px", height = "500px")
                            ),
               )
              ),
        ui.nav("Worster", 
               ui.output_ui("worster", container = ui.div, 
                                          style=css(
                                              height="800px",
                                              overflow="auto"
                                          )
                           )
              )
    )
)
 


def server(input, output, session):
    @output
    @render.plot
    def stackplot():
        
        subject = input.team()

        data = (graph_data[graph_data['team']==subject]
                .sort_values(by = ['week'], axis=0))
        
        cols = (graph_data[graph_data['team']==subject].sort_values(by = ['week'], axis=0))[['week','opponent']].drop_duplicates()['opponent']
        
        pivoted = (data.pivot(index='hypothetical rating', columns='opponent', values=['error'])
                   .reindex(cols, axis=1, level=1))
        
        pivoted.columns = pivoted.columns.droplevel(0)
        pivoted = pivoted.reset_index()
        #mpl.style.use('fivethirtyeight')
        fig = pivoted.plot.area(x='hypothetical rating', ylim = (0, y_max), ylabel = 'Error', xlabel = 'Hypothetical Rating', title = 'Why is my team team rated where it is?')
        line = ratings[ratings['Team']==subject]['Rating'].iloc[0]
        fig.axvline(x=line, color='black', linewidth = 1)
        fig.legend(bbox_to_anchor=(1,1), loc="upper left")
        
        return fig
    
    @output
    @render.table
    def underwood():
        infile = Path(__file__).parent / "underwood.csv"
        df = (pd.read_csv(infile).style
              .format(precision=2)
              .hide_index()
              #.set_properties(**{"border": "1px solid black"})
              .set_table_styles(
                  [{"selector":"tbody tr:nth-child(even)","props":[("background-color","Wheat")]}]
              )
              .set_table_styles(
                    [
                        dict(
                            selector="tbody tr:nth-child(even)", 
                            props=[("background-color","Wheat")]
                        ),
                        dict(
                            selector="th",
                            props=[("min-width", "80px")]
                        ),
                    ]
              )
              .set_sticky(axis="columns")
             )

        return df
    
    @output
    @render.table
    def worster():
        infile = Path(__file__).parent / "worster.csv"
        df = (pd.read_csv(infile).style
              .format(precision=2)
              .hide_index()
              #.set_properties(**{"border": "1px solid black"})
              .set_table_styles(
                  [{"selector":"tbody tr:nth-child(even)","props":[("background-color","Wheat")]}]
              )
              .set_table_styles(
                    [
                        dict(
                            selector="tbody tr:nth-child(even)", 
                            props=[("background-color","Wheat")]
                        ),
                        dict(
                            selector="th",
                            props=[("min-width", "200px")]
                        ),
                    ]
              )
              .set_sticky(axis="columns")
             )

        return df
    
    



app = App(app_ui, server)
