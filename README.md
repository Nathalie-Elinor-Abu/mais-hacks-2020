# MBTI Personality Classifier 
[![Build Status](https://travis-ci.org/Nathalie-Elinor-Abu/mais-hacks-2020.svg?branch=master)](https://travis-ci.org/Nathalie-Elinor-Abu/mais-hacks-2020) 

[Try it here!](http://159.203.33.173:5000)

*Winner of best overall hack for the MAIS 2020 Hackathon!*

This project was built for [MAIS Hacks 2020](https://maishacks.com/), a 24 HR virtual AI hackathon. Our goal was to build a simple machine learning model to apply (MBTI) Myers-Briggs Type Indicators to Twitter users based on tweet data gathered from the [Twitter API](https://developer.twitter.com/en/docs). 

Project made with caffeine and tears by [Nathalie](https://github.com/nredick), [Elinor](https://github.com/elinorpd) and [Abu](https://github.com/abubakardaud).

## Project Description

The dataset used to train the model was the [(MBTI) Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasnaek/mbti-type). The model makes a prediction based on data gathered from the users tweets and classifies them into one of the 16 MBTI types. The webapp is built on [Flask](https://flask.palletsprojects.com/en/1.1.x/), [Bootstrap](https://getbootstrap.com/).

## Repository Organization

This repository contains the scripts used preprocess the data, train the model, and deploy the webapp. 

- data/
  - Data in csv format used to train the model.

- model/
  - Contains Python files for the Twitter API, preprocessing data, and the script to build the model.

- static/
  - CSS scripts.

- templates/
  - HTML for the landing pages.

- webscraping/
  - DataCollection/	

- ./ 
  - Files used to deploy the app and a requirements.txt file. 
