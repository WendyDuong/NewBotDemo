# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import csv
from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction, FormValidationAction
import pandas as pd
import numpy as np
from pandas import Series
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from csv import writer


class FormInfo(Action):

    def name(self) -> Text:
        return "validate_form_info"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: "DomainDict") -> List[
        Dict[Text, Any]]:
        required_slot = ["1_activity", "2_outside", "3_workingtime", "4_environment", "5_company", "6_skills",
                         "7_personally",
                         "8_problems_dealing", "9_public",
                         "10_min_salary", "11_wish_salary"]

        for slot_name in required_slot:
            if tracker.slots.get(slot_name) is None:
                # The slot is not filled yet. Request the user to fill the slot next
                return [SlotSet("required_slot", slot_name)]

        # All slots are filled
        return [SlotSet("required_slot", None)]


class ActionSubmit(Action):

    def name(self) -> Text:
        return "action_submit"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global result
        result = ""

        user_input = ""
        if int(tracker.get_slot("1_activity")[0]) > 2:
            user_input += "Abwechslungsreiche Tätigkeit Herausfordernde Tätigkeit Aufstiegsmöglichkeiten " \
                          "Weiterbildungsmöglichkeiten" + " "
        if int(tracker.get_slot("2_outside")[0]) > 2:
            user_input += "Kundenkontakt Dienstreisen Gute Beziehung zu Vorgesetzten" + " "
        if int(tracker.get_slot("3_workingtime")[0]) > 2:
            user_input += "Flexible Arbeitszeit Überstundenkonto Arbeitszeiterfassung" + " "
        if int(tracker.get_slot("4_environment")[0]) > 2:
            user_input += "Lockere Arbeitsatmosphäre Anbindung an öffentliche Verkehrsmittel Kantine moderne Büro" + " "
        if int(tracker.get_slot("5_company")[0]) > 2:
            user_input += "Innovatives Unternehmen Etabliertes Unternehmen Start-up" + " "
        if int(tracker.get_slot("6_skills")[0]) > 2:
            user_input += "Kommunikationsfähigkeit Organisationsfähigkeit und Teamfähigkeit" + " "
        if int(tracker.get_slot("7_personally")[0]) > 2:
            user_input += "Lernbereitschaft Neugier Selbstdisziplin und analytisches Denkenvermögen" + " "
        if int(tracker.get_slot("8_problems_dealing")[0]) > 2:
            user_input += "Problemlösungskompetenz Kritikfähigkeit Stressresistenz" + " "
        if int(tracker.get_slot("9_public")[0]) > 2:
            user_input += "Integrationsbereitschaft Präsentationsskills und Empathie"

        title = 'User'
        # import input to csv file
        fields = [11, 'User', 'field', user_input]
        with open('test.csv', 'a') as f:
            mwriter = csv.writer(f)
            mwriter.writerow(fields)

        # helper functions
        def get_title_from_index(index):
            return df[df.index == index]["title"].values[0]

        def get_index_from_title(title):
            return df[df.title == title]["index"].values[0]

        ##################################################

        # Step 1: Read CSV File
        df = pd.read_csv("test.csv", encoding='unicode_escape')

        # Step 2: Select Features
        features = ['field', 'jobrequirements']

        # Step 3: Create a column in Dataframe which combines all selected features
        for feature in features:
            df[feature] = df[feature].fillna('')  # fill NA/NaN values

        def combine_features(row):
            return row['field'] + " " + row['jobrequirements']

        df["combined_features"] = df.apply(combine_features, axis=1)  # apply function along columns

        # Step 4: Create count matrix from this new combined column
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])  # transform array -> matrix

        # Step 5: Compute the Cosine Similarity based on the count_matrix
        cosine_sim = cosine_similarity(count_matrix)
        indices = pd.Series(df.index)

        # Step 6: Get index of the jobs from its title
        job_index = get_index_from_title(title)

        similar_jobs = list(enumerate(cosine_sim[job_index]))

        # Step 7: Get a list of similar jobs in descending order of similarity score
        sorted_similar_jobs = sorted(similar_jobs, key=lambda x: x[1], reverse=True)

        # Step 8: Print titles of first similar 5 jobs
        i = 0
        for element in sorted_similar_jobs:
            if 'User' not in get_title_from_index(element[0]):
                result += get_title_from_index(element[0]) + "\n"
            i = i + 1
            if i > 5:
                break

        dispatcher.utter_message(template="utter_slots_values", activity=tracker.get_slot("1_activity"),
                                 outside=tracker.get_slot("2_outside"), workingtime=tracker.get_slot("3_workingtime"),
                                 environment=tracker.get_slot("4_environment"), company=tracker.get_slot("5_company"),
                                 skills=tracker.get_slot("6_skills"),
                                 personally=tracker.get_slot("7_personally"),
                                 problems_dealing=tracker.get_slot("8_problems_dealing"),
                                 public=tracker.get_slot("9_public"),
                                 min_salary=tracker.get_slot("10_min_salary"),
                                 wish_salary=tracker.get_slot("11_wish_salary"),
                                 result=result)

        return []
