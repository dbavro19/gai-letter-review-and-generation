import boto3
import json
import botocore
#import os
#import sys
#import time
#import numpy as np
import streamlit as st

#from requests_aws4auth import AWS4Auth #testing

st.set_page_config(page_title="Appeal and Grievance Assistant ", page_icon=":tada", layout="wide")

#Headers
with st.container():
    st.header("Review and Generate Sample A&G letters ")
    st.subheader("")


#
with st.container():
    st.write("---")
    st.write("### Enter Case Information ###")
    company = st.text_input("Company")
    product = st.text_input("Product")
    issue = st.text_input("Issue Description")
    research = st.text_input("Resolution Research")
    resolution = st.text_input("Resolution Description")
    st.write("---")




#App Logic
#Embed, Search, Invoke LLM etc.



#Invoke LLM - Bedrock for Letter Generation
def generate_letter(bedrock, company, product, issue, research, resolution):


    format="""
    [Dear Member (do not assume a name unless one is provided)],

[Introduction / Why we are reaching out]
[Brief sumary of the issue]
[Summary of the Resaerch and Resolution]
[Any additonal infomration or actions being taken]
[Conclusion and statement about X's commitment to customer serve and care]

Sincerely,
[X Member Services]
    """

    ##Setup Prompt
    prompt_data = f"""
Human:
###
Please help me write a grievance (A&G) resolution letter for my X insurance company X.
The following will be provided to be used as the basis for the generated resolution letter
    Issue Description - a description of the issue/complaint the customer has filed
    Resolution Research - The research we did
    Resolution Description - The Description of the Resolution
    Format - the format your response letter should follow

The generated greivance resolution letter should be friendly, professional, consice, and written in a 5th grade reading level

###
Company: {company}
###
Product: {product}
###
Issue Description: {issue}
###
Resolution Research: {research}
###
Resolution Description: {resolution}
###
Format: {format}
###

Begin the task
Assistant:
"""

    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":2000,
                 "temperature":0,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 

    #Run Inference
    modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body.get('completion')
    print(llmOutput)
    return llmOutput


#Invoke LLM - Bedrock for the Review of Resolution
def review_resolution(bedrock, company, product, issue, research, resolution):

    ##Setup Prompt
    prompt_data = f"""
Human:
###
Based on the information provided below, does the Resolution Description fully address ALL of the concerns or allegations from the member's complaint
If not, what was not addressed?

Be concise with your repsonse. If there are any items that were not addresed please breifly ellaborate on them
Format your response in human readable format, include bullet points where applicable
###
Company: {company}
###
Product: {product}
###
Issue Description: {issue}
###
Resolution Research: {research}
###
Resolution Description: {resolution}


Begin the task
Assistant:
"""
    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":2000,
                 "temperature":0,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body.get('completion')
    print(llmOutput)
    return llmOutput


def review(company, product, issue, research, resolution):

    #bedrock client
    bedrock = boto3.client('bedrock' , 'us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com')
    bedrock.list_foundation_models()

    #invoke LLM
    llmOutput = review_resolution(bedrock, company, product, issue, research, resolution)
    return llmOutput

def generate(company, product, issue, research, resolution):
    #bedrock client
    bedrock = boto3.client('bedrock' , 'us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com')
    bedrock.list_foundation_models()

    #invoke LLM
    llmOutput = generate_letter(bedrock, company, product, issue, research, resolution)
    return llmOutput



##Back to Streamlit

result1=st.button("Review Resolution")
if result1:
    st.write(review(company, product, issue, research, resolution))

result=st.button("Generate Sample Letter")
if result:
    st.write(generate(company, product, issue, research, resolution))

