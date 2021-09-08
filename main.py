from fastapi import FastAPI
from model_utils import DTCModel

model = DTCModel()

app = FastAPI()

@app.get("/", status_code=200)
async def root():
  return {"message": "Hello World"}

@app.get("/prediction", status_code=200)
async def get_predictions(query: str):
  
  label, response = model.classify_query(query)
  model.recognize_entites(query)
  print("Query = {}, Label = {} Response = {}".format(query, label, response), end='\n')

  pred_response = dict()
  pred_response['message'] = response

  fetch_res = dict()
  fetch_res['news'] = dict()
  fetch_res['news']['present'] = False
  if "NEWS" in model.ent_dict.keys():
    kword = model.extract_keyword()
    loc = model.get_location()
    if loc == 'World':
      loc = 'India'
    fetch_res['news']['present'] = True
    fetch_res['news']['keyword'] = kword
    fetch_res['news']['location'] = loc

  fetch_res['stats'] = dict()
  fetch_res['stats']['present'] = False
  if "STATS" in model.ent_dict.keys():
    # Setting State
    fetch_res['stats']['present'] = True
    # Setting Location
    loc = model.get_location()
    fetch_res['stats']['location'] = loc

  if fetch_res['news']['present'] or fetch_res['stats']['present']:
    fetch_res['hasResources'] = True
  else:
    fetch_res['hasResources'] = False

  print(fetch_res)

  pred_response['fetchRes'] = fetch_res

  query_rec = []
  query_rec.append("What is Covid-19")
  query_rec.append("Symptoms of Covid-19")
  query_rec.append("Is Vaccination Important?")
  
  pred_response['queryRecommendation'] = query_rec

  return pred_response
  


